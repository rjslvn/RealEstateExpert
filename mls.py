# Import necessary libraries
from fuzzywuzzy import fuzz
from geopy.geocoders import Nominatim
from geopy.distance import geodesic
import pandas as pd
import requests
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import datetime

# API settings
BASE_URL = "https://api.mlsgrid.com/v2/Property"
HEADERS = {
    "Authorization": "Bearer 80c2ab3c769932b687dfa0e7d7c27398c25395db",
    "Accept": "application/json"
}

# Define fuzzy match function
def is_match(input_address, full_address, threshold=80):
    return fuzz.token_set_ratio(input_address.lower(), full_address.lower()) >= threshold

# Fetch data from the API
def fetch_data():
    df = pd.DataFrame()
    url = BASE_URL
    FILTER_PARAM = "OriginatingSystemName eq 'nwmls'"
    params = {
        "$filter": FILTER_PARAM,
        "$top": 100
    }
    while True:
        response = requests.get(url, headers=HEADERS, params=params)
        data = response.json()
        if "value" in data:
            df = pd.concat([df, pd.DataFrame(data["value"])], ignore_index=True)
        if "@odata.nextLink" in data:
            url = data["@odata.nextLink"]
        else:
            break
    return df

# Geocode properties and calculate distances
def geocode_properties(df, entered_coords):
    geolocator = Nominatim(user_agent="myGeocoder")
    active_listings = df[df['StandardStatus'] == 'Active']
    active_listings['Distance'] = None

    for idx, listing in active_listings.iterrows():
        location = geolocator.geocode(listing['UnparsedAddress'])
        if location is not None:
            listing_coords = (location.latitude, location.longitude)
            distance = geodesic(entered_coords, listing_coords).miles
            active_listings.loc[idx, 'Distance'] = distance

    if not active_listings['Distance'].isnull().all():
        active_listings = active_listings.sort_values('Distance')

    return active_listings

# Clean and process the data
def clean_and_process_data_base(file_path):
    # Load your DataFrame
    df = pd.read_csv(file_path)

    # Convert dates to pandas datetime format
    DATE_COLUMNS = ['List Date', 'Sold Date']
    for column in tqdm(DATE_COLUMNS, desc='Processing date columns'):
        df[column] = pd.to_datetime(df[column]).astype(np.int64) // 10**9

    # Drop the address column
    df = df.drop(['Address'], axis=1)

    # Get list of non-numeric columns
    non_numeric_columns = df.select_dtypes(include=['object']).columns.tolist()

    le = LabelEncoder()
    scaler = MinMaxScaler()

    # Convert non-numeric columns to numeric
    for column in tqdm(non_numeric_columns, desc='Processing non-numeric columns'):
        if column not in DATE_COLUMNS:
            df[column] = df[column].fillna('Unknown')
            df = pd.concat([df, pd.get_dummies(df[column], prefix=column)], axis=1)
            df[column] = le.fit_transform(df[column].astype(str))
        if len(df[column].unique()) > 1:
            df[column] = scaler.fit_transform(df[column].values.reshape(-1, 1))
        else:
            df[column] = 0

    df = df.drop(non_numeric_columns, axis=1)

    # Remove columns that only contain NaN values
    df = df.dropna(how='all', axis=1)

    # Fill NA/NAN values in dataframe
    df.fillna(0, inplace=True)
    if df.isnull().values.any():
        print("Still contains NaN values.")
    else:
        print("No NaN values in dataframe.")
    
    # Separate target from predictors
    y = df['List Price']
    X = df.drop('List Price', axis=1)

    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define the model
    gbm = GradientBoostingRegressor()

    # Fit model
    print("Fitting model. Start time:", datetime.datetime.now())
    gbm.fit(X_train, y_train)
    print("Model fitting complete. End time:", datetime.datetime.now())

    # Predict test set
    y_pred = gbm.predict(X_test)

    # Calculate and print MSE and R2 Score
    print(f"Mean Squared Error (MSE): {mean_squared_error(y_test, y_pred)}")
    print(f"R-squared: {r2_score(y_test, y_pred)}")

    # Tune the hyperparameters of the GBM model
    PARAM_GRID = {
        "n_estimators": [100, 200, 300],
        "learning_rate": [0.1, 0.01, 0.001],
        "max_depth": [4, 5, 6]
    }
    GRID = GridSearchCV(gbm, PARAM_GRID, cv=5)

    # Fit and tune model
    print("Tuning model. Start time:", datetime.datetime.now())
    GRID.fit(X_train, y_train)
    print("Model tuning complete. End time:", datetime.datetime.now())

    # Evaluate the tuned GBM model
    gbm_tuned = GRID.best_estimator_
    y_pred_tuned = gbm_tuned.predict(X_test)
    
    # Create a DataFrame for the model's performance metrics
    metrics = pd.DataFrame({
        "Metric": ["Tuned Mean Squared Error (MSE)", "Tuned R-squared"],
        "Value": [mean_squared_error(y_test, y_pred_tuned), r2_score(y_test, y_pred_tuned)]
    })
    print("Model Performance Metrics:")
    print(metrics)

    # Predict for the first 20 properties
    sample_properties = X_test[:20]
    sample_predictions = gbm.predict(sample_properties)
    
    # Create a DataFrame for the sample predictions
    predictions = pd.DataFrame({
        "Property Index": sample_properties.index,
        "Predicted Price": sample_predictions
    })
    print("Predictions for First 20 Properties:")
    print(predictions)
    
    # Create a DataFrame for the model's features
    features = pd.DataFrame({
        "Features": X.columns
    })
    print("Features Used by the Model:")
    print(features)

clean_and_process_data_base('CMA_Plus.csv')
