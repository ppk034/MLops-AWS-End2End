import xgboost as xgb
import pandas as pd

# Load the trained model
model = xgb.Booster()
model.load_model("xgboost_model.json")

# Sample new data (replace with actual input values)
sample_data = pd.DataFrame({
    'VendorID': [1],  # Add this to match the training data
    'passenger_count': [1],
    'trip_distance': [2.5],
    'pickup_longitude': [-73.985],
    'pickup_latitude': [40.758],
    'RateCodeID': [1],
    'store_and_fwd_flag': [0],  # Use 0 for 'N' and 1 for 'Y'
    'dropoff_longitude': [-73.965],
    'dropoff_latitude': [40.765],
    'payment_type': [1],
    'fare_amount': [10.5],
    'extra': [0.5],
    'mta_tax': [0.5],
    'tip_amount': [2.0],
    'tolls_amount': [0.0],
    'improvement_surcharge': [0.3],
    'pickup_timestamp': [1638345600],  # Example timestamp for pickup datetime
    'dropoff_timestamp': [1638347700]  # Example timestamp for dropoff datetime
})

# Convert sample data to DMatrix format
dmatrix = xgb.DMatrix(sample_data)

# Make predictions on the new data
# Make predictions on the new data
prediction = model.predict(dmatrix)

# Print the prediction
print(f"Predicted total amount: {prediction[0]}")