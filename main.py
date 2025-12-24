import pandas as pd 
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error , r2_score

# load the data

try:
    housing = fetch_california_housing(as_frame=True)
    df = housing.frame
except Exception as e:
    print(f"Could not fetch data from remote source: {e}")
    print("Using synthetic data instead...")
    df = pd.DataFrame({
        'MedInc': np.random.rand(20640) * 15,
        'HouseAge': np.random.rand(20640) * 52,
        'AveRooms': np.random.rand(20640) * 10,
        'AveBedrms': np.random.rand(20640) * 5,
        'AveOccup': np.random.rand(20640) * 1000,
        'Latitude': np.random.rand(20640) * 42 + 32,
        'Longitude': np.random.rand(20640) * 24 - 125,
        'MedHouseVal': np.random.rand(20640) * 5
    })

print("California Housing Dataset: ")
print(df.head()) 

# featurs and target variable 
X = df.drop('MedHouseVal', axis=1)
y = df['MedHouseVal']

# split the data into training and testing sets 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#train linear model 
model = LinearRegression()
model.fit(X_train, y_train)

#  make predictions on the test set
y_pred = model.predict(X_test)

# evaluate the model 
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")

#model coeffients 
print("Model Coefficients:", model.coef_)
print("Model Intercept:", model.intercept_) 

model_coef = pd.DataFrame(model.coef_, X.columns, columns=['Coefficient'])
print(model_coef)

# test model with new data 

new_data = pd.DataFrame({
  'MedInc': [8.3252],
  'HouseAge': [41.0],
  'AveRooms': [6.98412698],
  'AveBedrms': [1.02380952],
  'AveOccup': [322.0],
  'Latitude': [37.88],
  'Longitude': [-122.23]
})

predictPrice = model.predict(new_data)
print(f"Predicted House Price: {predictPrice[0]}")