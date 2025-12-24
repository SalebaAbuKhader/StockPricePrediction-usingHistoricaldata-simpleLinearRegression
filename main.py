import pandas as pd
from sklearn.model_selection import  train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error ,  r2_score
import matplotlib.pyplot as plt

# sample historical stock [price]

data = {
    'Date': pd.date_range(start='2023-01-01', periods=10, freq='D'),
    'Price': [100, 102, 105, 103, 107, 110, 112, 115, 118,111]
}
df = pd.DataFrame(data)

df['Date'] = df['Date'].map(pd.Timestamp.toordinal)

print("stock price data:")
print(df.head())

# features and target variable 

X = df[['Date']]
y = df['Price'] 

# split the data into training and testing sets 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) 

# create and train the linear regression model 
model = LinearRegression() 
model.fit(X_train, y_train)
# make predictions on the test set
y_pred = model.predict(X_test) 
# evaluate the model
mse = mean_squared_error(y_test, y_pred) 
r2 = r2_score(y_test, y_pred) 

print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")

plt.figure(figsize=(10, 6))
plt.plot(X_test, y_test, label='Actual Prices', marker='o')
plt.plot(X_test, y_pred, label='Predicted Prices', marker='x')
plt.title('Stock Price Prediction')
plt.xlabel('Date')
plt.ylabel('Price in $') 
plt.legend()
plt.savefig('stock_prediction.png')
print("Plot saved to stock_prediction.png")

future_date = pd.Timestamp('2024-01-11').toordinal()
future_date = pd.DataFrame({'Date': [future_date]}) 

predicted_price = model.predict(future_date)
print(f"Predicted Stock Price for 2024-01-11: ${predicted_price[0]:.2f}")



