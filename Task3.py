#🔍 1. Problem Statement & Goal
#Example: Predict housing prices based on features like location, size, etc.

#📥 2. Data Collection
#Source: Kaggle, web scraping, open APIs, etc.

#Tools: requests, BeautifulSoup, pandas, kaggle API
import pandas as pd
data = pd.read_csv("housing_data.csv")
#🧹 3. Data Preprocessing
#Handle missing values, categorical encoding, feature scaling

#Split into train/test

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Fill missing values
data.fillna(method='ffill', inplace=True)

# Encode categorical variables
data = pd.get_dummies(data)

# Feature scaling
scaler = StandardScaler()
X = scaler.fit_transform(data.drop("target", axis=1))
y = data["target"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
#🤖 4. Model Building
#Choose a model: Linear Regression, RandomForest, etc.

#Train & evaluate

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

model = RandomForestRegressor()
model.fit(X_train, y_train)
predictions = model.predict(X_test)

print("MSE:", mean_squared_error(y_test, predictions))
#💾 5. Save the Model

import joblib
joblib.dump(model, "model.pkl")
joblib.dump(scaler, "scaler.pkl")
#🌐 6. API Deployment with Flask (or FastAPI)
Flask Example:

from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json["data"]  # e.g., [3, 2, 1500, ...]
    scaled_data = scaler.transform([data])
    prediction = model.predict(scaled_data)
    return jsonify({"prediction": prediction[0]})

if __name__ == "__main__":
    app.run(debug=True)
#🚀 7. Deployment
Use: Render, Railway, Heroku, or deploy on a local server

#Include a README and requirements.txt

#Example for requirements.txt:

#makefile
#Flask==2.3.2
#scikit-learn==1.3.0
#joblib
#numpy
#pandas
