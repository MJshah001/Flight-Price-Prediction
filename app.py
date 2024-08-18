import pandas as pd
import joblib
from flask import (
    Flask,
    url_for,
    render_template,
    request,
    jsonify 
)
from forms import InputForm
import pickle
import xgboost as xgb
import numpy as np
from custom_functions import is_north
from custom_functions import *

app = Flask(__name__)
app.config["SECRET_KEY"] = "secret_key_ml_project"

# Loading the preprocessor pipeline
with open("preprocessor.pkl", "rb") as f:
    preprocessor = pickle.load(f)

# Loading the XGBoost model
with open("xgboost-model", "rb") as f:
    model = pickle.load(f)

# Route for the home page
@app.route("/")
@app.route("/home")
def home():
    return render_template("home.html", title="Home")

# Route for the API guide
@app.route("/api/help")
def api_predict_help():
    return render_template("api_guide.html")

# API route for prediction
@app.route("/api/predict", methods=["POST"])
def api_predict():
    if request.json:
        try:
            # Extracting the JSON data
            data = request.json
                        
            # Constructing the DataFrame
            x_new = pd.DataFrame({
                "airline": [data.get("airline")],
                "date_of_journey": [data.get("date_of_journey")],
                "source": [data.get("source")],
                "destination": [data.get("destination")],
                "dep_time": [data.get("dep_time")],
                "arrival_time": [data.get("arrival_time")],
                "duration": [data.get("duration")],
                "total_stops": [data.get("total_stops")],
                "additional_info": [data.get("additional_info")]
            })

            # Preprocessing the input data
            X_preprocessed = preprocessor.transform(x_new)

            # Convert the preprocessed data to DMatrix, which is required by XGBoost
            dmatrix = xgb.DMatrix(X_preprocessed)

            # Making prediction using the model
            prediction = model.predict(dmatrix)[0]

            # Returning the prediction as a JSON response
            return jsonify({"prediction": str(prediction)})
        
        except Exception as e:
            return jsonify({"error": str(e)})
    else:
        return jsonify({"error": "Invalid input, expecting JSON data."})

# Route for form-based prediction
@app.route("/predict", methods=["GET", "POST"])
def predict():
    try:
        # Creating an instance of the form
        form = InputForm()

        # If the form is submitted
        if form.validate_on_submit():
            x_new = pd.DataFrame({
                "airline": [form.airline.data],
                "date_of_journey": [form.date_of_journey.data.strftime("%Y-%m-%d")],
                "source": [form.source.data],
                "destination": [form.destination.data],
                "dep_time": [form.dep_time.data.strftime("%H:%M:%S")],
                "arrival_time": [form.arrival_time.data.strftime("%H:%M:%S")],
                "duration": [form.duration.data],
                "total_stops": [form.total_stops.data],
                "additional_info": [form.additional_info.data]
            })
            
            # Preprocess the input data
            X_preprocessed = preprocessor.transform(x_new)

            # Convert the preprocessed data to DMatrix, which is required by XGBoost
            dmatrix = xgb.DMatrix(X_preprocessed)

            # Making prediction using the model
            prediction = model.predict(dmatrix)[0]
            message = f"The predicted price is {prediction:,.0f} INR"

        else:
            message = "Please provide valid input details!"

        return render_template("predict.html", title="Predict", form=form, output=message)

    except Exception as e:
        return render_template("predict.html", title="Predict", form=form, output=str(e))

if __name__ == "__main__":
    app.run(debug=True)
