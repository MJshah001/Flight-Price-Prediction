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


app = Flask(__name__)
app.config["SECRET_KEY"] = "secret_key_ml_project"


###############
# custom functions
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np

def have_info(X):
    '''
    This function takes a DataFrame or a numpy array and returns a DataFrame with a new column
    that indicates whether the value in the column "additional_info" is "No Info" or not.
    If the value is "No Info", the new column will have a value of 0, otherwise it will have a value of 1.

    Parameters:
    X: DataFrame or numpy array

    Returns:
    DataFrame
    '''
    if isinstance(X, np.ndarray):
        X = pd.DataFrame(X, columns=["additional_info"])
    return X.assign(additional_info=X.additional_info.ne("No Info").astype(int))

def is_north(X):
    '''
    This function takes a DataFrame or a numpy array and returns a DataFrame with new columns
    that indicate whether the value in the columns "source" and "destination" are cities in the north of India.
    If the value is a city in the north, the new column will have a value of 1, otherwise it will have a value of 0.

    Parameters:
    X: DataFrame or numpy array

    Returns:
    DataFrame
    '''
    columns = X.columns.to_list()
    north_cities = ["Delhi","New Delhi","Kolkata","Mumbai"]
    return(
        X
        .assign(**{
            f"{col}_is_north": X.loc[:, col].isin(north_cities).astype(int)
            for col in columns
        })
        .drop(columns=columns)
    )

def flight_duration_categories(X, short=180, medium=400):
    '''
    This function takes a DataFrame or a numpy array and returns a DataFrame with a new column
    that categorizes the values in the column "duration" into three categories: "short", "medium", and "long".
    The categories are defined by the values of the parameters short and medium.

    Parameters:
    X: DataFrame or numpy array
    short: int, default=180
        The value that separates the "short" and "medium" categories.
    medium: int, default=400
        The value that separates the "medium" and "long" categories.

    Returns:
    DataFrame
    '''
    if isinstance(X, np.ndarray):
        X = pd.DataFrame(X, columns=["duration"]) 
    return (
        X
        .assign(flight_duration_categories=np.select(
            [X.duration.lt(short),
             X.duration.between(short, medium, inclusive="left")],
            ["short", "medium"],
            default="long"
        ))
        .drop(columns="duration")
    )


def is_over(X,value=1000):
    '''
    This function takes a DataFrame or a numpy array and returns a DataFrame with a new column
    that indicates whether the value in the column "duration" is greater than or equal to the specified value.
    If the value is greater than or equal to the specified value, the new column will have a value of 1, otherwise it will have a value of 0.

    Parameters:
    X: DataFrame or numpy array
    value: int, default=1000
        The value to compare the values in the column "duration".

    Returns:
    DataFrame
    '''
    if isinstance(X, np.ndarray):
        X = pd.DataFrame(X, columns=["duration"])
    return (
        X
        .assign(**{
            f"duration_over_{value}": X.duration.ge(value).astype(int)
        })
        .drop(columns="duration")
    )

def is_direct(X):
    '''
    This function takes a DataFrame or a numpy array and returns a DataFrame with a new column
    that indicates whether the value in the column "total_stops" is 0.
    If the value is 0, the new column will have a value of 1, otherwise it will have a value of 0.

    Parameters:
    X: DataFrame or numpy array

    Returns:
    DataFrame
    '''
    if isinstance(X, np.ndarray):
        X = pd.DataFrame(X, columns=["total_stops"])
    return (
        X
        .assign(
            is_direct_flight=X.total_stops.eq(0).astype(int)
        )
    )

def part_of_the_day(X,morning=4,afternoon=12,evening=16,night=20):
    '''
    This function takes a DataFrame or a numpy array and returns a DataFrame with new columns
    that indicate the part of the day of the values in the columns.
    The parts of the day are defined by the values of the parameters morning, afternoon, evening, and night.

    Parameters:
    X: DataFrame or numpy array
    morning: int, default=4
        The value that separates the "night" and "morning" parts of the day.
    afternoon: int, default=12
        The value that separates the "morning" and "afternoon" parts of the day.
    evening: int, default=16
        The value that separates the "afternoon" and "evening" parts of the day.
    night: int, default=20
        The value that separates the "evening" and "night" parts of the day.

    Returns:
    DataFrame
    '''
    if isinstance(X, np.ndarray):
        X = pd.DataFrame(X)

    columns = X.columns.to_list()
    X_temp = X.assign(**{
        col: pd.to_datetime(X.loc[:,col]).dt.hour
        for col in columns
    })

    return (
        X_temp
        .assign(**{
            f"{col}_part_of_day": np.select(
                [X_temp.loc[:,col].between(morning,afternoon,inclusive="left"),
                X_temp.loc[:,col].between(afternoon,evening,inclusive="left"),
                X_temp.loc[:,col].between(evening,night,inclusive="left"),],
                ["morning","afternoon","evening"],
                default="night"
            )
            for col in columns
        })
        .drop(columns=columns)
    )

class RBFPercentileSimilarity(BaseEstimator, TransformerMixin):
    '''
    This class creates a transformer that calculates the similarity between the values in the input DataFrame
    and the reference values using the Radial Basis Function (RBF) kernel.
    The reference values are the percentiles of the values in the input DataFrame columns.
    The similarity is calculated for each percentile specified in the percentiles parameter.

    Parameters:
    variables: list, default=None
        The variables to use for calculating the similarity.
        If None, all numerical variables will be used.
    percentiles: list, default=[0.25,0.5,0.75]
        The percentiles to use as reference values.
    gamma: float, default=0.1
        The gamma parameter of the RBF kernel.
    '''

    def __init__(self, variables=None, percentiles=[0.25,0.5,0.75],gamma=0.1):
        self.variables = variables
        self.percentiles = percentiles
        self.gamma = gamma

    def fit(self,X,y=None):
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=self.variables or range(X.shape[1]))
        if not self.variables:
            self.variables = X.select_dtypes(include="number").columns.to_list()

        self.reference_values_ = {
            col: (
                X
                .loc[:,col]
                .quantile(self.percentiles)
                .values
                .reshape(-1,1)
            )
            for col in self.variables
        }
        
        return self

    def transform(self,X):
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=self.variables or range(X.shape[1]))
        objects = []
        for col in self.variables:
            columns = [f"{col}_rbf_{int(percentile*100)}" for percentile in self.percentiles]
            obj = pd.DataFrame(
                data=rbf_kernel(X.loc[:,[col]],Y=self.reference_values_[col],gamma=self.gamma),
                columns=columns
            )
            objects.append(obj)
        return pd.concat(objects,axis=1)
###############

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

