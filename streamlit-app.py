import numpy as np
import pandas as pd

import streamlit as st
# import sklearn
# from sklearn.metrics import r2_score
# from sklearn.metrics.pairwise import rbf_kernel
# from sklearn.base import BaseEstimator, TransformerMixin
# from sklearn.pipeline import Pipeline, FeatureUnion
# from sklearn.compose import ColumnTransformer
# from sklearn.impute import SimpleImputer
# from sklearn.preprocessing import (
#     OneHotEncoder,
#     MinMaxScaler,
#     StandardScaler,
#     PowerTransformer,
#     FunctionTransformer,
#     OrdinalEncoder
# )
# from sklearn.ensemble import RandomForestRegressor

# from feature_engine.outliers import Winsorizer
# from feature_engine.encoding import (
#     RareLabelEncoder,
#     MeanEncoder,
#     CountFrequencyEncoder
# )
# from feature_engine.datetime import DatetimeFeatures
# from feature_engine.selection import SelectBySingleFeaturePerformance

import warnings
warnings.filterwarnings("ignore")

import pickle

import xgboost as xgb

import joblib

# sklearn.set_config(transform_output="pandas") 


from custom_functions import have_info, is_north, flight_duration_categories,is_over,is_direct, part_of_the_day,RBFPercentileSimilarity


# reading the training data
X_train = pd.read_csv("train.csv").drop(columns="price")



# web application

st.set_page_config(
    page_title="Flight Price Prediction",
    page_icon="✈️",
    layout="wide"
)

st.title("Flight Price Prediction")

# user inputs

airline = st.selectbox(
    "Airline:",
    options=X_train.airline.unique()
)

doj = st.date_input("Date of Journey:")

source = st.selectbox(
    "Source:",
    options=X_train.source.unique()
)

destination = st.selectbox(
    "Destination:",
    options=X_train.destination.unique()
)

dep_time = st.time_input("Departure Time:")

arrival_time = st.time_input("Arrival Time:")

duration = st.number_input(
    "Duration(mins):",
    step=5,
    min_value=0
)

total_stops = st.number_input(
    "Total Stops:",
    step=1,
    min_value=0
)

additional_info = st.selectbox(
    "Additional Info:",
    options=X_train.additional_info.unique()
)

x_new = pd.DataFrame(dict(
    airline=[airline],
    date_of_journey=[doj],
    source=[source],
    destination=[destination],
    dep_time=[dep_time],
    arrival_time=[arrival_time],
    duration=[duration],
    total_stops=[total_stops],
    additional_info=[additional_info]
)).astype({
    col: "str"
    for col in ["date_of_journey","dep_time","arrival_time"]
})

if st.button("Predict Price"):
    # Loading the preprocessor pipeline
    with open("preprocessor.pkl", "rb") as f:
        preprocessor = pickle.load(f)

    # Loading the XGBoost model
    with open("xgboost-model", "rb") as f:
        model = pickle.load(f)

    # repickling the preprocessor and model
    with open("preprocessor.pkl", "wb") as f:
        pickle.dump(preprocessor, f)

    with open("xgboost-model", "wb") as f:
        pickle.dump(model, f)

    with open("xgboost-model","rb") as f:
        model = pickle.load(f)
    # Preprocessing the input data
    X_preprocessed = preprocessor.transform(x_new)

    # Convert the preprocessed data to DMatrix, which is required by XGBoost
    dmatrix = xgb.DMatrix(X_preprocessed)

    # Making prediction using the model
    pred = model.predict(dmatrix)[0]
    # pred = 10000 # testing

    st.info(f"The predicted price is {pred:.2f} INR")