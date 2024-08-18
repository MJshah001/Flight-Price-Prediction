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


