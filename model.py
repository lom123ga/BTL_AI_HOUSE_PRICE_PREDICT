import warnings
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
import pickle

warnings.filterwarnings('ignore')
np.random.seed(2312)

def clean_dataframe(df):
    df.dropna(inplace=True)
    df = df.drop("id", axis=1)
    df['date'] = pd.to_datetime(df['date'])
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df = df.drop("date", axis=1)
    df = df[['sqft_living', 'grade', 'sqft_living15', 'bathrooms', 'view', 'sqft_basement', 'yr_renovated',
             'lat', 'waterfront', 'yr_built', 'bedrooms', 'price', 'year', 'month', 'day', 'floors']]
    return df

def linear_regression(X_train, y_train):
    A = np.dot(X_train.T, X_train)
    b = np.dot(X_train.T, y_train)
    w = np.dot(np.linalg.pinv(A), b)
    return w

def evaluate(w, X_test, y_test):
    y_pred = np.dot(X_test, w)
    return r2_score(y_pred, y_test)

def load_model():
    with open('data/weights.pkl', 'rb') as f:
        w = pickle.load(f)
    return w
