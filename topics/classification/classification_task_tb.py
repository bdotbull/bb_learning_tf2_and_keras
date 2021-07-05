# basically, "regression.ipynb" with the tensorboard callback

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
from datetime import datetime


df = pd.read_csv('../data/cancer_classification.csv')

# Train-Test-Split
# Target --> `benign_0_mal_1`
X = df.drop('benign_0_mal_1', axis=1).values
y = df['benign_0_mal_1'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

scaler = MinMaxScaler()
scaler.fit(X_train)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# Model Creation
