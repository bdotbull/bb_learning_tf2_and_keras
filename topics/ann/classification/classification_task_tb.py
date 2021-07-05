# basically, "regression.ipynb" with the tensorboard callback

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard

from datetime import datetime

df = pd.read_csv('../../../data/cancer_classification.csv')

# Train-Test-Split
# Target --> `benign_0_mal_1`
X = df.drop('benign_0__mal_1', axis=1).values
y = df['benign_0__mal_1'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

scaler = MinMaxScaler()
scaler.fit(X_train)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# Model Creation
early_stop = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=25)

#datetime.now().strftime("%Y-%m-%d--%H%M")  # datetime format
log_directory = f'logs\\fit\\{datetime.now().strftime("%Y-%m-%d--%H%M")}'

board = TensorBoard(log_dir=log_directory, histogram_freq=1,
                    write_graph=True,
                    write_images=True,
                    update_freq='epoch',
                    profile_batch=2,
                    embeddings_freq=1)
"""
    Arguments:
        log_dir: the path of the directory where to save the log files to be
          parsed by TensorBoard.

        histogram_freq: frequency (in epochs) at which to compute activation and
          weight histograms for the layers of the model. If set to 0, histograms
          won't be computed. Validation data (or split) must be specified for
          histogram visualizations.

        write_graph: whether to visualize the graph in TensorBoard. The log file
          can become quite large when write_graph is set to True.

        write_images: whether to write model weights to visualize as image in
          TensorBoard.

        update_freq: `'batch'` or `'epoch'` or integer. When using `'batch'`,
          writes the losses and metrics to TensorBoard after each batch. The same
          applies for `'epoch'`. If using an integer, let's say `1000`, the
          callback will write the metrics and losses to TensorBoard every 1000
          samples. Note that writing too frequently to TensorBoard can slow down
          your training.

        profile_batch: Profile the batch to sample compute characteristics. By
          default, it will profile the second batch. Set profile_batch=0 to
          disable profiling. Must run in TensorFlow eager mode.

        embeddings_freq: frequency (in epochs) at which embedding layers will
          be visualized. If set to 0, embeddings won't be visualized.
"""

model = Sequential()

model.add(Dense(30, activation='relu'))   # input layer
model.add(Dropout(0.5))
model.add(Dense(15, activation='relu'))   # hidden layer
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid')) # output layer

model.compile(loss='binary_crossentropy', optimizer='adam')

# Training
model.fit(x=X_train,
          y=y_train,
          epochs=500,
          validation_data=(X_test, y_test),
          verbose=1,
          callbacks=[early_stop, board])
