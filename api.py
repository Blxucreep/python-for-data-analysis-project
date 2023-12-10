from flask import Flask, jsonify

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import random
import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from keras.models import Sequential
from keras.layers import Input, Dense, Dropout
from keras.optimizers import Adam

app = Flask(__name__)
app.config["DEBUG"] = True

# create a dictionary to store the results
results = {}


# -------------- #
# pre-processing #
# -------------- #

# create the dataframe
df = pd.read_csv('projet/online_news_popularity.csv')

# create a mask where either ' n_tokens_title' or ' n_tokens_content' is 0 and other 3 columns are 0 (link/image/video)
mask = (df[' n_tokens_title'] == 0) | (df[' n_tokens_content'] == 0) & ((df[' num_hrefs'] == 0) & (df[' num_imgs'] == 0) & (df[' num_videos'] == 0))
# use the mask to drop the rows
df = df[~mask]

# rename the columns
for col in df.columns:
    df.rename(columns={col: col.strip()}, inplace=True)

# drop the columns
df.drop(columns=['url', 'timedelta', 'is_weekend'], inplace=True)


# ---------------------------- #
# modeling using NeuralNetwork #
# ---------------------------- #

# split the dataframe into two variables
X = df.drop(columns=['shares']) # features
Y = df['shares'] # target

# transform the task into a binary task using a decision threshold of 1400
Y = Y.apply(lambda x: 1 if x >= 1400 else 0)

# split the data into train and test sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42) # use 20% of the data for testing

# scale the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# create the model
def create_model(lr_rate=0.001, dropout_rate=0.5, units=[128, 64, 32]):
    model = Sequential()
    model.add(Input(shape=(X_train_scaled.shape[1],)))
    
    for unit in units:
        model.add(Dense(unit, activation='relu'))
        model.add(Dropout(dropout_rate))
    
    model.add(Dense(1, activation='sigmoid'))
    
    optimizer = Adam(learning_rate=lr_rate)
    model.compile(
        loss='binary_crossentropy',
        optimizer=optimizer,
        metrics=['accuracy']
    )
    
    return model

# create a binary classifier
def binary_class_model():
    # set the hyperparameters
    units = [128, 64, 32]
    epochs = 30
    dropout_rate = 0.3
    batch_size = 32
    
    model = Sequential()
    model.add(Input(shape=(X_train_scaled.shape[1],)))
    
    for unit in units:
        model.add(Dense(unit, activation='relu'))
        model.add(Dropout(dropout_rate))
    
    model.add(Dense(1, activation='sigmoid'))
    
    optimizer = Adam(learning_rate=0.001)
    
    model.compile(
        loss='binary_crossentropy',
        optimizer=optimizer,
        metrics=['accuracy']
    )
    
    model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, Y_test))
    
    return model

binary_model = binary_class_model()


# ---------- #
# prediction #
# ---------- #

# make a random prediction with the model trained
def prediction(model, sample):
    # store the sample in the dictionary
    results['sample'] = sample.tolist()
 
    sample = np.reshape(sample, (1, -1))
 
    pred = model.predict(sample)
    pred = 1 if pred > 0.5 else 0

    return pred

# here the line that we need to predict the shares
to_predict = X_test_scaled[random.randint(0, len(X_test_scaled) - 1)]

# make the prediction
predict = prediction(binary_model, to_predict)

# store the prediction in the dictionary
results['shares_prediction'] = predict

# store the name of the model used in the dictionary
results['model'] = 'neural_network'


# --------------- #
# flask api route #
# --------------- #

@app.route('/')
def shares():
    return jsonify(results)

app.run()