import pickle
import pandas as pd
import tensorflow as tf


def load_data():
    dict = pickle.load(open('california-housing-dataset.pkl','rb'))
    x_train, y_train = dict['x_train'], dict['y_train']
    x_test, y_test = dict['x_test'], dict['y_test']  
    return x_train, y_train, x_test, y_test

def normalize():
    
    return 

def build_model():

    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(64, input_shape=(8,), activation='relu'))
    model.add(tf.keras.layers.Dense(128, activation='relu'))
    model.add(tf.keras.layers.Dense(64, activation='relu'))
    model.add(tf.keras.layers.Dense(1))
    model.compile(optimizer='adam', loss='mse')
    model.summary()
    return model










def main():
    load_data()
    
    model = build_model()





if __name__ == '__main__':
    main()