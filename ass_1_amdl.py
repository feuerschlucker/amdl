import pickle
import pandas as pd
import numpy as np
from scipy import stats
import tensorflow as tf
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import SGD, RMSprop, Adam
from tensorflow.keras import regularizers
import json


def load_data():
    dict = pickle.load(open('california-housing-dataset.pkl', 'rb'))
    x_train, y_train = dict['x_train'], dict['y_train']
    x_test, y_test = dict['x_test'], dict['y_test']
    return x_train, y_train, x_test, y_test


def exclude_clipped(x_train, y_train):
    """
    Exclude data points from x_train and y_train where the target variable y_train
    is considered clipped based on the maximum value.
    """
    maximum = np.max(y_train)
    # Extracting the actual index array
    idx = np.where(y_train > maximum - 0.0001)[0]
    x_train_cleaned = np.delete(x_train, idx, axis=0)
    y_train_cleaned = np.delete(y_train, idx, axis=0)
    return x_train_cleaned, y_train_cleaned


def normalize_training_data(data):
    """
    Normalization of trainings data by subtracting the mean and 
    dividing by std. Mean and std are returned to be used also for 
    test data normalization
    """
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    data_n = (data-mean)/std
    return data_n, mean, std


def normalize_test_data(data, mean, std):
    """
    Test data normalization using mean and std from training data
    """
    data_n = (data-mean)/std
    return data_n


def histos(data, target, feature_names, figname):
    '''
    create a figure with nine histograms, containing features and targets
    '''

    no_features = ((data[0].shape[0]))
    print(len(feature_names))
    fig, axs = plt.subplots(3, 3, figsize=(10, 10))
    axs = axs.ravel()
    plt.suptitle(figname, size=20)
    for i in range(no_features):
        axs[i].hist(data[:, i], bins=50, range=(
            data[:, i].min(), np.percentile(data[:, i], 99.0)))
        axs[i].set_title(feature_names[i])
    axs[8].hist(target, bins=50, color='red')
    axs[8].set_title('Price of Houses')
    plt.show()
    fig.savefig(figname)


def xplot(data, target):
    '''
    create a figure with a scatter plot of latitude vs longitude with
    price as colorcode and "median income" of block group.
    '''
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_title("House Price - Regional Distribution", size=15)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    scatter = ax.scatter(data[:, 7], data[:, 6], c=target,
                         s=data[:, 0]*5, cmap='viridis', alpha=0.6)
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("Price in 100K-$", rotation=270, )
    plt.tight_layout()
    plt.savefig("Lat_long_price.png")
    plt.show()


def validation_set(data, target):
    '''
    Using 20 % of the training data, to create a validation set.  
    '''
    data_train, data_val, target_train, target_val = train_test_split(
        data, target, test_size=0.2, random_state=1)

    return data_train, data_val, target_train, target_val


def plot_training_history(history):
    """
    The function `plot_training_history` generates a plot showing the training and validation loss per
    epoch during model training.
    """
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(loss) + 1)
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, loss, 'b-', label='Training Loss')
    plt.plot(epochs, val_loss, 'r-', label='Validation Loss')
    plt.title('Training and Validation Loss per Epoch')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig("losses")
    plt.show()





def plot_prediction_actual(y_test, y_test_predict):

    y_test = y_test.ravel()
    y_test_predict = y_test_predict.ravel()
    correlation = np.corrcoef(y_test, y_test_predict)[0, 1]
    slope, intercept, r_value, p_value, std_err = stats.linregress(
        y_test, y_test_predict)
    fig, ax = plt.subplots(figsize=(10, 8))
    # ax.scatter(y_test, y_test_predict, label="Data Points")
    
    ax.hist2d(y_test, y_test_predict, bins=40, cmap='Blues')  # Adjust gridsize and colormap as needed
    #ax.colorbar(label='Density')
    
    ax.plot(y_test, intercept + slope * y_test, 'r',
            label=f'Regression Line: y = {slope:.2f}x + {intercept:.2f}', lw=2)
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
            'k--', lw=2, label="Perfect Prediction Line")
    textstr = f"Correlation: {correlation:.2f}"
    ax.text(0.05, 0.75, textstr, transform=ax.transAxes, fontsize=12,
            verticalalignment='top', bbox=dict(facecolor='white', alpha=0.5))
    plt.title("Prediction vs Actual")
    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    plt.legend()
    plt.show()
    fig.savefig("prediction_vs_actual.png")


def build_model_regression():
    model = tf.keras.models.Sequential()
    model.add(tf.keras.Input(shape=(8,)))
    model.add(tf.keras.layers.Dense(1))
    return model


def build_model_intermediate():
    model = tf.keras.models.Sequential()
    model.add(tf.keras.Input(shape=(8,)))
    model.add(tf.keras.layers.Dense(32,  activation='relu'))
    model.add(tf.keras.layers.Dense(128, activation='relu'))
    model.add(tf.keras.layers.Dense(64, activation='relu'))
    model.add(tf.keras.layers.Dense(1))
    return model


def build_model_advanced():
    model = tf.keras.models.Sequential()
    model.add(tf.keras.Input(shape=(8,)))
    model.add(tf.keras.layers.Dense(256,  activation='relu'))
    model.add(tf.keras.layers.Dense(512, activation='relu'))
    model.add(tf.keras.layers.Dense(256, activation='relu'))
    model.add(tf.keras.layers.Dense(256,  activation='relu'))
    model.add(tf.keras.layers.Dense(512, activation='relu'))
    model.add(tf.keras.layers.Dense(256, activation='relu'))
    model.add(tf.keras.layers.Dense(1))
    return model


def build_model_L2reg():
    model = tf.keras.models.Sequential()
    model.add(tf.keras.Input(shape=(8,)))
    model.add(tf.keras.layers.Dense(64,  activation='relu',
              kernel_regularizer=regularizers.l2(0.01)))
    model.add(tf.keras.layers.Dense(128, activation='relu',
              kernel_regularizer=regularizers.l2(0.01)))
    model.add(tf.keras.layers.Dense(256, activation='relu',
              kernel_regularizer=regularizers.l2(0.01)))
    model.add(tf.keras.layers.Dense(512, activation='relu',
              kernel_regularizer=regularizers.l2(0.01)))
    model.add(tf.keras.layers.Dense(1024, activation='relu',
              kernel_regularizer=regularizers.l2(0.1)))
    model.add(tf.keras.layers.Dense(512, activation='relu',
              kernel_regularizer=regularizers.l2(0.01)))
    model.add(tf.keras.layers.Dense(128, activation='relu',
              kernel_regularizer=regularizers.l2(0.01)))
    model.add(tf.keras.layers.Dense(64, activation='relu',
              kernel_regularizer=regularizers.l2(0.01)))
    model.add(tf.keras.layers.Dense(1))
    return model


def build_model_dropout():
    model = tf.keras.models.Sequential()
    model.add(tf.keras.Input(shape=(8,)))
    model.add(tf.keras.layers.Dense(256,  activation='relu'))
    model.add(tf.keras.layers.Dropout(0.3))
    model.add(tf.keras.layers.Dense(512, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.3))
    model.add(tf.keras.layers.Dense(256, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.3))
    model.add(tf.keras.layers.Dense(256,  activation='relu'))
    model.add(tf.keras.layers.Dropout(0.3))
    model.add(tf.keras.layers.Dense(512, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.3))
    model.add(tf.keras.layers.Dense(256, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.3))
    model.add(tf.keras.layers.Dense(1))
    return model


def model_compile_adam(model):
    learning_rate=0.01
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mse')
    model.summary()
    return model


def model_compile_sgd(model):
    learning_rate = 0.1
    weight_decay = 0.5
    optimizer = SGD(learning_rate=learning_rate, weight_decay=weight_decay)
    model.compile(optimizer='sgd', loss='mse')
    model.summary()
    return model


def model_compile_rmsprop(model):
    learning_rate = 0.0005
    rho = 0.9
    momentum = 0.0
    optimizer = RMSprop(learning_rate=learning_rate,
                        rho=rho, momentum=momentum)
    model.compile(optimizer=optimizer, loss='mse')
    model.summary()
    return model


def model_fit(model, data_train, target_train, data_val, target_val):
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=10)
    history = model.fit(data_train, target_train, epochs=500, batch_size=64,
                        validation_data=(data_val, target_val), callbacks=[early_stopping])
    model.save('model.keras')
    with open('model_history.json', 'w') as file:
        json.dump(history.history, file)

    return model, history


def main():
    f_names = ("MedInc", "HouseAge", "AveRooms", "AveBedrms",
               "Population", "AveOccup", "Latitude", "Longitude")  # feature names
    # LOAD DATA
    x_train, y_train, x_test, y_test = load_data()

    # DATA CLIPPING - Optional, only used for data analysis
    # x_train, y_train = exclude_clipped(x_train, y_train)
    # x_test, y_test = exclude_clipped(x_test, y_test)

    # figname = "Training Data - Raw"
    # histos(x_train, y_train, f_names, figname)
    # xplot(x_train, y_train)
    
    # figname = "Test Data - Raw"
    # histos(x_test, y_test, f_names, figname)
    # xplot(x_test, y_test)

    # DATA NORMALIZATION
    x_train, mean, std = normalize_training_data(x_train)
    x_test = normalize_test_data(x_test, mean, std)

    # figname = "Training Data - Normalized"
    # histos(x_train, y_train, f_names, figname)
    
    # VALIDATION SET CREATION
    data_train, data_val, target_train, target_val = validation_set(
        x_train, y_train)

    # SELECT MODEL ARCHITECTURE
    # model = build_model_regression()
    model = build_model_intermediate()
    # model = build_model_advanced()
    # model = build_model_L2reg()
    # model = build_model_dropout()

    # SELECT OPTIMIZER
    # model = model_compile_sgd(model)
    # model = model_compile_rmsprop(model)
    model = model_compile_adam(model)

    # MODEL FIT
    model, history = model_fit(
        model, data_train, target_train, data_val, target_val)

    plot_training_history(history)

    # MODEL PREDICT
    y_test_predict = model.predict(x_test)
    plot_prediction_actual(y_test, y_test_predict)


if __name__ == '__main__':
    main()
