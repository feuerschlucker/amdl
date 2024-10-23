import pickle
import pandas as pd
import numpy as np
from scipy import stats
import tensorflow as tf
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split


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
    

    idx = np.where(y_train > maximum - 0.0001)[0]  # Extracting the actual index array

    # Print maximum value and number of excluded rows for checking
    print(f"Maximum value in y_train: {maximum}")
    print(f"Number of rows to exclude: {len(idx)}")

    # Exclude rows from x_train and y_train based on the indices
    x_train_cleaned = np.delete(x_train, idx, axis=0)
    y_train_cleaned = np.delete(y_train, idx, axis=0)
    
    return x_train_cleaned, y_train_cleaned

def normalize_training_data(data):
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    data_n = (data-mean)/std
    return data_n, mean, std

def normalize_test_data(data, mean, std):
    data_n = (data-mean)/std
    return data_n


def histos(data, target, feature_names, figname):
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
    plt.show()
    fig.savefig(figname)


def xplot(data, target):
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_title("House Price - Regional Distribution", size=15)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    scatter = ax.scatter(data[:, 6], data[:, 7], c=target,
                         s=data[:, 0]*5, cmap='viridis', alpha=0.6)
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("Price in 100K-$", rotation=270, )
    plt.tight_layout()
    plt.savefig("Lat_long_price.png")
    plt.show()


def validation_set(data, target):
    data_train, data_val, target_train, target_val = train_test_split(
        data, target, test_size=0.2, random_state=1)
    print(data_train.shape)
    print(data_val.shape)
    print(target_train.shape)
    print(target_val.shape)
    return data_train, data_val, target_train, target_val


def plot_training_history(history):
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


def plot_prediction_actual_old(y_test, y_test_predict):
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(y_test, y_test_predict)
    ax.plot([y_test.min(), y_test.max()], [
            y_test.min(), y_test.max()], 'k--', lw=2)
    plt.show()
    plt.savefig("prediction_vs_actual")

def plot_prediction_actual(y_test, y_test_predict):

    y_test = y_test.ravel()
    y_test_predict = y_test_predict.ravel()
    correlation = np.corrcoef(y_test, y_test_predict)[0, 1]
    slope, intercept, r_value, p_value, std_err = stats.linregress(y_test, y_test_predict)
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(y_test, y_test_predict, label="Data Points")
    ax.plot(y_test, intercept + slope * y_test, 'r', label=f'Regression Line: y = {slope:.2f}x + {intercept:.2f}', lw=2)
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2, label="Perfect Prediction Line")
    textstr = f"Correlation: {correlation:.2f}"
    ax.text(0.05, 0.75, textstr, transform=ax.transAxes, fontsize=12,
            verticalalignment='top', bbox=dict(facecolor='white', alpha=0.5))
    plt.title("Prediction vs Actual")
    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    plt.legend()
    plt.show()
    fig.savefig("prediction_vs_actual.png")


def build_model():
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


def build_model_regression():
    model = tf.keras.models.Sequential()
    model.add(tf.keras.Input(shape=(8,)))
    model.add(tf.keras.layers.Dense(1))
    model.compile(optimizer='adam', loss='mse')
    model.summary()
    return model


def model_compile(model):
    model.compile(optimizer='adam', loss='mse')
    model.summary()
    return model


def model_fit(model, data_train, target_train, data_val, target_val):
    early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', patience=10)
    history = model.fit(data_train, target_train, epochs=100, batch_size=64,
              validation_data=(data_val, target_val), callbacks=[early_stopping])
    
    model.save('model.keras')
    
    return model, history




def main():
    f_names = ("MedInc", "HouseAge", "AveRooms", "AveBedrms",
               "Population", "AveOccup", "Latitude", "Longitude")  # feature names
    x_train, y_train, x_test, y_test = load_data()

    x_train, y_train= exclude_clipped(x_train, y_train)
    x_test, y_test =exclude_clipped(x_test, y_test)
    figname = "Training Data - Raw"
    histos(x_train, y_train, f_names, figname)
    # xplot(x_train,y_train)
    
    x_train, mean, std= normalize_training_data(x_train)
    x_test = normalize_test_data(x_test, mean,std)
    
    figname = "Training Data - Normalized"
    histos(x_train,y_train, f_names, figname)

    data_train, data_val, target_train, target_val = validation_set(
        x_train, y_train)

    model = build_model_regression()
    model = model_compile(model)
    model, history = model_fit(model, data_train, target_train, data_val, target_val)
    

    plot_training_history(history)
    
    
    y_test_predict = model.predict(x_test)
    
    
    plot_prediction_actual(y_test, y_test_predict)
    



if __name__ == '__main__':
    main()
