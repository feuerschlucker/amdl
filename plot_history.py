import json
import matplotlib.pyplot as plt
from matplotlib import cm


def load_files():
    files = ["model_history_1024.json", "model_history_intermediate_adam.json", "model_history_intermediate_rms_0005.json", "model_history_intermediate_sgd_01.json",
             "model_history_intermediate_sgd_001.json", "model_history_intermediate_sgd_05.json", "model_history_intermediate_sgd_wd_05.json"]
    labels = ['1024', 'adam', 'rmsprop', 'sgd_0.01',
              'sgd_0.001', 'sgd_0.05', 'sgd_0.05_wd']
    colors = ['red', 'blue', 'green', 'magenta', 'brown', 'black']
    history = []

    fig, ax = plt.subplots(figsize=(10, 8))
    cmap = cm.get_cmap('tab10', len(files))

    for j in range(len(files)):
        i = len(files)-1-j
        with open(files[i], 'r') as file:
            history.append(json.load(file))

        ax.plot(history[j]['loss'], label=labels[i] + " Tr.",
                color=cmap(i), linewidth=3, linestyle="-")
        ax.plot(history[j]['val_loss'],
                label=labels[i]+" Val.", color=cmap(i), linewidth=2, linestyle='dashdot')
        print(history[j]['loss'][-1], " , ", history[j]['val_loss'][-1])

    ax.set_ylim(0.2, 0.8)
    ax.set_xlim(-2, 100)
    ax.set_title('Model Loss - ADAM, SGD, RMSPROP')
    ax.set_ylabel('Loss')
    ax.set_xlabel('Epoch')
    ax.legend(loc='upper right')

    plt.savefig('losses_all_2')
    plt.show()


def main():
    load_files()

if __name__ == "__main__":
    main()
