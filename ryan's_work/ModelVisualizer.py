import matplotlib.pyplot as plt

class ModelVisualizer:
    def __init__(self):
        pass

    def plot_history_pickle_file(self, history_dict):
        # Directly use the dictionary without trying to access `.history`
        acc = history_dict['accuracy']
        val_acc_key = 'val_accuracy' if 'val_accuracy' in history_dict else 'val_acc'
        val_acc = history_dict[val_acc_key]
        loss = history_dict['loss']
        val_loss_key = 'val_loss' if 'val_loss' in history_dict else 'val_loss_loss'
        val_loss = history_dict.get(val_loss_key, "Not available")

        epochs = range(1, len(acc) + 1)

        # Plot training and validation accuracy
        plt.plot(epochs, acc, 'bo', label='Training accuracy')
        plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
        plt.title('Training and validation accuracy')
        plt.legend()

        plt.figure()

        # Plot training and validation loss
        plt.plot(epochs, loss, 'bo', label='Training loss')
        if val_loss != "Not available":
            plt.plot(epochs, val_loss, 'b', label='Validation loss')
        plt.title('Training and validation loss')
        plt.legend()

        plt.show()
