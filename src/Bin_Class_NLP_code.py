import numpy as np
import matplotlib.pyplot as plt 
from keras.datasets import imdb # type: ignore
from keras import models, layers
from tensorflow.keras.utils import plot_model # type: ignore

def binary_classification():
    """
    This function trains a binary classification model using a neural network.
    It uses the IMDB dataset for sentiment analysis.
    """
    # Load the IMDB dataset with a vocabulary size limit of 10,000 words
    (train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)
    
    print(train_data[0])  # Print an example of tokenized data
    
    # Get the word index mapping words to numerical values
    word_index = imdb.get_word_index()
    
    # Reverse the word index to reconstruct original reviews
    reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
    
    # Decode the first review (words start at index 3 due to reserved indices)
    decoded_review = ' '.join([reverse_word_index.get(i - 3, '?') for i in train_data[0]])
    print(decoded_review)
    print(train_labels[0])  # Print the label (0 = negative, 1 = positive)
    
    def vectorize_sequences(sequences, dimension=10000):
        """
        Converts lists of integers into a binary matrix representation.
        Each row corresponds to a review, where 1 indicates the presence of a word.
        """
        results = np.zeros((len(sequences), dimension))
        for i, sequence in enumerate(sequences):
            results[i, sequence] = 1.
        return results
    
    # Vectorize the training and test data
    x_train = vectorize_sequences(train_data)
    x_test = vectorize_sequences(test_data)
    print(x_train[0])  # Print an example of vectorized data
    
    # Convert labels to float32 for TensorFlow compatibility
    y_train = np.asarray(train_labels).astype('float32')
    y_test = np.asarray(test_labels).astype('float32')
    
    # Define the neural network model
    model = models.Sequential()
    model.add(layers.Dense(64, activation='relu', input_shape=(10000,)))  # First hidden layer with 64 neurons
    model.add(layers.Dense(16, activation='relu'))  # Second hidden layer with 16 neurons
    model.add(layers.Dense(1, activation='sigmoid'))  # Output layer for binary classification
    
    # Plot the model architecture
    plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
    
    # Split the training data into validation and training sets
    x_val = x_train[:10000]
    partial_x_train = x_train[10000:]
    
    y_val = y_train[:10000]
    partial_y_train = y_train[10000:]
    
    # Compile the model with an optimizer, loss function, and evaluation metric
    model.compile(optimizer='rmsprop',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    
    # Train the model for 3 epochs with a batch size of 256
    history = model.fit(partial_x_train,
                        partial_y_train,
                        epochs=3,
                        batch_size=256,
                        validation_data=(x_val, y_val))
    
    # Extract loss and accuracy history
    history_dict = history.history
    
    # Plot training and validation loss
    loss_values = history_dict['loss']
    val_loss_values = history_dict['val_loss']
    epochs = range(1, len(loss_values) + 1)
    
    plt.plot(epochs, loss_values, 'bo', label='Training loss')
    plt.plot(epochs, val_loss_values, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
    
    plt.clf()  # Clear the figure
    
    # Plot training and validation accuracy
    acc_values = history_dict['accuracy']
    val_acc_values = history_dict['val_accuracy']
    
    plt.plot(epochs, acc_values, 'bo', label='Training accuracy')
    plt.plot(epochs, val_acc_values, 'b', label='Validation accuracy')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # Evaluate the model on the test set
    model.evaluate(x_test, y_test)
    results = model.evaluate(x_test, y_test)
    print(results)  # Print test accuracy and loss
    
    # Make predictions on the first two test samples
    model.predict(x_test[0:2])
    
    plt.show()