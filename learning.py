import os
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from tensorflow import keras

layers = keras.layers


def main():
    #print("You have tensorflow version", tf.__version__)
    url = 'https://storage.googleapis.com/sara-cloud-ml/wine_data.csv'

    # Get the data
    path = tf.keras.utils.get_file(url.split('/')[-1], url)

    # Convert the data to a Pandas dataframe
    data = pd.read_csv(url)

    # Shuffle the data
    data = data.sample(frac=1)

    # Print the first five rows
    data.head()

    # Do some preprocessing to limit the number of wine varieties in the dataset
    data = data[pd.notnull(data['country'])]
    data = data[pd.notnull(data['price'])]
    data = data.drop(data.columns[0], axis=1)

    vareity_threshold = 600  # Anything less than this will be removed
    value_counts = data['variety'].value_counts()
    to_remove = value_counts[value_counts <= vareity_threshold].index
    data.replace(to_remove, np.nan, inplace=True)
    data = data[pd.notnull(data['variety'])]

    # Split data into train and test
    train_size = int(len(data) * .8)
    print("Train size: ", train_size)
    print("Test size: ", len(data) - train_size)

    # Train features
    description_train = data['description'][:train_size]
    variety_train = data['variety'][:train_size]

    # Train labels
    labels_train = data['price'][:train_size]

    # Test features
    description_test = data['description'][train_size:]
    variety_test = data['variety'][train_size:]

    # Test labels
    labels_test = data['price'][train_size:]

    # Create a tokenizer to preprocess our text description
    # This is a hyperparameter, experiment with different values for your dataset
    vocab_size = 12000
    tokenize = keras.preprocessing.text.Tokenizer(
        num_words=vocab_size, char_level=False)
    tokenize.fit_on_texts(description_train)

    # Wide feature 1: sparce bag of words (bow) vocab_size vector
    description_bow_train = tokenize.texts_to_matrix(description_train)
    description_bow_test = tokenize.texts_to_matrix(description_test)

    # Wide feature 2: one-hot vector of the variety categories

    # Use sklearn utility to convert labelstrings into numbered index
    encoder = LabelEncoder()
    encoder.fit(variety_train)
    variety_train = encoder.transform(variety_train)
    variety_test = encoder.transform(variety_test)
    num_classes = np.max(variety_train) + 1

    # Convert labels to one hot
    variety_train = keras.utils.to_categorical(variety_train, num_classes)
    variety_test = keras.utils.to_categorical(variety_test, num_classes)

    # Define our wide model with he functional API
    bow_inputs = layers.Input(shape=(vocab_size,))
    variety_inputs = layers.Input(shape=(num_classes,))
    merged_layer = layers.concatenate([bow_inputs, variety_inputs])
    merged_layer = layers.Dense(256, activation='relu')(merged_layer)
    predictions = layers.Dense(1)(merged_layer)
    wide_model = keras.Model(
        inputs=[bow_inputs, variety_inputs], outputs=predictions)

    wide_model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
    print(wide_model.summary())

    # Deep model feature: word embeddings of wine description
    train_embed = tokenize.texts_to_sequences(description_train)
    test_embed = tokenize.texts_to_sequences(description_test)

    max_seq_length = 170
    train_embed = keras.preprocessing.sequence.pad_sequences(
        train_embed, maxlen=max_seq_length, padding="post")
    test_embed = keras.preprocessing.sequence.pad_sequences(
        test_embed, maxlen=max_seq_length, padding="post")

    # Define our deep model with the Functional API
    deep_inputs = layers.Input(shape=(max_seq_length,))
    embedding = layers.Embedding(
        vocab_size, 8, input_length=max_seq_length)(deep_inputs)
    embedding = layers.Flatten()(embedding)
    embed_out = layers.Dense(1)(embedding)
    deep_model = keras.Model(inputs=deep_inputs, outputs=embed_out)
    print(deep_model.summary())

    deep_model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])

    # Combine wide and deep into one model
    merged_out = layers.concatenate([wide_model.output, deep_model.output])
    merged_out = layers.Dense(1)(merged_out)
    combined_model = keras.Model(
        wide_model.input + [deep_model.input], merged_out)
    print(combined_model.summary())

    combined_model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])

    # Run Training
    combined_model.fit([description_bow_train, variety_train] + [train_embed],
                       labels_train, epochs=10, batch_size=128)

    # Generate predictions
    predictions = combined_model.predict(
        [description_bow_test, variety_test] + [test_embed])

    # Compare predictions with actual values of the first few items in our test dataset
    num_predictions = 40
    diff = 0

    for i in range(num_predictions):
        val = predictions[i]
        print(description_test.iloc[i])
        print('Predicted: ', val[0], 'Actual: ', labels_test.iloc[i], '\n')
        diff += abs(val[0] - labels_test.iloc[i])

    # Compare the verage difference between actual price and model's predicted price
    print("Average prediction difference: ", diff/num_predictions)

    #f = open('data/output.txt', "w")
    # f.write(str(data))
    # f.close()


if __name__ == '__main__':
    main()
