"""A simple MLP to run the classification."""
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, LSTM


def mlp(X_train, Y_train, X_test, Y_test):
    document_max_num_words = 100
    num_classes = 135
    batch_size = 128
    nb_epoch = 5

    print(num_classes, 'classes')

    print('Vectorizing sequence data...')
    # tokenizer = Tokenizer(num_words=document_max_num_words)
    # x_train = tokenizer.sequences_to_matrix(X_train, mode='binary')
    # x_test = tokenizer.sequences_to_matrix(X_test, mode='binary')
    print('x_train shape:', X_train.shape)
    print('x_test shape:', X_test.shape)

    print('Convert class vector to binary class matrix '
          '(for use with categorical_crossentropy)')
    # y_train = keras.utils.to_categorical(Y_train, num_classes)
    # y_test = keras.utils.to_categorical(Y_test, num_classes)
    print('y_train shape:', Y_train.shape)
    print('y_test shape:', Y_test.shape)

    print('Building model...')
    model = Sequential()
    model.add(Dense(512, input_shape=(document_max_num_words,)))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))

    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    model.fit(X_train, Y_train,
              batch_size=batch_size,
              epochs=nb_epoch,
              verbose=1,
              validation_split=0.1)

    score = model.evaluate(X_test, Y_test,
                           batch_size=batch_size, verbose=1)

    print('Test score:', score[0])
    print('Test accuracy:', score[1])
