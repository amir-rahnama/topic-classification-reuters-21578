import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, LSTM
import sys

this = sys.modules[__name__]


def lstm(X_train, Y_train, X_test, Y_test):
    """Create the LSTM model."""
    document_max_num_words = 100
    num_features = 500
    num_categories = 135

    tb_callback = keras.callbacks.TensorBoard(log_dir='./tb', histogram_freq=0,
                                              write_graph=True, write_images=True)

    model = Sequential()

    model.add(LSTM(int(document_max_num_words * 1.5), input_shape=(document_max_num_words, num_features)))
    model.add(Dropout(0.3))
    model.add(Dense(num_categories))
    model.add(Activation('sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    model.fit(X_train, Y_train, batch_size=128, nb_epoch=5,
              validation_data=(X_test, Y_test), callbacks=[tb_callback])

    model.save('lstm_reuters.h5')

    score, acc = model.evaluate(X_test, Y_test, batch_size=128)

    print('Score: %1.4f' % score)
    print('Accuracy: %1.4f' % acc)
