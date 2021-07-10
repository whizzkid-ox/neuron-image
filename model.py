from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop
from keras.models import Sequential
from keras.layers.core import Dense, Activation

# Author: Ryo Segawa (whizznihil.kid@gmail.com)

def ann(x_train,y_train,x_test,y_test,batch_size,epochs):
    # create model
    model = Sequential()
    model.add(Dense(256, input_shape=(500,)))
    model.add(Activation('relu'))
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dense(1024))
    model.add(Activation('linear'))

    # Loss function is MSE、and optimiser is rmsprop
    model.compile(loss="mean_squared_error", optimizer="rmsprop")

    # learn
    print("Start learning")
    model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1)

    # Evaluate
    scores = model.evaluate(x_test, y_test, verbose=1)

    return scores, model

