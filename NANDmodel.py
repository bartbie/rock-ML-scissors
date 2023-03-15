from keras.backend import flatten
import numpy as np
from numpy._typing import NDArray
from keras.models import Sequential, Model
from keras.layers import Dense
from keras.optimizers import Adam


def train(model: Model, x: NDArray, y: NDArray, epochs: int = 10) -> None:
    model.fit(x, y, epochs=epochs, batch_size=2, verbose=1)


def compile_model():
    model = Sequential()
    model.add(Dense(2, input_dim=2, activation='relu', use_bias=True))
    model.add(Dense(1, activation="sigmoid"))
    adam = Adam(learning_rate=0.01)
    model.compile(loss='mean_squared_error', optimizer=adam)
    return model


def test(model: Model) -> None:
    prediction: NDArray = model.predict([[0, 0], [1, 0], [0, 1], [1, 1]])
    print(list(map(lambda x: f"{x:.2f}", prediction.flatten())))


def main():
    model = compile_model()
    x = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
    y = np.array([[1], [1], [1], [0]])
    train(model, x, y, epochs=2000)
    test(model)
    model.save("nand.h5")


if __name__ == "__main__":
    main()
