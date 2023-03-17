import numpy as np
import pandas as pd
from numpy.typing import NDArray

from keras.models import Sequential, Model
from keras.layers import Dense
from keras.optimizers import Adam
import keras.losses as losses
import keras.activations as activations
import keras.metrics as metrics

import itertools as it
from collections.abc import Iterable
from typing import Callable, TypeVar

OUTPUT_FILE = "RPS.h5"


T = TypeVar('T')
U = TypeVar('U', bound=np.generic, covariant=True)


def map_array(fn: Callable[[T], U], arr: Iterable[T]) -> NDArray[U]:
    """maps an iterable to a numpy array using provided function"""
    return np.array(list(map(fn, arr)))


def training_data() -> tuple[NDArray, NDArray]:
    # written using jupyter notebook

    moves = ["rock", "paper", "scissors"]
    # create all possible combinations
    combs = pd.DataFrame(it.product(moves, repeat=2), columns=["p1", "p2"])

    # convert string values into hot-encoded arrays
    encoded_p1 = pd.get_dummies(combs['p1']).values
    encoded_p2 = pd.get_dummies(combs['p2']).values

    # convert the arrays into scalar arrays,
    # calculate output using matrix substraction and scalar modulo
    to_number = lambda arr: arr@range(len(moves))
    p1_moves = map_array(to_number, encoded_p1)
    p2_moves = map_array(to_number, encoded_p2)
    output = (p1_moves-p2_moves)%3  # type: ignore

    # convert output to hot-encoded arrays
    I = np.identity(len(moves))
    to_array = lambda x: I[x]
    output = map_array(to_array, output)

    return pd.get_dummies(combs).values, output


def compile_model():
    model = Sequential()
    model.add(Dense(6, input_dim=6, activation=activations.relu))
    model.add(Dense(16, input_dim=6, activation=activations.relu))
    model.add(Dense(3, activation=activations.softmax))
    adam = Adam(learning_rate=0.01)
    model.compile(loss=losses.CategoricalCrossentropy(), optimizer=adam, metrics=[metrics.categorical_accuracy])
    return model


def train(model: Model, x: NDArray, y: NDArray, epochs: int = 10, verbose: bool = False) -> None:
    model.fit(x, y, epochs=epochs, batch_size=2, verbose=int(verbose))  # type: ignore


def test(model: Model, x, y, verbose: bool = False) -> tuple[NDArray, bool]:
    prediction: NDArray = model.predict(x)
    prediction_rounded = prediction.round(decimals=1)
    if verbose:
        prediction_formatted = map_array(lambda x: [f"{i:.2f}" for i in x], prediction)
        print(prediction_formatted)
    return prediction_rounded, (prediction_rounded==y).all()


def main():
    print("Compiling the model...")
    model = compile_model()
    print("getting the training data...")
    x, y = training_data()
    print("Training the model...")
    train(model, x, y, epochs=300)
    print("Testing the model...")
    test_results = test(model, x, y)
    print(f"Tests' results same as results from training data: {test_results[1]}")
    print(f"Saving the model to {OUTPUT_FILE}")
    model.save(OUTPUT_FILE)


if __name__ == "__main__":
    main()
