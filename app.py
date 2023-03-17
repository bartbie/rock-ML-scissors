from flask import Flask, render_template, request
from keras.models import load_model, Model
import numpy as np
from typing import cast, Literal

from numpy.typing import NDArray

MoveType = Literal["rock", "paper", "scissors"]

outcome = ("tie", "lost", "won")

app = Flask(__name__)


def parse_move(move: MoveType):
    if move == "rock":
        return [0, 1, 0]
    elif move == "paper":
        return [1, 0, 0]
    else:
        return [0, 0, 1]


@app.route('/', methods=['post','get'])
def predict() -> str:
    model = cast(Model, load_model('rps.h5'))

    player1: MoveType | None = request.form.get('player1') or None
    player2: MoveType | None = request.form.get('player2') or None

    if player1 is None or player2 is None:
        return render_template('index.html', result="Please select your option first!")
        # calling render_template will inject the variable 'result' and send index.html to the browser

    user_input = np.array([parse_move(player1) + parse_move(player2)])
    print(player1, player2)
    print(f"{user_input!r}")
    predictions = cast(NDArray, model.predict(user_input))
    print(f"{predictions!r}")
    predictions = predictions.flatten().round(decimals=1).astype(int)
    print(f"{predictions!r}")
    result = outcome[predictions@np.arange(3)]
    print(result)
    return render_template('index.html', result=f"You {result}!" if result != "tie" else "It's a tie!")


if __name__ == '__main__':
    app.run()
