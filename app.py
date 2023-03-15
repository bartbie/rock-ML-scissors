from flask import Flask, render_template, request
from keras.models import load_model, Model
import numpy as np
from typing import cast

from numpy._typing import NDArray

app = Flask(__name__)


# def format_prediction(predictions: NDArray) -> str:
#      ...


@app.route('/', methods=['post','get'])
def predict() -> str:
    model = cast(Model, load_model('nand.h5'))

    in1 = request.form.get('in1')
    in2 = request.form.get('in2')

    if in1 is None or in2 is None:
        return render_template('index.html', result="No input(s)")
        # calling render_template will inject the variable 'result' and send index.html to the browser

    user_input = np.array([[ float(in1), float(in2) ]])
    predictions = model.predict(user_input)
    return render_template('index.html', result=str(predictions[0][0]))


if __name__ == '__main__':
    app.run()
