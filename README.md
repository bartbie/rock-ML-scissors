# rock-ML-scissors
## what is this?
simple CLI generator + flask web server for Rock Paper Scissors ML model!

The Model takes an input in form of hot-encoded array of yours and opponent's moves, [Paper, Rock, Scissors],

and spits out prediction on the result in form array [Tie, Loss, Win]

#### for example
- input (paper vs rock) - `[1, 0, 0, 0, 1, 0]`
- output (paper vs rock => win) - `[0, 0, 1]`

## How to install and use this?
### dependecies
create a venv (i recommend using pyenv and pyenv-virtualenv if on UNIX but normal python-venv will also work)

 - (i'm too *lazy* to filter out dependencies needed to run the generator and webserver from ones needed for jupyter, ipython and some other tools etc.
so you **should** use some form of virtual environment lol)

then install the deps using pip

`pip install -r requirements.txt`

### generating model
run `RPSmodel.py -h` to see how to do it
### running flask server
after generating the model, run:

`python -m gunicorn app:app`

use flag `-b` for specyfing socket to bind (see more help using `-h` as usual):

`python -m gunicorn app:app -b 0.0.0.0`
