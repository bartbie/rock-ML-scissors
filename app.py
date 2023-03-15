from flask import Flask

app = Flask(__name__)

@app.route('/')
def hello_world():
	return 'Hello World! Prediction is coming…'

if __name__ == '__main__':
	app.run()
