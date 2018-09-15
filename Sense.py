## Text Sentiment Machine Learning Model

from keras.datasets import imdb

from keras import models
from keras import layers
from keras.models import load_model

import numpy as np
import flask
import os.path

app = flask.Flask(__name__)

model = None
word_index = None

num_words = 10000

def create_model():
	model = models.Sequential()
	model.add(layers.Dense(16, activation='relu', input_shape=(num_words,)))
	model.add(layers.Dense(16, activation='relu'))
	model.add(layers.Dense(1, activation='sigmoid'))
	return model

def compile_model(model):
	model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
	
def prep_data(sequences, dimension=num_words):
	results = np.zeros((len(sequences), dimension))
	for i, sequence in enumerate(sequences):
		results[i,sequence] = 1.
	return results

def train_model(model):	
	
	# load data
	(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=num_words)

	# process data
	x_train = prep_data(train_data)
	x_test = prep_data(test_data)
	
	# process labels
	y_train = np.asarray(train_labels).astype('float32')
	y_test = np.asarray(test_labels).astype('float32')

	# split train and validation
	x_val = x_train[:10000]
	partial_x_train = x_train[10000:]

	y_val = y_train[:10000]
	partial_y_train = y_train[10000:]

	# train
	history = model.fit( partial_x_train, partial_y_train, epochs=5, batch_size=512, validation_data=(x_val,y_val))

	history_dict = history.history
	for k,v in history_dict.items():
		print(f"{k} has {v}")

	# evaluate the model
	scores = model.evaluate(x_test, y_test, verbose=0)
	print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

	# make a test prediction
	first_item = np.array([x_test[0]])
	pred = model.predict(first_item)
	print(pred)

	# save
	model.save("model.h5")

	return history

def create_new_model():
	load_word_list()
	model = create_model()
	compile_model(model)
	train_model(model)
	return model

# Data Preprocessing Helper Methods

def load_word_list():
	global word_index
	if not word_index:
		word_index = imdb.get_word_index()

def decode_review(data):
	load_word_list()
	reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
	decoded_review = ' '.join([reverse_word_index.get(i - 3, '?') for i in data])
	return decoded_review

def encode_review(review_text):
	load_word_list()
	vector = [word_index.get(i,-2)+3 for i in review_text.split(" ")]
	vector = prep_data([vector])
	
	return vector

def predict_review(review_text):
	vector = encode_review(review_text)
	pred = model.predict(vector)
	return pred

# Routes
@app.route("/predict", methods=["POST"])
def predict():
	
	# Prep default response
	data = dict()
	data["success"] = False
	data["predictions"] = []

	# Get POST json content
	json = flask.request.get_json()
	
	# Encode text into vector and make prediction
	pred = predict_review(json["text"])
	if pred:
		for i in pred:
			data["predictions"].append(str(i))
		data["success"] = True	
		
	resp = flask.jsonify(data)
	return resp
	

if __name__ == "__main__":
	print(("* Loading model and  starting Flask server..."
	"please wait until server has fully started"))
	
	model_filename = "model.h5"

	# Check if file exists
	has_model = os.path.isfile(model_filename) 
	
	# Create if not found
	if not has_model:
		print("No saved model found, creating new one")
		model = create_new_model()
	else:
		model = load_model(model_filename)

	# Make test prediction
	predict_review("hello")

	# Start Flask
	app.run()

	# https://medium.com/coinmonks/deploy-your-first-deep-learning-neural-network-model-using-flask-keras-tensorflow-in-python-f4bb7309fc49