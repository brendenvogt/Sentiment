##-import the necessary packages
from keras.datasets import imdb
##
from keras import models
from keras import layers
from keras.models import load_model

import numpy as np
import flask
import io
import pickle
import os.path

##-todo use WSGI server for production
app = flask.Flask(__name__)

##-model
model = None
word_index = None

num_words = 10000

def load_data():
	global word_index
	if not word_index:
		word_index = imdb.get_word_index()

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
	load_data()
	model = create_model()
	compile_model(model)
	train_model(model)
	return model

# def save_model(model, filename):
	
# 	model_filename = f"{filename}.json"
# 	weight_filename = f"{filename}.h5"

# 	# serialize model to JSON
# 	print("Saving Model")
# 	model_json = model.to_json()
# 	with open(model_filename, "w") as f:
# 		f.write(model_json)
	
# 	# serialize weights to HDF5
# 	print("Saving Weights")
# 	model.save_weights(weight_filename)
	
# 	print("Done")

# def load_model(filename):
# 	global model 
# 	model_filename = f"{filename}.json"
# 	weight_filename = f"{filename}.h5"

# 	# check if saved model exists
# 	has_model = os.path.isfile(model_filename) 
# 	has_weights = os.path.isfile(weight_filename) 
# 	if not has_model and not has_weights:
# 		return None

# 	# deserialize model from JSON
# 	print("Loading Model")
# 	with open(model_filename, 'r') as json_file:
# 		loaded_model_json = json_file.read()
# 		model = model_from_json(loaded_model_json)
		
# 	# deserialize weights from HDF5
# 	print("Loading Weights")
# 	model.load_weights(weight_filename)
	
# 	#compile
# 	print("Compiling")
# 	compile_model()

# 	# return model
# 	return model

def visualize(history):
	import matplotlib.pyplot as plt

	history_dict = history.history
	loss_values = history_dict['loss']
	val_loss_values = history_dict['val_loss']

	epochs = range(1, len(loss_values) + 1)

	plt.plot(epochs, loss_values, 'bo', label='Training Loss')
	plt.plot(epochs, val_loss_values, 'b', label='Validation Loss')
	plt.title('Training and Validation Loss')
	plt.xlabel('Epochs')
	plt.ylabel('Loss')
	plt.legend()	

	img = io.BytesIO()
	plt.savefig(img)
	img.seek(0)

	return img	
	

def decode_review(data):
	reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
	decoded_review = ' '.join([reverse_word_index.get(i - 3, '?') for i in data])
	return decoded_review

def encode_review(review_text):
	load_data()
	vector = [word_index.get(i,-2)+3 for i in review_text.split(" ")]
	vector = prep_data([vector])
	
	return vector

def predict_review(review_text):
	vector = encode_review(review_text)
	pred = model.predict(vector)
	return pred


@app.route("/predict", methods=["POST"])
def predict():
	data = dict()
	data["success"] = False
	data["predictions"] = []

	review_text = flask.request.form['text']	
	pred = predict_review(review_text)
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
	has_model = os.path.isfile(model_filename) 
	if (has_model):
		model = load_model(model_filename)
	else:
		print("No saved model found, creating new one")
		model = create_new_model()

	predict_review("hello")

	app.run()

	# https://medium.com/coinmonks/deploy-your-first-deep-learning-neural-network-model-using-flask-keras-tensorflow-in-python-f4bb7309fc49