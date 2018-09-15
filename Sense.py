##-import the necessary packages
from keras.datasets import imdb
##
from keras import models
from keras import layers

# import tensorflow as tf
import numpy as np
import flask
import io

##-todo use WSGI server for production
app = flask.Flask(__name__)
##-model
model = None
# graph = None 
word_index = None

train_data = None
train_labels = None
test_data = None 
test_labels = None

history = None

def load_data():
    
	##load data
	global train_data, train_labels, test_data, test_labels	
	(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)

    ##load word index
	global word_index
	word_index = imdb.get_word_index()

def create_model():
	global model
	model = models.Sequential()
	model.add(layers.Dense(16, activation='relu', input_shape=(10000,)))
	model.add(layers.Dense(16, activation='relu'))
	model.add(layers.Dense(1, activation='sigmoid'))

	model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])

def prep_data(sequences, dimension=10000):
	results = np.zeros((len(sequences), dimension))
	for i, sequence in enumerate(sequences):
		results[i,sequence] = 1.
	return results

def train_model():
	global history
	
	x_train = prep_data(train_data)
	x_test = prep_data(test_data)
	
	y_train = np.asarray(train_labels).astype('float32')
	y_test = np.asarray(test_labels).astype('float32')

	x_val = x_train[:10000]
	partial_x_train = x_train[10000:]

	y_val = y_train[:10000]
	partial_y_train = y_train[10000:]

	history = model.fit( partial_x_train, partial_y_train, epochs=5, batch_size=512, validation_data=(x_val,y_val))

	history_dict = history.history
	for k,v in history_dict.items():
		print(f"{k} has {v}")
	
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
	'''
		takes in individual review vector and outputs original review text
	'''
	reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
	decoded_review = ' '.join([reverse_word_index.get(i -3, '?') for i in data])
	return decoded_review

@app.route("/train", methods=["GET"])
def train():
	load_data()
	create_model()
	train_model()
	

	resp = flask.jsonify(success=True)
	return resp

@app.route("/predict", methods=["POST"])
def predict():
	visualize(history)
	return send_file(img, mimetype='image/png')
	
if __name__ == "__main__":
    print(("* Loading Keras model and Flask starting server..."
        "please wait until server has fully started"))
    app.run()