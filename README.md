
# Sentiment Machine Learning Model
#### A text sentiment ML Model written in **Python**.
This model makes use of Google's Machine learning api named **Keras** to assemble and train the model. Keras uses a Google's ML frameworks called **Tensorflow** as the backend for this model. <br/>
The model is **trained on 50,000 reviews** from the imdb dataset, <br/>
and is served through a **Flask server**. <br/>
The model can be tested by sending an HTTP POST request to "localhost:5000/predict" to make predictions.

## Usage 
### 1. Install Necessary Libraries
#### Install Keras and Tensorflow
https://keras.io/

#### Install Flask
```
pip install flask requests
```
#### Install Numpy
```
python -m pip install numpy scipy matplotlib
```
<br/>
### 2. Run
```
python Sense.py
```
<img src="https://github.com/brendenvogt/Sentiment/raw/master/resources/SentimentStartup.png"/>
<br/>

### 3. Test Endpoint
#### 3.A Curl
```
curl -H "Content-Type: application/json" -X POST -d '{"text":"hello"}' http://localhost:5000/predict
```
Response
```
{"predictions":["[0.52933306]"],"success":true}
```
<img src="https://github.com/brendenvogt/Sentiment/raw/master/resources/SentimentCurl.png"/>
<br/>

#### 3.B Postman
```
http://localhost:5000/predict
```
```
{
	"text":"hello"
}
```
<img src="https://github.com/brendenvogt/Sentiment/raw/master/resources/SentimentPostman.png"/>
