
# Sentiment Machine Learning Model
#### A text sentiment ML Model written in **Python**.
This model makes use of Google's Machine learning api named **Keras** to assemble and train the model. Keras uses a Google's ML frameworks called **Tensorflow** as the backend for this model. <br/>
The model is **trained on 50,000 reviews** from the imdb dataset. <br/>
Hosted through a **Flask server**, the model can be tested by sending an HTTP POST request to "localhost:5000/predict" to make predictions

## Usage 
### Install Necessary Libraries
### Run Sense.py
```
python Sense.py
```
<img src="https://github.com/brendenvogt/Sentiment/raw/master/resources/SentimentStartup.png"/>
<br/>

### Test Endpoint
<br/>

#### Curl
```
curl -H "Content-Type: application/json" -X POST -d '{"text":"hello"}' http://localhost:5000/predict
```
Response
```
{"predictions":["[0.52933306]"],"success":true}
```
<img src="https://github.com/brendenvogt/Sentiment/raw/master/resources/SentimentCurl.png"/>
<br/>

#### Postman
```
Postman to http://localhost:5000/predict
```
<img src="https://github.com/brendenvogt/Sentiment/raw/master/resources/SentimentPostman.png"/>
<br/>

