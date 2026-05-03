#app.py
from flask import Flask, render_template, request
from flask import Flask, render_template, request, jsonify
from model.predict import predict_sentiment

# creating an instance of the Flask class,
app = Flask(__name__)  



@app.route('/predict', methods=['POST'])
def predict():
    print("Received request for prediction")
    data = request.get_json()
    review = data.get('review')

    sentiment, confidence = predict_sentiment(review)
    return jsonify({
        'sentiment': sentiment,
        'confidence': round(confidence * 100, 2)
    })

# defining a route for the home page of the web application
@app.route('/', methods=['GET']) 
def home():
    return render_template('base.html') 

#running the flask application in debug mode,  easier development and debugging
if __name__ == '__main__':
    # app.run(debug=True,host="127.0.0.1", port=5000)
    app.run(debug=True, host='127.0.0.1', port=8080) 


