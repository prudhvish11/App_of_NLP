from flask import Flask, request, jsonify,render_template,redirect
import os
from flask_cors import CORS, cross_origin
from spellcorrector import spell_corrector
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import speech_recognition as sr
import textToSPeech

os.putenv('LANG', 'en_US.UTF-8')
os.putenv('LC_ALL', 'en_US.UTF-8')

app = Flask(__name__)
CORS(app)


@app.route("/")
@cross_origin()
def home():
    return render_template('main.html')


@app.route("/t2s", methods=['GET'])
@cross_origin()
def text2():
    return render_template('index.html')


@app.route("/spell", methods=['GET'])
@cross_origin()
def spell():
    return render_template('index1.html')

@app.route("/spamm", methods=['GET','POST'])
def s():
    return render_template('spam.html') 

@app.route("/senti", methods=['GET'])
@cross_origin()
def senti():
    return render_template('sen.html')

@app.route("/s2t", methods=['GET','POST'])
@cross_origin()
def s2t():
    transcript = ""
    if request.method == "POST":
        print("FORM DATA RECEIVED")

        if "file" not in request.files:
            return redirect(request.url)

        file = request.files["file"]
        if file.filename == "":
            return redirect(request.url)

        if file:
            recognizer = sr.Recognizer()
            audioFile = sr.AudioFile(file)
            with audioFile as source:
                data = recognizer.record(source)
            transcript = recognizer.recognize_google(data, key=None)

    return render_template('speech2text.html', transcript=transcript)   

@app.route("/predict", methods=['POST'])
@cross_origin()
def predictRoute():
    data = request.json['data']
    result = textToSPeech.text2Speech(data)
    return {"data" : result.decode("utf-8")}


@app.route("/predict1", methods=['POST'])
@cross_origin()
def predictRoute1():
    data = request.json['data']
    result = spell_corrector(data)
    return jsonify({ "text" : result})    


@app.route('/spam', methods=['POST'])
def spam():
    df = pd.read_csv("spam.csv", encoding="latin-1")
    df.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1, inplace=True)
    df['label'] = df['class'].map({'ham': 0, 'spam': 1})
    X = df['message']
    y = df['label']
    cv = CountVectorizer()
    X = cv.fit_transform(X)  
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    clf = MultinomialNB()
    clf.fit(X_train, y_train)
    clf.score(X_test, y_test)
    if request.method == 'POST':
        message = request.form['message']
        data = [message]
        vect = cv.transform(data).toarray()
        my_prediction = clf.predict(vect)
    return render_template('spam.html', prediction=my_prediction)  

@app.route('/an',methods=['GET',"POST"])
def an():
    render_template('sen.html')
    if request.method=="POST":
        inp=request.form.get('message')
        sid=SentimentIntensityAnalyzer()
        score=sid.polarity_scores(inp)
        if score["neg"]!=0:
            return render_template('sen.html', answer='Negative')
        else:
            return render_template('sen.html', answer='Positive')
    return render_template('sen.html')            


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)