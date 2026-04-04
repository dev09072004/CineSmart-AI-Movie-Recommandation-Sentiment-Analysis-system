from flask import Flask, jsonify, request
from flask_cors import CORS
import requests
import pandas as pd
from datetime import datetime
import os
from textblob import TextBlob
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

app = Flask(__name__)
CORS(app)

API_KEY = "cb4963533656ec9f61fd85c513a1fee7"

FILE_PATH = "E:/C-Desktop/Desktop/feedback.xlsx"

folder = os.path.dirname(FILE_PATH)
if not os.path.exists(folder):
    os.makedirs(folder)

training_data = [
    ("I love this movie", "Positive"),
    ("This was amazing", "Positive"),
    ("Very good film", "Positive"),
    ("Awesome experience", "Positive"),
    ("I hate this movie", "Negative"),
    ("Worst movie ever", "Negative"),
    ("Very boring", "Negative"),
    ("Waste of time", "Negative"),
    ("It was okay", "Neutral"),
    ("Average movie", "Neutral"),
]

texts = [x[0] for x in training_data]
labels = [x[1] for x in training_data]

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(texts)

model = MultinomialNB()
model.fit(X, labels)

@app.route('/movies', methods=['GET'])
def get_movies():
    genre = request.args.get('genre')
    language = request.args.get('language')

    url = f"https://api.themoviedb.org/3/discover/movie?api_key={API_KEY}"

    if genre:
        url += f"&with_genres={genre}"
    if language:
        url += f"&with_original_language={language}"

    response = requests.get(url)
    data = response.json()

    movies = []

    for movie in data.get("results", [])[:15]:
        if movie.get("poster_path"):
            movies.append({
                "title": movie.get("title"),
                "poster": "https://image.tmdb.org/t/p/w500" + movie.get("poster_path"),
                "rating": movie.get("vote_average")
            })

    return jsonify({"movies": movies})

@app.route('/save-feedback', methods=['POST'])
def save_feedback():
    text = request.json.get("text")

    if not text:
        return jsonify({"error": "No feedback provided"})
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity

    if polarity > 0.2:
        ai_sentiment = "Positive"
    elif polarity < -0.2:
        ai_sentiment = "Negative"
    else:
        ai_sentiment = "Neutral"

  
    transformed = vectorizer.transform([text])
    ml_sentiment = model.predict(transformed)[0]

   
    if ai_sentiment == ml_sentiment:
        final_sentiment = ai_sentiment
    else:
        final_sentiment = ml_sentiment

    new_data = {
        "Review": text,
        "Sentiment": final_sentiment,
        "Date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    try:
        if os.path.exists(FILE_PATH):
            df = pd.read_excel(FILE_PATH)
            df = pd.concat([df, pd.DataFrame([new_data])], ignore_index=True)
        else:
            df = pd.DataFrame([new_data])

        df.to_excel(FILE_PATH, index=False)

    except Exception as e:
        return jsonify({"error": str(e)})
    return jsonify({"message": "Saved"})
if __name__ == '__main__':
    app.run(debug=True)