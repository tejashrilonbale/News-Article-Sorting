from flask import Flask, render_template, request
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

model = pickle.load(open('model.pkl', 'rb'))

# Load the TF-IDF vectorizer with the same settings used during training
tfidf = pickle.load(open('tfidf_vectorizer.pkl', 'rb'))

def ValuePredictor(texts, model, tfidf_vectorizer):
    # Use the same TF-IDF vectorizer during prediction
    text_features = tfidf_vectorizer.transform(texts)
    predictions = model.predict(text_features)
    return predictions

app = Flask(__name__)

@app.route('/', methods=['GET'])
def index():
    return render_template('a.html')

@app.route('/result', methods=['GET', 'POST'])
def result():
    if request.method == 'POST':
        to_predict_list = request.form.to_dict()
        texts = list(to_predict_list.values())
        results = ValuePredictor(texts, model, tfidf)

        if results == 0:
            pred = 'business'
        elif results == 1:
            pred = 'entertainment'
        elif results == 2:
            pred = 'politics'
        elif results == 3:
            pred = 'sport'
        elif results == 4:
            pred = 'tech'
    return render_template('index.html', prediction=pred)
if __name__ == "__main__":
    app.run(host="localhost", port=8000, debug=True)



