from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle

# Initialize the Flask app
app = Flask(__name__)

CORS(app)

# Load the trained model and TF-IDF vectorizer
with open('../ML Files/model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('../ML Files/tfidf_vectorizer.pkl', 'rb') as vectorizer_file:
    tfidf_vectorizer = pickle.load(vectorizer_file)

# Function to clean and preprocess user input text
def clean_text(text):
    import re
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize
    from nltk.stem import WordNetLemmatizer

    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))

    text = re.sub(r'http\S+', '', text)  # Remove URLs
    text = re.sub(r'@\w+|#\w+|[^\w\s]', '', text)  # Remove @mentions, hashtags, special characters
    tokens = word_tokenize(text)  # Tokenize text
    tokens = [lemmatizer.lemmatize(word.lower()) for word in tokens if word.lower() not in stop_words]  # Remove stopwords and lemmatize
    return ' '.join(tokens)

#Route for home page
@app.route('/', methods=['GET'])
def index():
    return "Hello"


# Route to predict the emotion from input text
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    text = data['text']

    # Preprocess the input text
    text_cleaned = clean_text(text)
    text_vectorized = tfidf_vectorizer.transform([text_cleaned])

    # Predict the emotion
    predicted_emotion = model.predict(text_vectorized)
    
    # Map the predicted label to the corresponding emotion
    emotion_dict = {0: 'sadness', 1: 'joy', 2: 'love', 3: 'anger', 4: 'fear', 5: 'surprise'}
    emotion = emotion_dict[predicted_emotion[0]]

    # Return the prediction as a JSON response
    return jsonify({'emotion': emotion})

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
