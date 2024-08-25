import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pickle

# Load and clean the dataset
data = pd.read_csv('../Data/text.csv')
data = data.drop(columns=['Unnamed: 0'])

# Download NLTK resources (only needed once)
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Initialize lemmatizer and stopwords
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Function to clean text
def clean_text(text):
    text = re.sub(r'http\S+', '', text)  # Remove URLs
    text = re.sub(r'@\w+|#\w+|[^\w\s]', '', text)  # Remove @mentions, hashtags, and special characters
    tokens = word_tokenize(text)  # Tokenize text
    tokens = [lemmatizer.lemmatize(word.lower()) for word in tokens if word.lower() not in stop_words]  # Remove stopwords and lemmatize tokens
    return ' '.join(tokens)

# Apply the cleaning function to the text column
data['cleaned_text'] = data['text'].apply(clean_text)

# Vectorize the text using TF-IDF
tfidf_vectorizer = TfidfVectorizer(max_features=5000)
X = tfidf_vectorizer.fit_transform(data['cleaned_text'])
y = data['label']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Set up and perform hyperparameter tuning with GridSearchCV
param_grid = {
    'C': [0.01, 0.1, 1, 10, 100],  # Regularization strength
    'solver': ['liblinear', 'saga'],  # Solvers for small datasets
}
grid_search = GridSearchCV(estimator=LogisticRegression(max_iter=1000), param_grid=param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

# Get the best model from grid search
best_model = grid_search.best_estimator_

# Save the model and TF-IDF vectorizer to .pkl files
with open('model.pkl', 'wb') as model_file:
    pickle.dump(best_model, model_file)

with open('tfidf_vectorizer.pkl', 'wb') as vectorizer_file:
    pickle.dump(tfidf_vectorizer, vectorizer_file)

# Function to preprocess user input text and predict emotion
def preprocess_input(text):
    text_cleaned = clean_text(text)  # Apply the same cleaning as during training
    text_vectorized = tfidf_vectorizer.transform([text_cleaned])
    return text_vectorized

# Example: Predicting emotion for a user input
user_input = input("Enter text to predict the emotion: ")
input_vectorized = preprocess_input(user_input)
predicted_emotion = best_model.predict(input_vectorized)

# Map the predicted label to the corresponding emotion
emotion_dict = {0: 'sadness', 1: 'joy', 2: 'love', 3: 'anger', 4: 'fear', 5: 'surprise'}
predicted_label = emotion_dict[predicted_emotion[0]]

print(f"The predicted emotion is: {predicted_label}")
