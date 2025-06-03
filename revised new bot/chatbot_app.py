import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import random
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import string
import logging
from flask import Flask, render_template, request, jsonify

# Set up logging
logging.basicConfig(filename='chatbot_errors.log', level=logging.INFO, format='%(asctime)s - %(message)s')

# Download required NLTK data
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

# Initialize lemmatizer and stopwords
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Load the saved model and preprocessing objects
model = tf.keras.models.load_model('medical_chatbot_model.h5')
with open('medical_chatbot_tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)
with open('medical_chatbot_labelencoder.pkl', 'rb') as f:
    le = pickle.load(f)
with open('medical_chatbot_replies.pkl', 'rb') as f:
    replies = pickle.load(f)
with open('medical_chatbot_maxlength.pkl', 'rb') as f:
    max_length = pickle.load(f)

# Preprocess text function
def preprocess_text(text):
    text = [char.lower() for char in text if char not in string.punctuation]
    text = ''.join(text)
    words = nltk.word_tokenize(text)
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return ' '.join(words)

# Initialize Flask app
app = Flask(__name__)

# Route for the home page
@app.route('/')
def home():
    return render_template('chatbot.html')

# Route for handling chatbot queries
@app.route('/chat', methods=['POST'])
def chat():
    try:
        user_input = request.json.get('message', '')
        confidence_threshold = 0.8

        if user_input.lower() == 'quit':
            return jsonify({'response': 'Medical Chatbot: Goodbye!'})

        if len(user_input.strip()) < 3:
            return jsonify({'response': 'Medical Chatbot: Please provide more details about your query.'})

        # Preprocess user input
        user_input_processed = preprocess_text(user_input)
        sequence = tokenizer.texts_to_sequences([user_input_processed])
        padded = pad_sequences(sequence, maxlen=max_length, padding='post')

        # Predict intent
        output = model.predict(padded, verbose=0)
        confidence = np.max(output)
        predicted_tag = le.inverse_transform([np.argmax(output)])[0]

        # Check confidence
        if confidence < confidence_threshold:
            logging.info(f"Low confidence ({confidence:.2f}) for input: {user_input} -> Processed: {user_input_processed} -> Predicted: {predicted_tag}")
            return jsonify({'response': "Medical Chatbot: I'm not sure I understand. Could you clarify your query?"})

        # Respond with a random response from the predicted intent
        response = random.choice(replies[predicted_tag])
        logging.info(f"Input: {user_input} -> Processed: {user_input_processed} -> Predicted: {predicted_tag} -> Confidence: {confidence:.2f}")
        return jsonify({'response': f"Medical Chatbot: {response}"})

    except Exception as e:
        logging.error(f"Error processing input: {user_input} -> Error: {str(e)}")
        return jsonify({'response': "Medical Chatbot: An error occurred. Please try again."})

if __name__ == "__main__":
    print("Starting Flask server at http://127.0.0.1:5000")
    app.run(debug=True, host='127.0.0.1', port=5000)
