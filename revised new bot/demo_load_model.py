
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

# Chatbot interaction loop with confidence threshold
def chatbot():
    print("Medical Chatbot: Hello! I'm here to help with medical queries. Type 'quit' to exit.")
    confidence_threshold = 0.8
    while True:
        user_input = input('You: ')
        if user_input.lower() == 'quit':
            print("Medical Chatbot: Goodbye!")
            break
        if len(user_input.strip()) < 3:
            print("Medical Chatbot: Please provide more details about your query.")
            continue
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
            print("Medical Chatbot: I'm not sure I understand. Could you clarify your query?")
            logging.info(f"Low confidence ({confidence:.2f}) for input: {user_input} -> Processed: {user_input_processed} -> Predicted: {predicted_tag}")
            continue
        # Respond with a random response from the predicted intent
        response = random.choice(replies[predicted_tag])
        print("Medical Chatbot:", response)
        # Log prediction for debugging
        logging.info(f"Input: {user_input} -> Processed: {user_input_processed} -> Predicted: {predicted_tag} -> Confidence: {confidence:.2f}")

if __name__ == "__main__":
    chatbot()
