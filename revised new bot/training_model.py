
import tensorflow as tf
import json
import numpy as np
import pandas as pd
import nltk
from tensorflow.keras.layers import Input, Embedding, Bidirectional, LSTM, Dense, GlobalMaxPooling1D, Dropout, Attention
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.utils.class_weight import compute_class_weight
import string
import pickle
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
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

# Load the medical dataset
with open('medicaldata.json') as file:
    data = json.load(file)

# Extract tags, queries, and responses
tags = []
queries = []
replies = {}

for intent in data['intents']:
    replies[intent['tag']] = intent['responses']
    for pattern in intent['patterns']:
        queries.append(pattern)
        tags.append(intent['tag'])

# Create DataFrame
df = pd.DataFrame({'text': queries, 'intent': tags})

# Preprocess text: remove punctuation, convert to lowercase, lemmatize, remove stopwords
def preprocess_text(text):
    text = [char.lower() for char in text if char not in string.punctuation]
    text = ''.join(text)
    words = nltk.word_tokenize(text)
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return ' '.join(words)

df['text'] = df['text'].apply(preprocess_text)

# Tokenize the text
tokenizer = Tokenizer(oov_token='<OOV>', num_words=5000)
tokenizer.fit_on_texts(df['text'])
sequences = tokenizer.texts_to_sequences(df['text'])

# Pad sequences
max_length = max(len(seq) for seq in sequences)
input_shape = max_length
padded_sequences = pad_sequences(sequences, maxlen=input_shape, padding='post')

# Encode labels
le = LabelEncoder()
encoded_labels = le.fit_transform(df['intent'])
num_classes = len(le.classes_)

# Compute class weights to handle imbalance
class_weights = compute_class_weight('balanced', classes=np.unique(encoded_labels), y=encoded_labels)
class_weight_dict = dict(enumerate(class_weights))

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(padded_sequences, encoded_labels, test_size=0.2, random_state=42)

# Build the LSTM model with attention mechanism
input_layer = Input(shape=(input_shape,))
embedding = Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=128)(input_layer)
lstm1 = Bidirectional(LSTM(256, return_sequences=True))(embedding)
attention = Attention()([lstm1, lstm1])
pooling = GlobalMaxPooling1D()(attention)
dropout1 = Dropout(0.5)(pooling)
dense = Dense(128, activation='relu')(dropout1)
dropout2 = Dropout(0.5)(dense)
output = Dense(num_classes, activation='softmax')(dropout2)

model = Model(inputs=input_layer, outputs=output)
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model with class weights
hist = model.fit(X_train, y_train, epochs=300, batch_size=16, validation_data=(X_test, y_test), class_weight=class_weight_dict, verbose=1)

# Save the model and preprocessing objects
model.save('medical_chatbot_model.h5')
with open('medical_chatbot_tokenizer.pkl', 'wb') as f:
    pickle.dump(tokenizer, f)
with open('medical_chatbot_labelencoder.pkl', 'wb') as f:
    pickle.dump(le, f)
with open('medical_chatbot_replies.pkl', 'wb') as f:
    pickle.dump(replies, f)
with open('medical_chatbot_maxlength.pkl', 'wb') as f:
    pickle.dump(max_length, f)

print("Model and preprocessing objects saved successfully.")

