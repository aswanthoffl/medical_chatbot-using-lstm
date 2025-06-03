

# 🩺 Medical Chatbot using LSTM 

This project is a Medical Chatbot built with TensorFlow and Keras using an LSTM-based neural network model enhanced with attention mechanisms. It helps users by responding to medical queries based on pre-defined intents and responses. The project includes model training, preprocessing, and deployment via a Flask web application.

---

## 📂 Project Structure

```
medical_chatbot/
├── training_model.py                 # Script to train the LSTM model
├── demo_load_model.py               # CLI interface for testing chatbot
├── chatbot_app.py                   # Flask web app for chatbot interaction
├── medical_chatbot_model.h5         # Trained Keras LSTM model
├── medical_chatbot_tokenizer.pkl    # Tokenizer object for preprocessing
├── medical_chatbot_labelencoder.pkl # LabelEncoder for intent classification
├── medical_chatbot_replies.pkl      # Intent-to-response mapping
├── medical_chatbot_maxlength.pkl    # Max sequence length used for padding
├── templates/
│   └── chatbot.html                 # Frontend for Flask app (you may create this)
├── medicaldata.json                 # Training data (intents, patterns, responses)
└── chatbot_errors.log               # Runtime logs for debugging
```

---

## 🚀 How to Run

### 1. Install Dependencies

```bash
pip install tensorflow flask nltk scikit-learn pandas
```

### 2. Download NLTK Resources

The scripts auto-download required NLTK datasets. You can also download them manually:

```python
import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
```

### 3. Train the Model

To train the chatbot model:

```bash
python training_model.py
```

This will save the trained model and preprocessing artifacts.

### 4. Run Chatbot via Terminal (Optional)

```bash
python demo_load_model.py
```

This provides a command-line interface to chat with the bot.

### 5. Launch Web Application

```bash
python chatbot_app.py
```

Visit: [http://127.0.0.1:5000](http://127.0.0.1:5000)

---

## 🧠 Model Architecture

* Embedding Layer
* Bidirectional LSTM (256 units)
* Attention Layer
* GlobalMaxPooling1D
* Dense Layers with Dropout
* Softmax Output Layer

Trained using `sparse_categorical_crossentropy` with class weights to manage imbalance in intents.

---

## 🛠️ Preprocessing

* Lowercasing
* Punctuation removal
* Tokenization using NLTK
* Lemmatization
* Stopword removal
* Tokenization via Keras `Tokenizer`
* Sequence padding to max length

---

## 🧪 Example Query

User: `"What are the symptoms of flu?"`
Bot: `"Common symptoms of flu include fever, cough, sore throat, and body aches."`

---

## 📒 Notes

* Confidence threshold is set to `0.8` for prediction reliability.
* Logs are saved in `chatbot_errors.log`.
* Responses are chosen randomly from a pool corresponding to the predicted intent.

---

## 📧 Contact

Maintained by Aswanth A.

---

Let me know if you’d like this in Markdown format or if you want a sample chatbot.html template included too.
