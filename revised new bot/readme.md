

# 🏥 Medical Chatbot using LSTM Model

This project implements a **Medical Chatbot** using a **Bidirectional LSTM model with an attention mechanism** built using **TensorFlow** and **Keras**. The chatbot classifies medical-related queries and responds with predefined answers using a web interface powered by **Flask**.

---

## 🔍 Overview

- **Natural Language Processing**: Tokenization, lemmatization, stopword removal (NLTK).
- **Deep Learning**: Bidirectional LSTM with attention for intent classification.
- **Web Interface**: Flask app with HTML front-end.
- **Confidence Threshold**: Rejects predictions with low confidence (< 0.8).
- **Logging**: All predictions and errors are logged to `chatbot_errors.log`.

---

## 🚀 Features

- Intelligent response generation based on intent recognition.
- Robust NLP preprocessing pipeline.
- Simple browser-based chat interface.
- Model training script included.
- Safe response filtering based on confidence level.

---

## 🧰 Prerequisites

- **Python**: 3.7 or higher (Anaconda recommended)
- Install dependencies:

```bash
pip install tensorflow flask numpy pandas nltk scikit-learn
````

* **NLTK Data**: Automatically downloads required data (punkt, wordnet, stopwords).
* **Dataset**: `medicaldata.json` with structure:

```json
{
  "intents": [
    {
      "tag": "intent_name",
      "patterns": ["query1", "query2", "..."],
      "responses": ["response1", "response2", "..."]
    },
    ...
  ]
}
```

---

## 📁 Project Structure

```
project_directory/
├── app.py                             # Flask application
├── train_medical_chatbot.py          # Model training script
├── templates/
│   └── chatbot.html                   # Chatbot UI
├── medicaldata.json                  # Dataset with intents and responses
├── medical_chatbot_model.h5          # Trained LSTM model
├── medical_chatbot_tokenizer.pkl     # Saved tokenizer
├── medical_chatbot_labelencoder.pkl  # Saved label encoder
├── medical_chatbot_replies.pkl       # Saved responses
├── medical_chatbot_maxlength.pkl     # Saved max sequence length
├── chatbot_errors.log                # Error and prediction log
```

---

## 🧠 Training the Model

Train the model and generate preprocessing artifacts:

```bash
python train_medical_chatbot.py
```

This will:

* Load and preprocess `medicaldata.json`
* Train a Bidirectional LSTM model with attention
* Save:

  * `medical_chatbot_model.h5`
  * `medical_chatbot_tokenizer.pkl`
  * `medical_chatbot_labelencoder.pkl`
  * `medical_chatbot_replies.pkl`
  * `medical_chatbot_maxlength.pkl`

---

## 💬 Running the Chatbot

Start the Flask app:

```bash
python app.py
```

Visit in browser:

```
http://127.0.0.1:5000
```

### ✅ Chatbot Behavior:

* Type your medical query and press "Send".
* Responses are based on the predicted intent.
* Type `quit` to end the session.
* Input with < 3 characters or low confidence gets a clarification prompt.

---

## 🧪 Example Interaction

```
User: What are symptoms of flu?
Chatbot: Medical Chatbot: Common flu symptoms include fever, cough, sore throat, body aches, and fatigue.

User: quit
Chatbot: Medical Chatbot: Goodbye!
```

---

## ⚙️ Notes

* Ensure all model files exist before running `app.py`.
* If port 5000 is busy, change the port:

  ```python
  app.run(debug=True, port=5001)
  ```
* Suppress TensorFlow oneDNN warnings:

  ```bash
  set TF_ENABLE_ONEDNN_OPTS=0
  ```

---

## 🛠️ Troubleshooting

| Issue            | Solution                                           |                                     |
| ---------------- | -------------------------------------------------- | ----------------------------------- |
| Page not loading | Verify `app.py` and `templates/chatbot.html` exist |                                     |
| No response      | Check browser console and Flask logs               |                                     |
| Port in use      | Use \`netstat -a -n -o                             | find "5000"\` to identify conflicts |

---

## 📜 License

This project is for **educational purposes only**. It uses open-source libraries under their respective licenses (TensorFlow, Flask, NLTK, etc.).

---

## 📬 Contact

For issues or suggestions, please raise an issue in this GitHub repository or contact the developer.



