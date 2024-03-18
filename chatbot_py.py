import nltk

nltk.download('punkt')
nltk.download('wordnet')
import random
import numpy as np
import json
import pickle
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model
lemmatizer=WordNetLemmatizer()

with open(r'C:/Users/NETCOM/Downloads/Health-Care-Chatbot-main/intents.json') as json_file:
    intents = json.load(json_file)

words=pickle.load(open(r'C:/Users/NETCOM/Downloads/Health-Care-Chatbot-main/words2.pkl','rb'))
classes=pickle.load(open(r'C:/Users/NETCOM/Downloads/Health-Care-Chatbot-main/classes2.pkl','rb'))
model=load_model(r'C:/Users/NETCOM/Downloads/Health-Care-Chatbot-main/chatbotmodel2.h5')

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bag_of_words(sentence):
  sentence_words=clean_up_sentence(sentence)
  bag=[0]*len(words)
  for w in sentence_words:
    for i,word in enumerate(words):
      if word == w:
        bag[i]=1
  return np.array(bag)

def predict_class(sentence):
  bow=bag_of_words(sentence)
  res=model.predict(np.array([bow]))[0]
  ERROR_THRESHOLD=0.25
  results=[[i,r] for i,r in enumerate(res) if r> ERROR_THRESHOLD]

  results.sort(key=lambda x:x[1],reverse=True)
  return_list=[]
  for r in results:
    return_list.append({'intent': classes[r[0]],'probability':str(r[1])})
  return return_list

def get_response(intents_list,intents_json):
  tag=intents_list[0]['intent']
  list_of_intents=intents_json['intents']
  for i in list_of_intents:
    if i['tag']==tag:
      result=random.choice(i['responses'])
      break
  return result

print("Hi! I am MedBot. How can I help you?")

# while True:
#   message=input("")
#   ints=predict_class(message)
#   res=get_response(ints,intents)
#   print(res)

import tkinter as tk

def send_message(event=None):
    message = entry.get()
    if message.strip() == "":
        return
    entry.delete(0, tk.END)
    response = "I am sorry, but I cannot answer that."
    ints = predict_class(message)
    res = get_response(ints, intents)
    response = res
    messages.config(state=tk.NORMAL)
    messages.insert(tk.END, "You: " + message + "\n")
    messages.insert(tk.END, "MedBot: " + response + "\n")
    messages.config(state=tk.DISABLED)
    messages.see(tk.END)

root = tk.Tk()
root.title("MedBot")

messages = tk.Text(root, width=50, height=20, state=tk.DISABLED)
messages.pack()

messages.config(state=tk.NORMAL)
messages.insert(tk.END, "MedBot: Hi! I am MedBot. How can I help you?\n")
messages.config(state=tk.DISABLED)

entry = tk.Entry(root, width=50)
entry.pack()
entry.bind("<Return>", send_message)

send_button = tk.Button(root, text="Send", command=send_message)
send_button.pack()

root.mainloop()
