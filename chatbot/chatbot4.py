import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Sample Conversations
conversations = [
    {"input": "Hi", "output": "Hello! How can I assist you today?"},
    {"input": "What services do you offer?", "output": "We offer a range of services including customer support, product information, and technical assistance."},
    {"input": "How can I contact customer support?", "output": "You can contact our customer support team at support@example.com or by phone at 1-800-123-4567."},
    {"input": "Thank you!", "output": "You're welcome!"},
    {"input": "Bye", "output": "Goodbye! Have a great day!"},
    {"input": "What is your name?", "output": "My name is Chatbot. How can I help you?"},
    {"input": "How are you?", "output": "I'm just a bot, but I'm here to assist you."},
    {"input": "What time is it?", "output": "The current time is [insert current time]."},
    {"input": "Can you help me?", "output": "Of course! What do you need assistance with?"},
    {"input": "What can you do?", "output": "I can help you with various tasks like providing information, answering questions, and offering assistance."},
    {"input": "Where are you located?", "output": "I exist in the digital realm and can assist you from anywhere with an internet connection."},
    {"input": "How old are you?", "output": "I don't have an age as I am a computer program."},
    {"input": "Do you have siblings?", "output": "No, I'm the only chatbot here."},
    {"input": "Tell me a joke", "output": "Why don't scientists trust atoms? Because they make up everything!"},
    {"input": "Can you sing?", "output": "I'm not equipped with vocal cords, but I can certainly provide song lyrics if you'd like."},
    {"input": "What's the weather like today?", "output": "I'm not connected to the internet to provide real-time weather updates, but you can check a weather website or app for the current forecast."},
    {"input": "What's your favorite color?", "output": "I don't have preferences like humans do, but I suppose I'd say my favorite color is binary code."},
    {"input": "Can you do math?", "output": "Yes, I can perform basic arithmetic calculations."},
    {"input": "Tell me about yourself", "output": "I'm a chatbot designed to assist users with their queries and tasks. Feel free to ask me anything!"},
    {"input": "How can I improve my programming skills?", "output": "Practice regularly, work on projects, and explore new technologies and concepts."},
    {"input": "What's the meaning of life?", "output": "That's a profound question! Philosophers have debated it for centuries."},
    {"input": "Are you a human?", "output": "No, I'm an artificial intelligence chatbot programmed to assist users."},
    {"input": "What's the capital of France?", "output": "The capital of France is Paris."},
    {"input": "Who is the president of the United States?", "output": "As of my last update, the president of the United States is [insert current president]."},
]

# Preprocessing
def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    tokens = [token for token in tokens if token.isalnum()]
    tokens = [token for token in tokens if token not in stopwords.words('english')]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    return ' '.join(tokens)

# Preprocess and tokenize input and output texts
inputs = [preprocess_text(conv['input']) for conv in conversations]
outputs = [conv['output'] for conv in conversations]

# Vectorize input texts
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(inputs)

# Train a simple Naive Bayes classifier
clf = MultinomialNB()
clf.fit(X, outputs)

# Intent Recognition using PyTorch
class IntentClassifier(nn.Module):
    def __init__(self, input_size, output_size):
        super(IntentClassifier, self).__init__()
        self.fc = nn.Linear(input_size, output_size)

    def forward(self, x):
        x = self.fc(x)
        return x

input_size = X.shape[1]
output_size = len(set(outputs))

# Convert inputs and outputs to tensors
X_tensor = torch.from_numpy(X.toarray()).float()
y_tensor = torch.tensor([outputs.index(label) for label in outputs]).long()

# Initialize model, loss function, and optimizer
model = IntentClassifier(input_size, output_size)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train model
num_epochs = 100
for epoch in range(num_epochs):
    optimizer.zero_grad()
    outputs = model(X_tensor)
    loss = criterion(outputs, y_tensor)
    loss.backward()
    optimizer.step()
    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Intent Recognition using trained classifier
def recognize_intent(input_text):
    preprocessed_input = preprocess_text(input_text)
    vectorized_input = vectorizer.transform([preprocessed_input])
    input_tensor = torch.from_numpy(vectorized_input.toarray()).float()
    output_tensor = model(input_tensor)
    _, predicted = torch.max(output_tensor, 1)
    return outputs[predicted.item()]

# Response Generation based on Intent
def generate_response(intent):
    for conv in conversations:
        if conv['output'] == intent:
            return conv['output']

# User Interaction
def chat():
    print("Chatbot: Hello! How can I assist you today?")
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            print("Chatbot: Goodbye! Have a great day!")
            break
        intent = recognize_intent(user_input)
        response = generate_response(intent)
        print("Chatbot:", response)

if __name__ == "__main__":
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')
    chat()
