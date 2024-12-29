import os
import nltk
import ssl
import streamlit as st
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


# SSL workaround for downloading NLTK data
ssl._create_default_https_context = ssl._create_unverified_context
nltk.data.path.append(os.path.abspath("nltk_data"))
nltk.download('punkt')

# Define intents
intents = [
    {
        "tag": "greeting",
        "patterns": ["Hi", "Hello", "Hey", "How are you", "What's up"],
        "responses": ["Hi there", "Hello", "Hey", "I'm fine, thank you", "Nothing much"]
    },
    {
        "tag": "goodbye",
        "patterns": ["Bye", "See you later", "Goodbye", "Take care"],
        "responses": ["Goodbye", "See you later", "Take care"]
    },
    {
        "tag": "thanks",
        "patterns": ["Thank you", "Thanks", "Thanks a lot", "I appreciate it"],
        "responses": ["You're welcome", "No problem", "Glad I could help"]
    },
    {
        "tag": "about",
        "patterns": ["What can you do", "Who are you", "What are you", "What is your purpose"],
        "responses": ["I am a chatbot", "My purpose is to assist you", "I can answer questions and provide assistance"]
    },
    {
        "tag": "help",
        "patterns": ["Help", "I need help", "Can you help me", "What should I do"],
        "responses": ["Sure, what do you need help with?", "I'm here to help. What's the problem?", "How can I assist you?"]
    },
    {
        "tag": "age",
        "patterns": ["How old are you", "What's your age"],
        "responses": ["I don't have an age. I'm a chatbot.", "I was just born in the digital world.", "Age is just a number for me."]
    },
    {
        "tag": "general_query",
        "patterns": ["Tell me something interesting", "What's the meaning of life?", "Why is the sky blue?"],
        "responses": ["That's a great question! Let me look that up for you.", "Life is what you make of it!", "The sky appears blue due to the scattering of sunlight by the atmosphere."]
    },
    {
        "tag": "weather",
        "patterns": ["What's the weather like", "How's the weather today"],
        "responses": ["I'm sorry, I cannot provide real-time weather information.", "You can check the weather on a weather app or website."]
    },
    {
        "tag": "small_talk",
        "patterns": ["How are you?", "What's your hobby?", "Do you like music?", "Tell me about yourself"],
        "responses": ["I'm just a chatbot, so I don't have hobbies, but I love helping you!", "I'm here to chat with you anytime!", "I can't listen to music, but I imagine it's wonderful."]
    },
    {
        "tag": "programming_help",
        "patterns": ["How to write a for loop in Python?", "What is an if statement?", "Can you explain recursion?"],
        "responses": [
            "A for loop in Python can be written as: `for i in range(5): print(i)`.",
            "An if statement is used for decision-making. For example: `if x > 5: print('x is greater than 5')`.",
            "Recursion is when a function calls itself. For example: `def factorial(n): return 1 if n == 0 else n * factorial(n-1)`."
        ]
    },
    {
        "tag": "jokes",
        "patterns": ["Tell me a joke", "Make me laugh", "Do you know any jokes?"],
        "responses": [
            "Why did the scarecrow win an award? Because he was outstanding in his field!",
            "Why don't skeletons fight each other? They don't have the guts!",
            "What do you call cheese that isn't yours? Nacho cheese!"
        ]
    },
    {
        "tag": "motivation",
        "patterns": ["I feel down", "Motivate me", "Can you give me a quote?"],
        "responses": [
            "Believe in yourself! Every great achievement was once considered impossible.",
            "You're stronger than you think! Keep pushing forward.",
            "The best way to predict the future is to create it. Keep going!"
        ]
    },
    {
        "tag": "current_events",
        "patterns": ["What's happening in the world?", "Tell me some news", "What's new today?"],
        "responses": [
            "I don't have access to live news, but you can check trusted news websites for updates.",
            "I'm not connected to live feeds, but let me know if there's a specific topic you'd like to discuss."
        ]
    },
    {
        "tag": "learning_recommendations",
        "patterns": ["How can I learn Python?", "Suggest resources for learning AI", "What are some good books?"],
        "responses": [
            "For Python, try Codecademy, freeCodeCamp, or Python.org's tutorial section.",
            "To learn AI, start with Andrew Ng's Machine Learning course on Coursera or fast.ai's deep learning course.",
            "For general learning, I recommend reading 'Atomic Habits' by James Clear or 'Deep Work' by Cal Newport."
        ]
    },
    {
        "tag": "personalized_responses",
        "patterns": ["Do you know me?", "What's my name?", "Are we friends?"],
        "responses": [
            "I may not know your name, but I'm here for you anytime!",
            "Of course, we're friends! I'm always happy to chat with you.",
            "I don't store personal data, but I enjoy our conversations!"
        ]
    },
    {
        "tag": "fallback",
        "patterns": [],
        "responses": ["I'm not sure about that. Could you rephrase?", "I don’t have an answer to that yet.", "Interesting! Let me learn about it and get back to you."]
    }
    
]


# Create the vectorizer and classifier
vectorizer = TfidfVectorizer()
clf = LogisticRegression(random_state=0, max_iter=10000)

# Preprocess the data
tags = []
patterns = []
for intent in intents:
    for pattern in intent['patterns']:
        tags.append(intent['tag'])
        patterns.append(pattern)

# training the model
x = vectorizer.fit_transform(patterns)
y = tags
clf.fit(x, y)


def chatbot(input_text):
    input_text = vectorizer.transform([input_text])
    tag = clf.predict(input_text)[0]
    for intent in intents:
        if intent['tag'] == tag:
            response = random.choice(intent['responses'])
            return response
        
counter = 0

# Streamlit app
def main():
    global counter
    st.title("Chatbot")
    st.write("Welcome to the chatbot. Please type a message and press Enter to start the conversation.")

    counter += 1
    user_input = st.text_input("You:", key=f"user_input_{counter}")

    if user_input:
        response = chatbot(user_input)
        st.text_area("Chatbot:", value=response, height=100, max_chars=None, key=f"chatbot_response_{counter}")

        if response.lower() in ['goodbye', 'bye']:
            st.write("Thank you for chatting with me. Have a great day!")
            st.stop()

if __name__ == '__main__':
    main()
