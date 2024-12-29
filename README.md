# Chatbot with Intent Recognition

This project is a simple chatbot built using Python, Streamlit, and machine learning techniques. The chatbot can respond to a variety of user inputs based on predefined intents and patterns. It uses a logistic regression model to classify user inputs and generate appropriate responses.

## Features
- Responds to common queries like greetings, goodbyes, and thanks.
- Provides help with Python programming concepts.
- Shares motivational quotes and jokes.
- Offers learning recommendations for Python, AI, and books.
- Handles fallback cases for unrecognized inputs.
- Built-in SSL workaround for downloading NLTK data.

## Technologies Used
- **Python**: Core programming language for the chatbot.
- **Streamlit**: For building the user interface.
- **NLTK**: For natural language preprocessing.
- **scikit-learn**: For vectorization (TfidfVectorizer) and intent classification (Logistic Regression).

## Installation
1. Clone this repository:
   ```bash
   git clone https://github.com/sanjuraveer/chatbot.git
   cd chatbot
   ```

2. Create a virtual environment and activate it:
   ```bash
   python -m venv venv
   # Activate on Windows:
   venv\Scripts\activate
   # Activate on macOS/Linux:
   source venv/bin/activate
   ```

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Download the necessary NLTK data:
   ```bash
   python -c "import nltk; nltk.download('punkt')"
   ```

## Usage
1. Run the chatbot application:
   ```bash
   streamlit run chatbot.py
   ```

2. Open your browser and navigate to the URL provided by Streamlit (usually `http://localhost:8501`).

3. Start chatting with the chatbot by typing your message in the input box.

## File Structure
```
chatbot/
├── chatbot.py          # Main application script
├── nltk_data/          # Directory for NLTK data
├── requirements.txt    # Python dependencies
└── README.md           # Project documentation
```

## Example Intents
Below are some example intents and patterns supported by the chatbot:

| Tag             | Example Patterns                           |
|-----------------|--------------------------------------------|
| greeting        | Hi, Hello, Hey, How are you, What's up     |
| goodbye         | Bye, See you later, Goodbye, Take care     |
| thanks          | Thank you, Thanks, I appreciate it         |
| programming_help| How to write a for loop in Python?         |
| jokes           | Tell me a joke, Make me laugh              |
| motivation      | I feel down, Motivate me, Give me a quote  |
| learning_recommendations | How can I learn Python?, Good books |

## Customization
To add more intents:
1. Update the `intents` list in `chatbot.py` with new tags, patterns, and responses.
2. Retrain the model by running the script again.

## License
This project is licensed under the MIT License. See the LICENSE file for more information.

## Contributing
Contributions are welcome! Feel free to open issues or submit pull requests to improve the chatbot.

## Author
Developed by [Sanju Raj](https://github.com/sanjurajveer).
credits:Aman Kharwal

