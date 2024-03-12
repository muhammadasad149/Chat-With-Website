# Chat with Websites using Streamlit

This is a simple application that allows users to chat with websites using Streamlit, a popular Python library for building interactive web applications. The application leverages various natural language processing (NLP) techniques and external APIs to enable conversational interactions with websites.

## Features

- **Website Chat:** Users can enter the URL of a website and start a conversation with a bot.
- **Conversational AI:** The application uses a conversational AI model to generate responses based on user queries and context.
- **Web Document Parsing:** Web documents are parsed to extract relevant information for generating responses.
- **History Management:** Conversational history is managed to provide context-aware responses.
- **Settings:** Users can customize settings such as the website URL.

## Installation

1. Clone this repository:

    ```
    git clone https://github.com/muhammadasad149/Chat-With-Website.git
    ```

2. Navigate to the project directory:

    ```
    cd Chat-With-Website
    ```

3. Install the dependencies:

    ```
    pip install -r requirements.txt
    ```

## Usage

1. Run the Streamlit application:

    ```
    streamlit run app.py
    ```

2. Open your web browser and go to the provided URL (typically http://localhost:8501).
3. Enter the URL of the website you want to chat with in the sidebar.
4. Start chatting by typing your messages in the input box.

## Requirements

- Python 3.x
- Streamlit
- Other dependencies listed in `requirements.txt`

## Configuration

Before running the application, ensure you have set up the necessary environment variables. In particular, provide your Google API key in the `.env` file.

## Acknowledgments

- This project utilizes the capabilities of Streamlit, a powerful library for building interactive web applications in Python.
- We acknowledge the contributions of various open-source libraries and models used in this project, including LangChain, Google Generative AI, and others.
