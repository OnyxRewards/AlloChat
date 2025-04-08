# AlloChat - Document Q&A

A web application that allows you to chat with your PDF documents using AI. Upload a PDF and ask questions about its content!

## Features

- 📄 Upload PDF documents
- 🤖 Chat with your documents using AI
- 💬 Real-time responses
- 🎯 Accurate document-based answers

## How to Use

1. Upload a PDF document using the sidebar
2. Wait for the document to be processed
3. Start asking questions about the document's content
4. Get instant AI-powered responses

## Local Development

To run the app locally:

1. Clone this repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Create a `.env` file with your OpenAI API key:
   ```
   OPENAI_API_KEY=your_api_key_here
   ```
4. Run the app:
   ```bash
   streamlit run app.py
   ```

## Deployment

This app is deployed on Streamlit Cloud. You can access it at: [Your Streamlit Cloud URL]

## Security Note

- The app processes PDFs locally and uses OpenAI's API for generating responses
- Your documents are not stored permanently
- API keys should be kept secure and not shared 