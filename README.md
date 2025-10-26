# RAG HR Chatbot 🤖

A Retrieval-Augmented Generation (RAG) chatbot that answers HR policy questions using local document knowledge and LLM-powered responses. Built with Flask, Streamlit, FAISS, and Groq LLM API.

## Features

- 🔍 Hybrid search combining FAISS (vector similarity) and BM25 (keyword matching)
- 💬 Interactive Streamlit UI with chat history
- 🚀 Fast responses using Groq's LLM API
- 📊 Document chunking and embedding using SentenceTransformers
- 🔒 Environment-based configuration for API keys
- 🐳 Docker support for easy deployment

## Project Structure

```
rag-hr-chatbot/
├── data/                    # Data files and indices
│   ├── hr_text.json        # Source HR documents
│   ├── hr_embeddings.npz   # Processed text chunks and embeddings
│   └── hr_index.faiss      # FAISS similarity search index
├── src/
│   ├── embedding_store.py  # Process documents and create embeddings
│   ├── retriever.py       # Build and manage FAISS index
│   ├── api.py            # Flask backend with Groq integration
│   └── app.py           # Streamlit frontend UI
├── docker/              # Docker configuration
└── requirements.txt    # Python dependencies
```

## Setup Instructions

### Local Development

1. Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. Install dependencies:

```powershell
python -m pip install -r requirements.txt
```

3. Set up your Groq API key:

   - Create a `.env` file in the project root
   - Add your Groq API key: `GROQ_API_KEY=your_key_here`

4. Process documents and create embeddings:

```powershell
python src/embedding_store.py  # Creates embeddings from HR documents
python src/retriever.py       # Builds FAISS index
```

5. Start the services:
   - Backend API (in one terminal):
     ```powershell
     python src/api.py  # Runs on http://localhost:5000
     ```
   - Frontend UI (in another terminal):
     ```powershell
     python -m streamlit run src/app.py  # Opens in browser
     ```

### Docker Deployment

Build and run the application using Docker:

```powershell
# Build the image
docker build -t rag-hr-chatbot -f docker/Dockerfile .

# Run the container (replace with your Groq API key)
docker run -p 8501:8501 -e GROQ_API_KEY="your_key_here" rag-hr-chatbot
```

Or use an environment file:

```powershell
# Create groq.env with your API key
echo "GROQ_API_KEY=your_key_here" > groq.env

# Run with env file
docker run -p 8501:8501 --env-file groq.env rag-hr-chatbot
```

## Usage

1. Open the Streamlit UI in your browser (default: http://localhost:8501)
2. Enter your HR policy question in the text input
3. Click "Ask" to get an AI-powered response
4. View past conversations in the left sidebar
5. Click on previous conversations to review them

## API Endpoints

### POST /query

Query the HR knowledge base and get an AI-generated response.

Request:

```json
{
  "question": "What is the leave policy?"
}
```

Response:

```json
{
  "answer": "...",
  "sources": ["..."] // Relevant document chunks used
}
```

## Development Notes

### Adding New HR Documents

1. Add your documents to `data/hr_text.json`
2. Run `embedding_store.py` to process them
3. Run `retriever.py` to update the search index

### Environment Variables

- `GROQ_API_KEY`: Your Groq API key (required)
- `FLASK_ENV`: Set to 'development' for debug mode
- `PORT`: Override default ports (Flask: 5000, Streamlit: 8501)

## Troubleshooting

### Common Issues

1. **ModuleNotFoundError: No module named 'faiss'**

   - On Windows, install via conda: `conda install -c pytorch faiss-cpu`
   - Or use pip: `pip install faiss-cpu`

2. **Groq API errors**

   - Verify your API key is set in `.env`
   - Check API rate limits and quotas
   - Ensure the model name in `api.py` matches available Groq models

3. **Docker build fails**
   - Some packages might need build tools. Add to Dockerfile:
     ```dockerfile
     RUN apt-get update && apt-get install -y build-essential
     ```

### Tips

- Use Python 3.9+ for best compatibility
- For large documents, adjust chunk size in `embedding_store.py`
- Monitor RAM usage with large document sets
- Consider using GPU-enabled FAISS for larger indices

## Contributing

1. Fork the repository
2. Create a feature branch
3. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.
