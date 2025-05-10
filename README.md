# AutoGen RAG Agent

This is a RAG (Retrieval-Augmented Generation) agent implementation built on the AutoGen framework.

## Features

- Document processing and vector storage using LangChain
- Chroma as the vector database for efficient similarity search
- Multi-agent collaboration using the AutoGen framework
- Document retrieval and intelligent Q&A capabilities
- Support for various document formats and sizes
- Configurable retrieval parameters for optimal performance

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd agent_rag
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Configuration

1. Create a `.env` file in the root directory and add your OpenAI API key:
```
OPENAI_API_KEY=your_api_key_here
```

2. Place your documents in the `documents` directory (supports .txt format)

## Usage

1. Ensure your documents are placed in the `documents` directory
2. Run the example:
```bash
python rag_agent.py
```

## Customization

You can customize the RAG agent's behavior by modifying the following parameters:

- `chunk_size`: Size of document chunks for processing (default: 1000)
- `chunk_overlap`: Overlap between document chunks (default: 200)
- `k`: Number of documents to retrieve (default: 4)
- Model configuration in `config_list`:
  - Model selection
  - Temperature
  - Max tokens
  - Other model-specific parameters

## Project Structure

```
agent_rag/
├── documents/          # Directory for input documents
├── rag_agent.py       # Main implementation file
├── process_data.py   # Data preprocess module
├── requirements.txt   # Project dependencies
└── .env              # Environment variables
```

## Best Practices

- Ensure you have sufficient OpenAI API credits
- Use plain text format for documents
- Keep document sizes moderate for optimal retrieval
- Consider document structure and formatting
- Monitor API usage and costs

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 
