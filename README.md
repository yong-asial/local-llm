# Local LLM

This is simple python application to test LLM on your local pc.

## Prerequisites

### Ollama

You need to install ollama and download following models

- gemma:7b
- mistral:7b
- nomic-embed-text

### Python (Mac)

Python 3.10 is required

```bash
# install python3.10
brew install python@3.10

# install pip packages
python3.10 -m pip install langchain
```

Also, following packages are needed

- crewai
- langchain
- bs4
- chromadb

Note: this is just one way of installing python3 and pip packages.

## Run

Ask AI agents to write blog for any topic

```bash
python3.10 blog.py "AI and data science trends in 2024"
```

Ask AI about some questions with provided url as context

```bash
python3.10 ask.py "Who is current President in USA?" "https://en.wikipedia.org/wiki/President_of_the_United_States"
```

Note: Again, if you just have one python installed. Then you may run it with `python ask.py`
