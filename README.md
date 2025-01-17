# Retrieval-Augmented Financial Analysis and Knowledge Graph Application

This project is an advanced **Retrieval-Augmented Generation (RAG)** system for financial analysis, news insights, and relationship extraction. It leverages **Streamlit**, **Hugging Face Transformers**, and **Neo4j** to combine state-of-the-art NLP, graph-based knowledge representation, and financial data analysis.

---

## Features

### 1. **Data Collection**
- Fetches news articles using the **News API**.
- Retrieves historical stock data from **Yahoo Finance**.

### 2. **Natural Language Processing (NLP)**
- Performs **Named Entity Recognition (NER)** using **Hugging Face models**.
- Extracts relationships between financial entities.
- Conducts **sentiment analysis** for financial insights using **FinBERT**.

### 3. **Knowledge Graph**
- Creates and queries a knowledge graph in **Neo4j** to represent entities and their relationships.

### 4. **Retrieval-Augmented Generation (RAG)**
- Uses **FAISS** for similarity search.
- Integrates **ChatGroq** for AI-powered financial analysis based on contextual data.

### 5. **Visualization**
- Displays:
  - Knowledge graph relationships using **NetworkX**.
  - Stock trends with **Plotly** and **Matplotlib**.

### 6. **Streamlit Integration**
- Offers a user-friendly UI to:
  - Interact with NLP pipelines.
  - Query the knowledge graph.
  - Analyze financial data.

---

## Usage

1. **Data Input**: Enter stock ticker symbols or news search queries.
2. **Entity Analysis**: Extract entities and their relationships from financial texts.
3. **Knowledge Graph**: Visualize relationships in the Neo4j-powered graph.
4. **Sentiment Analysis**: Analyze sentiment trends in financial news or stock data.

---

## Tech Stack

- **Backend**: Python, Transformers, Neo4j, FAISS
- **Frontend**: Streamlit
- **APIs**: NewsAPI, Yahoo Finance
- **Visualization**: Plotly, Matplotlib, NetworkX

---

## Future Enhancements

- Add advanced relationship extraction models.
- Enhance the UI with personalized recommendations.
- Expand knowledge graph queries for deeper insights.

---

## Getting Started

### Prerequisites
- Python 3.8+
- Neo4j Community Edition or Enterprise Edition
- API keys for NewsAPI and Yahoo Finance
- Hugging Face Transformers library
- FAISS for similarity search

### Installation
1. Clone this repository:
   ```bash
   git clone https://github.com/AtharvaGhodki/Financial-Insights-Bot.git
   cd financial-knowledge-graph
