import streamlit as st
import pandas as pd
import numpy as np
import spacy
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    pipeline
)
import torch
from typing import List, Dict, Any
import neo4j
from neo4j import GraphDatabase
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores.faiss import FAISS
from langchain_groq import ChatGroq
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from newsapi import NewsApiClient
import yfinance as yf
import networkx as nx
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from datetime import datetime, timedelta
import json
import os
import logging
import re

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataCollector:
    """Handles data collection from various sources"""
    def __init__(self, api_keys: Dict[str, str]):
        self.newsapi = NewsApiClient(api_key=api_keys['newsapi_key'])
        
    def get_news(self, query: str, days: int = 7) -> List[Dict]:
        """Fetch news articles"""
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            articles = self.newsapi.get_everything(
                q=query,
                from_param=start_date.strftime('%Y-%m-%d'),
                to=end_date.strftime('%Y-%m-%d'),
                language='en',
                sort_by='relevancy'
            )
            return articles['articles']
        except Exception as e:
            logger.error(f"Error fetching news: {str(e)}")
            return []
    
    def get_stock_data(self, ticker: str, period: str = "1y") -> pd.DataFrame:
        """Fetch stock market data"""
        try:
            stock = yf.Ticker(ticker)
            return stock.history(period=period)
        except Exception as e:
            logger.error(f"Error fetching stock data: {str(e)}")
            return pd.DataFrame()

class NLPPipeline:
    """Handles all NLP tasks"""
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")
    
        # Initialize Hugging Face models and pipelines
        ner_model_name = "dbmdz/bert-large-cased-finetuned-conll03-english"  # Replace with your fine-tuned NER model if available
        re_model_name = "textattack/bert-base-uncased-SST-2"  # Replace with your fine-tuned RE model if available

        # Load tokenizer and models
        self.tokenizer = AutoTokenizer.from_pretrained(ner_model_name)
        self.ner_model = AutoModelForTokenClassification.from_pretrained(ner_model_name)
        self.re_model = AutoModelForSequenceClassification.from_pretrained(re_model_name)
        
         # Create a Hugging Face NER pipeline
        self.ner_pipeline = pipeline("ner", model=self.ner_model, tokenizer=self.tokenizer)

        self.sentiment_analyzer = pipeline(
            "sentiment-analysis",
            model="ProsusAI/finbert"
        )
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        
        
    def clean_text(self, text: str) -> str:
        """Clean and preprocess text"""
        text = re.sub(r'<[^>]+>', '', text)  # Remove HTML tags
        text = re.sub(r'\s+', ' ', text)     # Normalize whitespace
        return text.strip()
    
    def extract_entities(self,text):
        ner_results = self.ner_pipeline(text)
        entities = []
        for result in ner_results:
            if result["score"] > 0.85:  # Confidence threshold for filtering
                entities.append({
                    "entity": result["entity"],  # Entity type (e.g., ORG, PERSON)
                    "text": result["word"],  # Extracted entity text
                    "start": result["start"],  # Start position in the text
                    "end": result["end"]  # End position in the text
                })
        return entities
    
    def extract_relationships(self, text, entities):
        relationships = []
        for i, entity1 in enumerate(entities):
            for j, entity2 in enumerate(entities):
                if i < j:  # Avoid duplicate pairs
                    # Combine entity texts to form a relationship input
                    entity_pair = f"{entity1['text']} [SEP] {entity2['text']}"
                    
                    # Tokenize and pass to the relationship extraction model
                    inputs = self.tokenizer(entity_pair, return_tensors="pt")
                    outputs = self.re_model(**inputs)
                    
                    # Get probabilities and the most likely label
                    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
                    confidence, label_id = torch.max(probs, dim=-1)
                    
                    # Filter based on confidence threshold
                    if confidence > 0.85:
                        relationships.append({
                            "entity1": entity1["text"],
                            "entity2": entity2["text"],
                            "relationship": self.re_model.config.id2label[label_id.item()],
                            "confidence": confidence.item()
                        })
        return relationships
    
    def analyze_sentiment(self, text: str) -> Dict:
        """Analyze sentiment of text"""
        return self.sentiment_analyzer(text)[0]

class GraphDatabaseHandler:
    """Handles Neo4j graph database operations"""
    def __init__(self, uri: str, user: str, password: str):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        
    def close(self):
        """Closes the database connection."""
        self.driver.close()
        
    def create_entity(self, entity: Dict):
        """
        Creates or merges an entity in the Neo4j database.
        Args:
            entity (Dict): A dictionary containing entity details (text, type, start, end).
        """
        with self.driver.session() as session:
            session.run("""
                MERGE (e:Entity {text: $text})
                ON CREATE SET 
                    e.type = $type,
                    e.start = $start,
                    e.end = $end,
                    e.created = datetime()
            """, 
            {
                "text": entity["text"],
                "type": entity["entity"],
                "start": entity["start"],
                "end": entity["end"]
            })
            
    def create_relationship(self, relationship: Dict):
        """
        Creates or merges a relationship between two entities in the Neo4j database.
        Args:
            relationship (Dict): A dictionary containing relationship details (entity1, entity2, relationship, confidence).
        """
        with self.driver.session() as session:
            session.run("""
                MATCH (e1:Entity {text: $entity1})
                MATCH (e2:Entity {text: $entity2})
                MERGE (e1)-[r:RELATED {type: $relationship}]->(e2)
                ON CREATE SET 
                    r.confidence = $confidence,
                    r.created = datetime()
            """, 
            {
                "entity1": relationship["entity1"],
                "entity2": relationship["entity2"],
                "relationship": relationship["relationship"],
                "confidence": relationship["confidence"]
            })
            
    def query_graph(self, query: str) -> List[Dict]:
        """
        Executes a Cypher query on the Neo4j database and returns the results.
        Args:
            query (str): A Cypher query string.
        Returns:
            List[Dict]: A list of dictionaries representing the query results.
        """
        with self.driver.session() as session:
            result = session.run(query)
            return [record.data() for record in result]

    def query_knowledge_graph(self):
        query = """
        MATCH (e1:Entity)-[r:RELATED]->(e2:Entity)
        RETURN e1.text AS source, e2.text AS target, r.type AS relationship
        LIMIT 50
        """
        with self.driver.session() as session:
            results = session.run(query)
            graph_data = [{"source": record["source"], "target": record["target"], "relationship": record["relationship"]} for record in results]
        self.driver.close()
        return graph_data    

class RAGSystem:
    """Handles Retrieval-Augmented Generation with ChatGroq"""
    def __init__(self, groq_api_key: str):
        self.llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            api_key=groq_api_key,
            temperature=0.7,
            max_tokens=400
        )
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        
    def initialize_vector_store(self, texts: List[str]) -> FAISS:
        """Initialize FAISS vector store with texts and return it"""
        if not texts:
            return None
        
        try:
            vector_store = FAISS.from_texts(
                texts,
                self.embeddings
            )
            return vector_store
        except Exception as e:
            logger.error(f"Error initializing vector store: {str(e)}")
            return None
        
    def query(self, vector_store: FAISS, question: str) -> str:
        """Query the RAG system using ChatGroq"""
        if vector_store is None:
            return "Vector store not initialized. Please fetch some data first."
            
        try:
            # Get relevant documents
            docs = vector_store.similarity_search(question, k=3)
            
            # Create a more detailed prompt for the financial context
            prompt = PromptTemplate(
                input_variables=["context", "question"],
                template="""You are a financial analysis AI assistant. Use the following context to answer the question.
                If you can't answer the question based on the context, say so.

                Context:
                {context}

                Question: {question}

                Provide a detailed, analytical response. Include specific data points from the context when available.
                
                Answer:"""
            )
            
            # Create chain with ChatGroq
            chain = LLMChain(llm=self.llm, prompt=prompt)
            
            # Generate response
            response = chain.run(
                context="\n".join([doc.page_content for doc in docs]),
                question=question
            )
            
            return response
            
        except Exception as e:
            logger.error(f"Error in RAG query: {str(e)}")
            return f"An error occurred while processing your query. Please try again."

class StreamlitApp:
    """Main Streamlit application"""
    def __init__(self, config: Dict[str, str]):
        self.data_collector = DataCollector(config)
        self.nlp_pipeline = NLPPipeline()
        NEO4J_URI="neo4j+ssc://0b4b7a4a.databases.neo4j.io"
        NEO4J_USERNAME="neo4j"
        NEO4J_PASSWORD="Bg9xY7m3DnpisJRQEnV5hqelbGJaPyJDnMICEKL7qyk"
        self.graph_db = GraphDatabaseHandler(
            NEO4J_URI,
            NEO4J_USERNAME,
            NEO4J_PASSWORD
        )
        # self.graph_db = GraphDatabaseHandler(
        #     config['neo4j_uri'],
        #     config['neo4j_user'],
        #     config['neo4j_password']
        # )
        GROQ_API_KEY="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"
        self.rag_system = RAGSystem(GROQ_API_KEY)  # Updated to use Groq API key
        #self.rag_system = RAGSystem(config['groq_api_key'])  # Updated to use Groq API key

        # Initialize session state for vector store
        if 'vector_store' not in st.session_state:
            st.session_state.vector_store = None
        if 'has_data' not in st.session_state:
            st.session_state.has_data = False

    def run(self):
        st.set_page_config(page_title="Knowledge Graph RAG", layout="wide")
        st.title("Financial Knowledge Graph RAG System (Powered by ChatGroq)")
        
        # Sidebar
        with st.sidebar:
            st.header("Configuration")
            query = st.text_input("Search Query:", "Tech companies earnings")
            days = st.slider("Days of News:", 1, 30, 7)
            
        # Main content
        tab1, tab2, tab3 = st.tabs(["Data Collection", "Knowledge Graph", "Chat"])
        
        # Data Collection Tab
        with tab1:
            st.header("Data Collection")
            if st.button("Fetch New Data"):
                with st.spinner("Fetching and processing data..."):
                    try:
                        # Collect news
                        articles = self.data_collector.get_news(query, days)
                        
                        # Process articles
                        all_entities = []
                        all_relationships = []
                        all_texts = []
                        
                        progress_bar = st.progress(0)
                        for idx, article in enumerate(articles):
                            text = self.nlp_pipeline.clean_text(article['description'])
                            all_texts.append(text)
                            
                            # Extract information
                            entities = self.nlp_pipeline.extract_entities(text)
                            relationships = self.nlp_pipeline.extract_relationships(text,entities)
                            sentiment = self.nlp_pipeline.analyze_sentiment(text)
                            # print(entities)
                            # print(relationships)
                            # Store in graph
                            if entities:
                                for entity in entities:
                                    self.graph_db.create_entity(entity)
                                all_entities.extend(entities)

                            if relationships:
                                for rel in relationships:
                                    self.graph_db.create_relationship(rel)
                                all_relationships.extend(relationships)
                            
                            # Update progress
                            progress_bar.progress((idx + 1) / len(articles))
                        
                        # Initialize RAG
                        print(all_texts)
                        # Initialize RAG and store in session state
                        st.session_state.vector_store = self.rag_system.initialize_vector_store(all_texts)
                        st.session_state.has_data = True
                        print('created')
                        st.success(f"Successfully processed {len(articles)} articles")
                        
                        # Display summary
                        st.subheader("Processing Summary")
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Entities Extracted", len(all_entities))
                        with col2:
                            st.metric("Relationships Found", len(all_relationships))
                            
                    except Exception as e:
                        st.error(f"Error during data collection: {str(e)}")
                    
        # Knowledge Graph Tab
        with tab2:
            st.header("Knowledge Graph Visualization")
            if st.button("Show Knowledge Graph"):
                with st.spinner("Loading graph..."):
                    try:
                        # Query graph data
                        graph_data = self.graph_db.query_knowledge_graph()

                        # Create NetworkX graph
                        G = nx.DiGraph()
                        for edge in graph_data:
                            G.add_edge(edge["source"], edge["target"], relationship=edge["relationship"])

                        # Visualize the graph
                        plt.figure(figsize=(10, 6))
                        pos = nx.spring_layout(G)
                        nx.draw(G, pos, with_labels=True, node_color="lightblue", edge_color="gray", node_size=500, font_size=10, font_weight="bold")
                        st.pyplot(plt)
                    except Exception as e:
                        st.error(f"Failed to load the graph: {e}")

        
        # Chat Tab
        with tab3:
            st.header("Chat Interface (Powered by ChatGroq)")
            
            # Initialize chat history
            if "messages" not in st.session_state:
                st.session_state.messages = []

            # Display chat history
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

            # Chat input
            if prompt := st.chat_input("Ask about the financial data..."):
                # Add user message to history
                st.session_state.messages.append({"role": "user", "content": prompt})
                with st.chat_message("user"):
                    st.markdown(prompt)

                # Generate and display response
                with st.chat_message("assistant"):
                    with st.spinner("Generating response..."):
                        try:
                            response = self.rag_system.query(st.session_state.vector_store,prompt)
                            st.markdown(response)
                            st.session_state.messages.append({"role": "assistant", "content": response})
                        except Exception as e:
                            error_message = f"Error generating response: {str(e)}"
                            st.error(error_message)
                            st.session_state.messages.append({"role": "assistant", "content": error_message})

def main():
    # Load configuration
    config = {
        'newsapi_key': os.getenv('NEWSAPI_KEY'),
        'neo4j_uri': os.getenv('NEO4J_URI'),
        'neo4j_user': os.getenv('NEO4J_USER'),
        'neo4j_password': os.getenv('NEO4J_PASSWORD'),
        'groq_api_key': os.getenv('GROQ_API_KEY')  # Updated to use Groq API key
    }
    
    # Initialize and run app
    app = StreamlitApp(config)
    app.run()

if __name__ == "__main__":
    main()
