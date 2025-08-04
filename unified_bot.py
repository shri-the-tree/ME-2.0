#!/usr/bin/env python3
"""
Unified Marine Edge Bot - Single script to run everything
Combines Telegram bot, FastAPI endpoint, and core RAG functionality
"""

import logging
import re
import os
import pickle
import faiss
import numpy as np
from typing import List, Tuple, Dict, Any
from datetime import datetime

# Web framework imports
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

# Telegram imports
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes

# AI and document processing imports
from together import Together
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
from rank_bm25 import BM25Okapi
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# Environment setup
from dotenv import load_dotenv

load_dotenv()

# Download NLTK data
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('tokenizers/punkt_tab')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt')
    nltk.download('punkt_tab')
    nltk.download('stopwords')

# Configuration
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Llama-70B-free"
PDF_DIRECTORY = os.getenv("PDF_DIRECTORY", "./pdfs")
VECTOR_DB_DIRECTORY = os.getenv("VECTOR_DB_DIRECTORY", "./faiss_db")

# System prompt
SYSTEM_PROMPT = """
You are Marine Edge Assistant, a specialized AI consultant for the Indian Maritime University Common Entrance Test (IMUCET) and DNS Sponsorship exams. You provide authoritative guidance on maritime education and Merchant Navy careers in India.

Provide direct, actionable answers with structured formatting. Use bold for critical details and organize information clearly.
"""

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)


# ============================================================================
# CORE RAG SYSTEM
# ============================================================================

class UnifiedMarineEdgeRAG:
    def __init__(self):
        self.pdf_directory = PDF_DIRECTORY
        self.persist_directory = VECTOR_DB_DIRECTORY
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

        # FAISS components
        self.faiss_index = None
        self.documents = []

        # BM25 components
        self.bm25 = None
        self.tokenized_docs = []

        # File paths
        self.faiss_index_path = os.path.join(self.persist_directory, "faiss_index.bin")
        self.documents_path = os.path.join(self.persist_directory, "documents.pkl")
        self.bm25_path = os.path.join(self.persist_directory, "bm25.pkl")

        # Initialize
        self.create_or_load_vector_store()

    def create_or_load_vector_store(self, force_reload=False):
        """Create or load the hybrid vector store"""
        if (os.path.exists(self.faiss_index_path) and
                os.path.exists(self.documents_path) and
                os.path.exists(self.bm25_path) and not force_reload):
            logger.info("Loading existing vector store...")
            self._load_existing_store()
        else:
            logger.info("Creating new vector store...")
            self._create_new_store()

    def _load_existing_store(self):
        """Load existing indices"""
        self.faiss_index = faiss.read_index(self.faiss_index_path)
        with open(self.documents_path, 'rb') as f:
            self.documents = pickle.load(f)
        with open(self.bm25_path, 'rb') as f:
            bm25_data = pickle.load(f)
            self.bm25 = bm25_data['bm25']
            self.tokenized_docs = bm25_data['tokenized_docs']

    def _create_new_store(self):
        """Create new indices"""
        # Load documents
        loader = DirectoryLoader(self.pdf_directory, glob="**/*.pdf", loader_cls=PyPDFLoader)
        documents = loader.load()

        # Split documents
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_documents(documents)
        self.documents = chunks

        # Create embeddings and FAISS index
        texts = [doc.page_content for doc in chunks]
        embeddings_list = self.embeddings.embed_documents(texts)
        embeddings_array = np.array(embeddings_list).astype('float32')

        dimension = embeddings_array.shape[1]
        self.faiss_index = faiss.IndexFlatIP(dimension)
        faiss.normalize_L2(embeddings_array)
        self.faiss_index.add(embeddings_array)

        # Create BM25 index
        self.tokenized_docs = [self._preprocess_text(doc.page_content) for doc in chunks]
        self.bm25 = BM25Okapi(self.tokenized_docs)

        # Save everything
        self._save_store()
        logger.info(f"Created vector store with {len(self.documents)} documents")

    def _save_store(self):
        """Save indices to disk"""
        os.makedirs(self.persist_directory, exist_ok=True)
        faiss.write_index(self.faiss_index, self.faiss_index_path)

        with open(self.documents_path, 'wb') as f:
            pickle.dump(self.documents, f)

        with open(self.bm25_path, 'wb') as f:
            pickle.dump({'bm25': self.bm25, 'tokenized_docs': self.tokenized_docs}, f)

    def _preprocess_text(self, text):
        """Tokenize for BM25"""
        tokens = word_tokenize(text.lower())
        stop_words = set(stopwords.words('english'))
        return [token for token in tokens if token.isalnum() and token not in stop_words]

    def hybrid_search(self, query: str, k: int = 5) -> List[Tuple[Document, float]]:
        """Perform hybrid search"""
        # Semantic search
        query_embedding = np.array([self.embeddings.embed_query(query)]).astype('float32')
        faiss.normalize_L2(query_embedding)
        scores, indices = self.faiss_index.search(query_embedding, k * 2)

        semantic_results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.documents):
                semantic_results.append((self.documents[idx], float(score)))

        # Keyword search
        query_tokens = self._preprocess_text(query)
        bm25_scores = self.bm25.get_scores(query_tokens)
        top_indices = np.argsort(bm25_scores)[::-1][:k * 2]

        keyword_results = []
        for idx in top_indices:
            if idx < len(self.documents) and bm25_scores[idx] > 0:
                keyword_results.append((self.documents[idx], float(bm25_scores[idx])))

        # Combine results (70% semantic, 30% keyword)
        combined_scores = {}

        for doc, score in semantic_results:
            doc_id = id(doc)
            combined_scores[doc_id] = {'doc': doc, 'semantic': score, 'keyword': 0.0}

        for doc, score in keyword_results:
            doc_id = id(doc)
            if doc_id in combined_scores:
                combined_scores[doc_id]['keyword'] = score
            else:
                combined_scores[doc_id] = {'doc': doc, 'semantic': 0.0, 'keyword': score}

        # Calculate weighted scores
        final_results = []
        for info in combined_scores.values():
            final_score = 0.7 * info['semantic'] + 0.3 * info['keyword']
            final_results.append((info['doc'], final_score))

        final_results.sort(key=lambda x: x[1], reverse=True)
        return final_results[:k]


# ============================================================================
# BOT SERVICE
# ============================================================================

class BotService:
    def __init__(self):
        self.client = Together(api_key=TOGETHER_API_KEY)
        self.rag_system = UnifiedMarineEdgeRAG()
        self.user_conversations = {}

    def get_response(self, user_input: str, user_id: str = "default") -> str:
        """Generate response using RAG + LLM"""
        try:
            # Get relevant context
            results = self.rag_system.hybrid_search(user_input, k=5)

            context_texts = []
            for doc, score in results:
                if score > 0.1:
                    source = doc.metadata.get('source', 'Marine Edge Knowledge Base')
                    context_texts.append(f"SOURCE: {source}\nCONTENT: {doc.page_content}")

            # Create enhanced prompt
            if context_texts:
                enhanced_input = f"""
RELEVANT CONTEXT FROM MARINE EDGE KNOWLEDGE BASE:
{chr(10).join(context_texts)}

USER QUERY: {user_input}

Please provide a helpful response based on the above context.
"""
            else:
                enhanced_input = user_input

            # Generate response
            response = self.client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": enhanced_input}
                ],
                temperature=0.2,
                max_tokens=1024
            )

            bot_response = response.choices[0].message.content

            # Clean response
            bot_response = re.sub(r'<think>.*?</think>', '', bot_response, flags=re.DOTALL)
            bot_response = bot_response.strip()

            return bot_response

        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return "I'm sorry, I encountered an error. Please try again."


# ============================================================================
# FASTAPI SETUP
# ============================================================================

app = FastAPI(title="Marine Edge API", description="Unified Marine Edge RAG System")
bot_service = BotService()


class ChatRequest(BaseModel):
    message: str
    user_id: str = "api_user"


class ChatResponse(BaseModel):
    response: str
    timestamp: str


@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """Main chat endpoint"""
    try:
        response = bot_service.get_response(request.message, request.user_id)
        return ChatResponse(
            response=response,
            timestamp=datetime.now().isoformat()
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "documents": len(bot_service.rag_system.documents)}


# ============================================================================
# TELEGRAM BOT SETUP
# ============================================================================

user_conversations = {}


def clean_response(text):
    """Clean thinking tags from response"""
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    return text.strip()


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Telegram /start command"""
    await update.message.reply_text(
        "Hi! I'm the Marine Edge Assistant. Ask me about IMUCET, DNS sponsorship, and maritime courses!"
    )


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Telegram /help command"""
    await update.message.reply_text(
        "Ask me about:\n"
        "• IMUCET exam details\n"
        "• Eligibility criteria\n"
        "• Marine Edge courses\n"
        "• DNS sponsorship\n"
        "• Preparation strategies"
    )


async def handle_telegram_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle Telegram messages"""
    user_id = str(update.effective_user.id)
    user_input = update.message.text
    username = update.effective_user.username or "Unknown"

    logger.info(f"Telegram message from {username}: {user_input}")

    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")

    response = bot_service.get_response(user_input, user_id)
    await update.message.reply_text(response)


def setup_telegram_bot():
    """Setup Telegram bot"""
    if not TELEGRAM_TOKEN:
        logger.warning("TELEGRAM_TOKEN not found - Telegram bot disabled")
        return None

    application = Application.builder().token(TELEGRAM_TOKEN).build()
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_telegram_message))

    return application


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def run_telegram_in_background():
    """Run Telegram bot in background with proper event loop"""
    try:
        import asyncio
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        telegram_app = setup_telegram_bot()
        if telegram_app:
            logger.info("Telegram bot starting...")
            telegram_app.run_polling(allowed_updates=Update.ALL_TYPES)
    except Exception as e:
        logger.error(f"Telegram bot error: {str(e)}")


def main():
    """Run the unified system"""
    logger.info("Starting Unified Marine Edge Bot...")
    logger.info(f"RAG system loaded with {len(bot_service.rag_system.documents)} documents")

    # Start Telegram bot in background thread
    import threading
    telegram_thread = threading.Thread(target=run_telegram_in_background, daemon=True)
    telegram_thread.start()

    # Start FastAPI server (blocks main thread)
    logger.info("Starting FastAPI server on http://localhost:8000")
    logger.info("API docs available at http://localhost:8000/docs")

    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
