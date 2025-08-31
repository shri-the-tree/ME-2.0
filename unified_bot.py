"""
Unified Marine Edge Bot with Global Rate Limiting and Queueing
"""

import logging
import re
import os
import pickle
import faiss
import numpy as np
from typing import List, Tuple, Dict, Any, Optional
from datetime import datetime, timedelta
import asyncio
import time
from collections import deque
import threading
from dataclasses import dataclass
import uuid



# Web framework imports
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
import uvicorn

# Telegram imports
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes

# AI and document processing imports
from groq import Groq
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
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
MODEL_NAME = "llama3-70b-8192"
PDF_DIRECTORY = os.getenv("PDF_DIRECTORY", "./pdfs")
VECTOR_DB_DIRECTORY = os.getenv("VECTOR_DB_DIRECTORY", "./faiss_db")

# Rate limiting configuration
MAX_REQUESTS_PER_MINUTE = 25  # Safety buffer (Groq limit is 30)
MAX_TOKENS_PER_MINUTE = 5000  # Safety buffer (Groq limit is 6K)
ESTIMATED_TOKENS_PER_REQUEST = 500  # Conservative estimate

def format_response_for_channel(text: str, channel: str) -> str:
    """Format response appropriately for different channels"""
    if channel == "telegram":
        # Convert markdown to HTML for Telegram
        # Telegram HTML is more reliable than markdown
        text = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', text)  # **bold** to <b>bold</b>
        text = re.sub(r'\*(.*?)\*', r'<i>\1</i>', text)      # *italic* to <i>italic</i>
        return text
    elif channel == "fastapi":
        # Keep markdown for API responses
        return text
    else:
        # Strip markdown for plain text
        text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)
        text = re.sub(r'\*(.*?)\*', r'\1', text)
        return text

# System prompt
SYSTEM_PROMPT = """
You are Marine Edge Assistant, a specialized AI consultant for the Indian Maritime University Common Entrance Test (IMUCET) and DNS Sponsorship exams. You provide guidance on maritime education and Merchant Navy careers in India.

SAFETY AND RESPONSIBILITY GUIDELINES:
- Always encourage users to verify information through official sources (IMU, DGS, shipping companies)
- State clearly when information may be outdated or requires official confirmation
- For critical decisions (course selection, medical requirements, sponsorship applications), recommend consulting with Marine Edge counselors or official authorities
- Never guarantee admission, job placement, or sponsorship outcomes
- Acknowledge limitations: "This information is based on available data and should be verified"
- For medical or legal questions, direct users to qualified professionals

RESPONSE PROTOCOL:
1. Provide direct, actionable answers with structured formatting
2. Use clear headings and organized lists for complex information
3. Include specific data when available (dates, percentages, requirements)
4. End with relevant follow-up suggestions when helpful
5. Format critical information prominently but appropriately for the output channel

CONTENT BOUNDARIES:
- Focus strictly on maritime education, IMUCET, DNS programs, and related career guidance  
- Do not provide medical advice beyond general DGS fitness requirements
- Do not guarantee outcomes or make promises about admissions/placements
- Direct complex technical or legal questions to appropriate authorities
- Maintain professional, encouraging tone while being realistic about challenges

ACCURACY STANDARDS:
- Prioritize information from provided context and Marine Edge knowledge base
- When uncertain, clearly state limitations: "Based on available information..." 
- Encourage verification: "Please confirm current requirements with [specific authority]"
- Update disclaimers: "Requirements may have changed since last update"

FORMATTING APPROACH:
Use clear organization without excessive markup. Present information in scannable format with proper headings and structured lists.

Your mission: Provide responsible, accurate guidance that empowers students while encouraging proper verification and professional consultation for critical decisions.
"""

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)


@dataclass
class QueuedRequest:
    id: str
    query: str
    user_id: str
    timestamp: datetime
    response_channel: str  # 'fastapi' or 'telegram'
    telegram_update: Optional[Update] = None
    telegram_context: Optional[ContextTypes.DEFAULT_TYPE] = None
    future: Optional[asyncio.Future] = None


# ============================================================================
# GLOBAL RATE LIMITER AND QUEUE MANAGER
# ============================================================================

class GlobalRateLimiter:
    def __init__(self):
        self.request_queue = deque()
        self.request_timestamps = deque()
        self.token_usage = deque()
        self.processing = False
        self.lock = threading.Lock()

    def can_process_request(self) -> bool:
        """Check if we can process a request immediately"""
        now = datetime.now()

        # Clean old timestamps (older than 1 minute)
        self._clean_old_data(now)

        # Check request rate limit
        if len(self.request_timestamps) >= MAX_REQUESTS_PER_MINUTE:
            return False

        # Check token rate limit (estimated)
        current_token_usage = sum(usage for _, usage in self.token_usage)
        if current_token_usage + ESTIMATED_TOKENS_PER_REQUEST > MAX_TOKENS_PER_MINUTE:
            return False

        return True

    def _clean_old_data(self, now: datetime):
        """Remove timestamps older than 1 minute"""
        cutoff = now - timedelta(minutes=1)

        # Clean request timestamps
        while self.request_timestamps and self.request_timestamps[0] < cutoff:
            self.request_timestamps.popleft()

        # Clean token usage
        while self.token_usage and self.token_usage[0][0] < cutoff:
            self.token_usage.popleft()

    def add_request(self, request: QueuedRequest) -> Dict[str, Any]:
        """Add request to queue and return status"""
        with self.lock:
            if self.can_process_request() and len(self.request_queue) == 0:
                # Process immediately
                self.request_queue.append(request)
                return {
                    "status": "processing",
                    "message": "Processing your request...",
                    "position": 0,
                    "estimated_wait": 0
                }
            else:
                # Add to queue
                self.request_queue.append(request)
                position = len(self.request_queue)
                estimated_wait = position * 2.5  # ~2.5 seconds per request

                return {
                    "status": "queued",
                    "message": f"Request queued. Position #{position} in queue.",
                    "position": position,
                    "estimated_wait": int(estimated_wait)
                }

    def record_request_processed(self, tokens_used: int = ESTIMATED_TOKENS_PER_REQUEST):
        """Record that a request was processed"""
        now = datetime.now()
        self.request_timestamps.append(now)
        self.token_usage.append((now, tokens_used))

    def get_queue_status(self) -> Dict[str, Any]:
        """Get current queue status"""
        return {
            "queue_length": len(self.request_queue),
            "requests_last_minute": len(self.request_timestamps),
            "estimated_tokens_last_minute": sum(usage for _, usage in self.token_usage),
            "can_process_immediately": self.can_process_request()
        }


# Global rate limiter instance
rate_limiter = GlobalRateLimiter()


# ============================================================================
# CORE RAG SYSTEM (unchanged)
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
# BOT SERVICE WITH RATE LIMITING
# ============================================================================

class BotService:
    def __init__(self):
        if not GROQ_API_KEY:
            raise ValueError("GROQ_API_KEY not found in environment variables")

        self.client = Groq(api_key=GROQ_API_KEY)
        self.rag_system = UnifiedMarineEdgeRAG()
        self.processed_requests = {}

    async def process_request(self, request: QueuedRequest) -> str:
        """Process a single request and return response"""
        try:
            greetings = ['hi', 'hii', 'hello', 'hey', 'start', 'wassup', 'sup', 'yo']
            if request.query.lower().strip() in greetings:
                return "Hi! I'm Marine Edge Assistant. Ask me about IMUCET exams, DNS sponsorship, or Marine Edge courses!"
            # Get relevant context
            results = self.rag_system.hybrid_search(request.query, k=5)

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

USER QUERY: {request.query}

Please provide a helpful response based on the above context.
"""
            else:
                enhanced_input = request.query

            # Generate response using Groq
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
            bot_response = format_response_for_channel(bot_response, request.response_channel)

            # Record successful processing
            rate_limiter.record_request_processed()

            return bot_response

        except Exception as e:
            logger.error(f"Error processing request {request.id}: {str(e)}")
            return "I'm sorry, I encountered an error processing your request. Please try again."


# Global bot service
bot_service = BotService()


# ============================================================================
# BACKGROUND QUEUE PROCESSOR
# ============================================================================

async def process_queue():
    """Background task to process queued requests"""
    logger.info("Starting background queue processor...")

    while True:
        try:
            if rate_limiter.request_queue and rate_limiter.can_process_request():
                request = rate_limiter.request_queue.popleft()
                logger.info(f"Processing request {request.id} from {request.user_id}")

                response = await bot_service.process_request(request)

                if request.response_channel == "fastapi" and request.future:
                    request.future.set_result(response)
                elif request.response_channel == "telegram":
                    # Use sync version to avoid event loop conflicts
                    send_telegram_response_sync(request, response)

                await asyncio.sleep(2.5)
            else:
                await asyncio.sleep(1)

        except Exception as e:
            logger.error(f"Error in queue processor: {str(e)}")
            await asyncio.sleep(5)


def send_telegram_response_sync(request: QueuedRequest, response: str):
    """Send response to Telegram synchronously"""
    try:
        if request.telegram_update and request.telegram_context:
            # Store response for the message handler to pick up
            import threading

            def send_async():
                try:
                    import asyncio
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)

                    async def send_message():
                        await request.telegram_context.bot.send_message(
                            chat_id=request.telegram_update.effective_chat.id,
                            text=response,
                            parse_mode='HTML'
                        )

                    loop.run_until_complete(send_message())
                    loop.close()
                except Exception as e:
                    logger.error(f"Error in async send: {str(e)}")

            # Run in separate thread
            thread = threading.Thread(target=send_async)
            thread.start()

    except Exception as e:
        logger.error(f"Error sending Telegram response: {str(e)}")


# ============================================================================
# FASTAPI SETUP WITH RATE LIMITING
# ============================================================================

app = FastAPI(title="Marine Edge API", description="Rate Limited Marine Edge RAG System")


class ChatRequest(BaseModel):
    message: str
    user_id: str = "api_user"


class ChatResponse(BaseModel):
    response: str
    timestamp: str
    queue_info: Optional[Dict[str, Any]] = None


@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """Rate limited chat endpoint"""
    try:
        # Create queued request
        queued_request = QueuedRequest(
            id=str(uuid.uuid4()),
            query=request.message,
            user_id=request.user_id,
            timestamp=datetime.now(),
            response_channel="fastapi",
            future=asyncio.Future()
        )

        # Add to rate limiter queue
        queue_status = rate_limiter.add_request(queued_request)

        if queue_status["status"] == "processing":
            # Wait for response
            try:
                response = await asyncio.wait_for(queued_request.future, timeout=300)  # 5 minute timeout
                return ChatResponse(
                    response=response,
                    timestamp=datetime.now().isoformat()
                )
            except asyncio.TimeoutError:
                raise HTTPException(status_code=408, detail="Request timeout")
        else:
            # Return queue status
            return ChatResponse(
                response=f"Your request is in queue. Position #{queue_status['position']}, estimated wait: {queue_status['estimated_wait']} seconds.",
                timestamp=datetime.now().isoformat(),
                queue_info=queue_status
            )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/queue/status")
async def queue_status():
    """Get current queue status"""
    return rate_limiter.get_queue_status()


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "documents": len(bot_service.rag_system.documents),
        "queue_status": rate_limiter.get_queue_status()
    }


# ============================================================================
# TELEGRAM BOT WITH RATE LIMITING
# ============================================================================

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Telegram /start command"""
    await update.message.reply_html(
        "Hi! I'm the Marine Edge Assistant. Ask me about IMUCET, DNS sponsorship, and maritime courses!\n\n"
        "Note: Due to high demand, your request might be queued. I'll respond as soon as possible!"
    )


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Telegram /help command"""
    queue_info = rate_limiter.get_queue_status()
    await update.message.reply_html(
        f"Ask me about:\n"
        f"• IMUCET exam details\n"
        f"• Eligibility criteria\n"
        f"• Marine Edge courses\n"
        f"• DNS sponsorship\n"
        f"• Preparation strategies\n\n"
        f"Current queue: {queue_info['queue_length']} requests"
    )


async def handle_telegram_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle Telegram messages with rate limiting"""
    user_id = str(update.effective_user.id)
    user_input = update.message.text
    username = update.effective_user.username or "Unknown"

    logger.info(f"Telegram message from {username}: {user_input}")

    # Create queued request
    queued_request = QueuedRequest(
        id=str(uuid.uuid4()),
        query=user_input,
        user_id=user_id,
        timestamp=datetime.now(),
        response_channel="telegram",
        telegram_update=update,
        telegram_context=context
    )

    # Add to rate limiter queue
    queue_status = rate_limiter.add_request(queued_request)

    if queue_status["status"] == "queued":
        await update.message.reply_html(
            f"🚦 {queue_status['message']}\n"
            f"⏱️ Estimated wait: ~{queue_status['estimated_wait']} seconds\n"
            f"I'll respond as soon as I process your request!"
        )
    else:
        await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")


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


@app.on_event("startup")
async def startup_event():
    """Start background queue processor"""
    asyncio.create_task(process_queue())


def main():
    """Run the unified system with rate limiting"""
    logger.info("Starting Rate Limited Marine Edge Bot...")
    logger.info(f"Rate limits: {MAX_REQUESTS_PER_MINUTE} req/min, {MAX_TOKENS_PER_MINUTE} tokens/min")
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
