"""
Marine Edge Integrated Bot - Vision + RAG Hybrid System
Telegram bot with intelligent mode switching
"""

import logging
import os
import base64
import pickle
import faiss
import numpy as np
from io import BytesIO
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import asyncio
import re

# Telegram
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes

# AI APIs
from openai import OpenAI
from groq import Groq

# Document processing
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema import Document

# Try new import first, fallback to old
try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    from langchain_community.embeddings import HuggingFaceEmbeddings

from rank_bm25 import BM25Okapi
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# Environment
from dotenv import load_dotenv

load_dotenv()


# ============================================================================
# CONFIGURATION
# ============================================================================

# API Keys - robust loading with cleaning
def load_api_key(key_name: str) -> str:
    """Load and clean API key from environment"""
    key = os.getenv(key_name, "")
    # Remove quotes, spaces, newlines
    key = key.strip().strip('"').strip("'").replace('\n', '').replace('\r', '').replace(' ', '')
    return key


from dotenv import load_dotenv
load_dotenv()

OPENAI_API_KEY = ""
GROQ_API_KEY = load_api_key("GROQ_API_KEY")  # Keep this from .env
TELEGRAM_TOKEN = load_api_key("TELEGRAM_TOKEN")

# Model configuration
OPENAI_MODEL = "gpt-4o-mini"
GROQ_MODEL = "llama-3.1-8b-instant"

# Paths
PDF_DIRECTORY = os.getenv("PDF_DIRECTORY", "./pdfs")
VECTOR_DB_DIRECTORY = os.getenv("VECTOR_DB_DIRECTORY", "./faiss_db")

# Session limits
MAX_IMAGES_PER_SESSION = 5
SESSION_EXPIRY_HOURS = 24
MAX_HISTORY_MESSAGES = 6

# Logging setup
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO,
    handlers=[
        logging.FileHandler("integrated_bot.log", encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# NLTK setup
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('tokenizers/punkt_tab')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)
    nltk.download('stopwords', quiet=True)

# ============================================================================
# SYSTEM PROMPTS - Professional, no emojis
# ============================================================================

VISION_SYSTEM_PROMPT = """You are a Marine Edge educational assistant for IMUCET exam preparation. Help students understand concepts from question images.

Format responses for Telegram (plain text, no markdown):
- Use simple structure with clear sections
- Use "Subject:", "Topic:", "Step 1:", etc. as plain text labels
- Keep it professional but conversational
- Break down solutions step-by-step
- Check understanding with follow-up questions

Be clear, direct, and helpful."""

RAG_SYSTEM_PROMPT = """You are Marine Edge Assistant for IMUCET and DNS Sponsorship guidance.

Format for Telegram (plain text):
- Direct, actionable answers
- Use clear section labels without special formatting
- Present information in numbered/bulleted lists naturally
- Include specific data (dates, percentages, requirements)

Professional, knowledgeable tone."""


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class ImageContext:
    """Stores image data for vision mode"""
    id: int
    base64_data: str
    upload_time: datetime

    def to_api_format(self) -> dict:
        """Convert to OpenAI image format"""
        return {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{self.base64_data}"
            }
        }


@dataclass
class IntegratedSession:
    """Unified session tracking both modes"""
    user_id: str
    mode: str = "rag"  # "rag" or "vision"
    images: List[ImageContext] = field(default_factory=list)
    conversation_history: List[dict] = field(default_factory=list)
    last_activity: datetime = field(default_factory=datetime.now)

    def add_image(self, base64_data: str) -> int:
        """Add image and switch to vision mode"""
        image_id = len(self.images) + 1
        self.images.append(ImageContext(
            id=image_id,
            base64_data=base64_data,
            upload_time=datetime.now()
        ))
        self.mode = "vision"
        self.last_activity = datetime.now()
        logger.info(f"User {self.user_id}: Added image {image_id}, switched to vision mode")
        return image_id

    def add_to_history(self, role: str, content: str):
        """Add message to conversation history"""
        self.conversation_history.append({"role": role, "content": content})
        if len(self.conversation_history) > MAX_HISTORY_MESSAGES:
            self.conversation_history = self.conversation_history[-MAX_HISTORY_MESSAGES:]
        self.last_activity = datetime.now()

    def clear_session(self):
        """Reset session to RAG mode"""
        self.images.clear()
        self.conversation_history.clear()
        self.mode = "rag"
        self.last_activity = datetime.now()
        logger.info(f"User {self.user_id}: Session cleared, switched to RAG mode")

    def is_expired(self) -> bool:
        """Check if session has expired"""
        return datetime.now() - self.last_activity > timedelta(hours=SESSION_EXPIRY_HOURS)

    def get_images_for_api(self, limit: int = 3) -> List[dict]:
        """Get recent images in API format"""
        return [img.to_api_format() for img in self.images[-limit:]]


# Global session storage
user_sessions: Dict[str, IntegratedSession] = {}


def get_or_create_session(user_id: str) -> IntegratedSession:
    """Get or create user session"""
    user_id = str(user_id)
    if user_id not in user_sessions or user_sessions[user_id].is_expired():
        user_sessions[user_id] = IntegratedSession(user_id=user_id)
        logger.info(f"Created new session for user {user_id}")
    return user_sessions[user_id]


# ============================================================================
# RAG SYSTEM - From unified_bot.py
# ============================================================================

class RAGVectorStore:
    """Hybrid semantic + keyword search for RAG"""

    def __init__(self):
        self.pdf_directory = PDF_DIRECTORY
        self.persist_directory = VECTOR_DB_DIRECTORY
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

        self.faiss_index = None
        self.documents = []
        self.bm25 = None
        self.tokenized_docs = []

        self.faiss_index_path = os.path.join(self.persist_directory, "faiss_index.bin")
        self.documents_path = os.path.join(self.persist_directory, "documents.pkl")
        self.bm25_path = os.path.join(self.persist_directory, "bm25.pkl")

        self.initialize()

    def initialize(self):
        """Load or create vector store"""
        if (os.path.exists(self.faiss_index_path) and
                os.path.exists(self.documents_path) and
                os.path.exists(self.bm25_path)):
            logger.info("Loading existing RAG vector store...")
            self._load_existing()
        else:
            logger.info("Creating new RAG vector store...")
            self._create_new()

    def _load_existing(self):
        """Load from disk"""
        self.faiss_index = faiss.read_index(self.faiss_index_path)
        with open(self.documents_path, 'rb') as f:
            self.documents = pickle.load(f)

        # Handle both old and new BM25 pickle formats
        with open(self.bm25_path, 'rb') as f:
            bm25_data = pickle.load(f)
            if isinstance(bm25_data, dict):
                self.bm25 = bm25_data.get('bm25')
                self.tokenized_docs = bm25_data.get('tokenized_docs', [])
            else:
                # Old format: direct BM25 object
                self.bm25 = bm25_data
                self.tokenized_docs = []

        # Recreate if incomplete
        if not self.bm25 or not self.tokenized_docs:
            logger.warning("Incomplete BM25 index, recreating...")
            self._create_bm25_index()

        logger.info(f"Loaded {len(self.documents)} documents")

    def _create_new(self):
        """Create new vector store"""
        # Load PDFs
        loader = DirectoryLoader(self.pdf_directory, glob="**/*.pdf", loader_cls=PyPDFLoader)
        documents = loader.load()

        # Split into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_documents(documents)
        self.documents = chunks

        # Create FAISS index
        texts = [doc.page_content for doc in chunks]
        embeddings_list = self.embeddings.embed_documents(texts)
        embeddings_array = np.array(embeddings_list).astype('float32')

        dimension = embeddings_array.shape[1]
        self.faiss_index = faiss.IndexFlatIP(dimension)
        faiss.normalize_L2(embeddings_array)
        self.faiss_index.add(embeddings_array)

        # Create BM25 index
        self._create_bm25_index()

        # Save
        self._save()
        logger.info(f"Created vector store with {len(self.documents)} documents")

    def _create_bm25_index(self):
        """Create or recreate BM25 index from existing documents"""
        self.tokenized_docs = [self._tokenize(doc.page_content) for doc in self.documents]
        self.bm25 = BM25Okapi(self.tokenized_docs)
        logger.info(f"Created BM25 index with {len(self.tokenized_docs)} documents")

    def _save(self):
        """Save to disk"""
        os.makedirs(self.persist_directory, exist_ok=True)
        faiss.write_index(self.faiss_index, self.faiss_index_path)
        with open(self.documents_path, 'wb') as f:
            pickle.dump(self.documents, f)
        with open(self.bm25_path, 'wb') as f:
            pickle.dump({'bm25': self.bm25, 'tokenized_docs': self.tokenized_docs}, f)

    def _tokenize(self, text: str) -> List[str]:
        """Tokenize for BM25"""
        tokens = word_tokenize(text.lower())
        stop_words = set(stopwords.words('english'))
        return [t for t in tokens if t.isalnum() and t not in stop_words]

    def hybrid_search(self, query: str, k: int = 5) -> List[Tuple[Document, float]]:
        """Hybrid semantic + keyword search"""
        # Semantic search
        query_embedding = np.array([self.embeddings.embed_query(query)]).astype('float32')
        faiss.normalize_L2(query_embedding)
        scores, indices = self.faiss_index.search(query_embedding, k * 2)

        semantic_results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.documents):
                semantic_results.append((self.documents[idx], float(score)))

        # Keyword search
        query_tokens = self._tokenize(query)
        bm25_scores = self.bm25.get_scores(query_tokens)
        top_indices = np.argsort(bm25_scores)[::-1][:k * 2]

        keyword_results = []
        for idx in top_indices:
            if idx < len(self.documents) and bm25_scores[idx] > 0:
                keyword_results.append((self.documents[idx], float(bm25_scores[idx])))

        # Combine (70% semantic, 30% keyword)
        combined = {}
        for doc, score in semantic_results:
            doc_id = id(doc)
            combined[doc_id] = {'doc': doc, 'sem': score, 'key': 0.0}

        for doc, score in keyword_results:
            doc_id = id(doc)
            if doc_id in combined:
                combined[doc_id]['key'] = score
            else:
                combined[doc_id] = {'doc': doc, 'sem': 0.0, 'key': score}

        # Weighted score
        final = []
        for info in combined.values():
            final_score = 0.7 * info['sem'] + 0.3 * info['key']
            final.append((info['doc'], final_score))

        final.sort(key=lambda x: x[1], reverse=True)
        return final[:k]


# ============================================================================
# VISION SERVICE - OpenAI
# ============================================================================

class VisionService:
    """OpenAI vision API for image analysis"""

    def __init__(self):
        if not OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY not found in environment")

        self.client = OpenAI(api_key=OPENAI_API_KEY)
        logger.info(f"OpenAI client initialized (key length: {len(OPENAI_API_KEY)})")

        # In VisionService.analyze_initial(...)
    async def analyze_initial(self, base64_image: str, session: IntegratedSession) -> str:
        messages = [
            {"role": "system", "content": VISION_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text",
                        "text": "Analyze this question image. Identify the subject, topic, and key concepts. Then ask how I can help with this question."},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                ]
            }
        ]

        response = await self._generate(messages, max_tokens=600)

        # ✅ NEW: persist the initial exchange in conversation history
        session.add_to_history("user", "[Image uploaded] Analyze this question image.")
        session.add_to_history("assistant", response)

        return response

    async def answer_doubt(self, session: IntegratedSession, user_query: str) -> str:
        """Answer follow-up questions about images"""
        # Build content with query + images
        content = [{"type": "text", "text": user_query}]
        content.extend(session.get_images_for_api(limit=3))

        # Build messages with history
        messages = [{"role": "system", "content": VISION_SYSTEM_PROMPT}]
        messages.extend(session.conversation_history[-4:])
        messages.append({"role": "user", "content": content})

        response = await self._generate(messages, max_tokens=1024)

        # Update history
        session.add_to_history("user", user_query)
        session.add_to_history("assistant", response)

        return response

    async def _generate(self, messages: List[dict], max_tokens: int = 800) -> str:
        """Call OpenAI API"""
        try:
            response = await asyncio.to_thread(
                self.client.chat.completions.create,
                model=OPENAI_MODEL,
                messages=messages,
                max_tokens=max_tokens,
                temperature=0.7
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"OpenAI API error: {str(e)}")
            raise


# ============================================================================
# RAG SERVICE - Groq
# ============================================================================

class RAGService:
    """Groq-based RAG for text queries"""

    def __init__(self, vector_store: RAGVectorStore):
        if not GROQ_API_KEY:
            raise ValueError("GROQ_API_KEY not found in environment")

        self.client = Groq(api_key=GROQ_API_KEY)
        self.vector_store = vector_store
        logger.info(f"Groq client initialized (key length: {len(GROQ_API_KEY)})")

    async def answer_query(self, session: IntegratedSession, user_query: str) -> str:
        """Answer query using RAG"""
        # Handle greetings
        if user_query.lower().strip() in ['hi', 'hello', 'hey', 'start']:
            return ("**Welcome to Marine Edge Assistant**\n\n"
                    "I can help you with:\n"
                    "- IMUCET exam information and eligibility\n"
                    "- DNS sponsorship details\n"
                    "- Marine Edge courses and preparation\n"
                    "- Maritime education guidance\n\n"
                    "You can also upload question images for detailed explanations.\n\n"
                    "How can I assist you today?")

        # Get relevant context
        results = self.vector_store.hybrid_search(user_query, k=5)

        context_texts = []
        for doc, score in results:
            if score > 0.1:
                source = doc.metadata.get('source', 'Knowledge Base')
                context_texts.append(f"SOURCE: {source}\nCONTENT: {doc.page_content}")

        # Build prompt
        if context_texts:
            enhanced_query = f"""RELEVANT CONTEXT:
{chr(10).join(context_texts)}

USER QUERY: {user_query}

Provide a helpful response based on the context. Include all relevant details."""
        else:
            enhanced_query = user_query

        # Generate response
        messages = [
            {"role": "system", "content": RAG_SYSTEM_PROMPT},
            {"role": "user", "content": enhanced_query}
        ]

        response = await self._generate(messages)

        # Update history
        session.add_to_history("user", user_query)
        session.add_to_history("assistant", response)

        return response

    async def _generate(self, messages: List[dict], max_tokens: int = 1024) -> str:
        """Call Groq API"""
        try:
            response = await asyncio.to_thread(
                self.client.chat.completions.create,
                model=GROQ_MODEL,
                messages=messages,
                temperature=0.2,
                max_tokens=max_tokens
            )
            content = response.choices[0].message.content
            # Clean any thinking tags
            content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()
            return content
        except Exception as e:
            logger.error(f"Groq API error: {str(e)}")
            raise


# ============================================================================
# RESPONSE FORMATTER
# ============================================================================

def format_response(text: str) -> str:
    """Clean response for Telegram"""
    # Remove thinking tags
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)

    # Remove HTML-style bold (but keep content)
    text = re.sub(r'<b>(.*?)</b>', r'\1', text)
    text = re.sub(r'</?[^>]+>', '', text)

    # Clean excessive whitespace
    text = re.sub(r'\n\s*\n\s*\n', '\n\n', text)

    # Remove markdown symbols for cleaner look
    text = text.replace('**', '').replace('__', '').replace('###', '')

    return text.strip()


# ============================================================================
# SERVICE INITIALIZATION - Done in main() after config validation
# ============================================================================

rag_vector_store = None
rag_service = None
vision_service = None


# ============================================================================
# TELEGRAM HANDLERS
# ============================================================================

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /start"""
    await update.message.reply_text(
        "Marine Edge Integrated Assistant\n\n"
        "I provide two types of assistance:\n\n"
        "RAG Mode - General Questions\n"
        "Ask about IMUCET, DNS sponsorship, eligibility, courses, companies, etc.\n\n"
        "Vision Mode - Question Images\n"
        "Upload question images for detailed explanations and step-by-step solutions.\n\n"
        "The bot automatically detects whether your query is about an uploaded image "
        "or a general question and routes it accordingly.\n\n"
        "Commands:\n"
        "/help - Show this message\n"
        "/clear - Clear uploaded images\n"
        "/status - Check session info"
    )


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /help"""
    await update.message.reply_text(
        "How to Use Marine Edge Assistant\n\n"
        "For General Questions:\n"
        "Ask about IMUCET, DNS, eligibility, courses, companies, stipends, etc.\n"
        "Examples: 'What's the stipend at Fleet Management?', 'Tell me about IMUCET eligibility'\n\n"
        "For Question Images:\n"
        "1. Upload an image of your question\n"
        "2. Ask 'solve this' or 'explain this question'\n"
        "3. Upload more images if needed (max 5)\n\n"
        "The bot automatically detects whether you're asking about an image or general information.\n\n"
        "Commands:\n"
        "/start - Introduction\n"
        "/status - Check session info\n"
        "/clear - Clear uploaded images\n"
        "/help - This message"
    )


async def status_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /status"""
    user_id = str(update.effective_user.id)
    session = get_or_create_session(user_id)

    mode_name = "Vision Mode" if session.mode == "vision" else "RAG Mode"
    status_text = f"Current Session Status\n\n"
    status_text += f"Mode: {mode_name}\n"
    status_text += f"Images Uploaded: {len(session.images)}/{MAX_IMAGES_PER_SESSION}\n"
    status_text += f"Conversation Messages: {len(session.conversation_history)}\n"

    if session.images:
        status_text += f"\nImages in Session:\n"
        for img in session.images:
            mins_ago = (datetime.now() - img.upload_time).seconds // 60
            status_text += f"- Image {img.id} (uploaded {mins_ago}m ago)\n"

    await update.message.reply_text(status_text)


async def clear_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /clear"""
    user_id = str(update.effective_user.id)
    session = get_or_create_session(user_id)
    session.clear_session()

    await update.message.reply_text(
        "Session Cleared\n\n"
        "All images and conversation history have been cleared.\n"
        "Switched back to RAG Mode.\n\n"
        "You can now ask general questions or upload new images."
    )


async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle photo uploads"""
    user_id = str(update.effective_user.id)
    username = update.effective_user.username or "Unknown"

    logger.info(f"Photo received from {user_id} ({username})")

    try:
        await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")

        session = get_or_create_session(user_id)

        # Check limit
        if len(session.images) >= MAX_IMAGES_PER_SESSION:
            await update.message.reply_text(
                f"**Image Limit Reached**\n\n"
                f"Maximum {MAX_IMAGES_PER_SESSION} images per session.\n"
                f"Use /clear to start a new session.",
                parse_mode='Markdown'
            )
            return

        # Download and encode
        photo = update.message.photo[-1]
        file = await context.bot.get_file(photo.file_id)
        buffer = BytesIO()
        await file.download_to_memory(buffer)
        buffer.seek(0)
        base64_image = base64.b64encode(buffer.read()).decode('utf-8')

        logger.info(f"Image encoded for {user_id} (size: {len(base64_image)} chars)")

        # Analyze with Vision AI
        await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")

        initial_response = await vision_service.analyze_initial(base64_image, session)
        image_id = session.add_image(base64_image)

        formatted_response = format_response(initial_response)

        await update.message.reply_text(
            f"Image {image_id} Received\n\n{formatted_response}\n\n"
            f"Mode: Vision | Images: {len(session.images)}/{MAX_IMAGES_PER_SESSION}"
        )

        logger.info(f"Vision response sent to {user_id}")

    except Exception as e:
        logger.error(f"Error handling photo for {user_id}: {str(e)}", exc_info=True)
        await update.message.reply_text(
            "Error Processing Image\n\n"
            "There was an error processing your image. Please try uploading again.\n"
            "If the issue persists, try /clear to reset the session."
        )


def is_image_related_query(query: str, session: IntegratedSession) -> bool:
    """Decide if the user's query refers to uploaded images (Vision mode) or general info (RAG mode)."""
    has_images = bool(session.images)
    q = query.lower().strip()

    image_keywords = [
        'solve', 'explain', 'answer', 'solution', 'step', 'this', 'that',
        'above', 'question', 'problem', 'image', 'picture', 'photo',
        'help me with', 'work through', 'show me how', 'walkthrough', 'show steps'
    ]
    general_keywords = [
        'what is', 'what are', 'tell me about', 'eligibility', 'requirement',
        'criteria', 'exam', 'imucet', 'dns', 'sponsorship', 'stipend',
        'college', 'course', 'university', 'age limit', 'medical', 'salary',
        'company', 'ship', 'merchant navy', 'marine edge', 'career'
    ]
    short_followups = ['yes', 'ok', 'okay', 'yeah', 'yep', 'please', 'go ahead', 'continue', 'proceed', 'do it']

    # Case 1: No images uploaded → must be RAG
    if not has_images:
        return False

    # Case 2: If currently in Vision mode, bias heavily toward Vision
    if session.mode == "vision":
        if any(g in q for g in general_keywords):
            return False
        if any(k in q for k in image_keywords) or q in short_followups or len(q) <= 3:
            return True
        # Default: still prefer Vision mode
        return True

    # Case 3: Not currently in Vision mode, fall back to basic keyword check
    if any(g in q for g in general_keywords):
        return False
    if any(k in q for k in image_keywords):
        return True
    return False



async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle text messages - route to Vision or RAG based on mode"""
    user_id = str(update.effective_user.id)
    user_input = update.message.text
    username = update.effective_user.username or "Unknown"

    logger.info(f"Text from {user_id} ({username}): {user_input[:50]}")

    try:
        await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")

        session = get_or_create_session(user_id)

        # Intelligent routing: check if query is about images or general knowledge
        use_vision = is_image_related_query(user_input, session)

        if use_vision:
            # Vision mode - answer about images
            logger.info(f"Routing to Vision mode for {user_id} (image-related query)")
            response = await vision_service.answer_doubt(session, user_input)
            mode_indicator = f"Mode: Vision | Images: {len(session.images)}"
        else:
            # RAG mode - general queries
            logger.info(f"Routing to RAG mode for {user_id} (general query)")
            response = await rag_service.answer_query(session, user_input)
            mode_indicator = "Mode: RAG"

        formatted_response = format_response(response)

        await update.message.reply_text(
            f"{formatted_response}\n\n{mode_indicator}"
        )

        logger.info(f"Response sent to {user_id}")

    except Exception as e:
        logger.error(f"Error handling text for {user_id}: {str(e)}", exc_info=True)
        await update.message.reply_text(
            "Error Processing Request\n\n"
            "There was an error processing your request. Please try:\n"
            "- Rephrasing your question\n"
            "- Using /clear to reset the session\n"
            "- Trying again in a moment"
        )


# ============================================================================
# MAIN
# ============================================================================

def validate_config():
    """Validate API keys and configuration"""
    errors = []
    warnings = []

    # OpenAI Key validation
    if not OPENAI_API_KEY:
        errors.append("OPENAI_API_KEY not found or empty")
    else:
        logger.info(f"OpenAI key loaded: {len(OPENAI_API_KEY)} chars")
        logger.info(f"OpenAI key format: {OPENAI_API_KEY[:15]}...{OPENAI_API_KEY[-10:]}")

        if len(OPENAI_API_KEY) < 40:
            warnings.append(f"OPENAI_API_KEY appears short (length: {len(OPENAI_API_KEY)})")

        if not OPENAI_API_KEY.startswith('sk-'):
            errors.append(f"OPENAI_API_KEY has wrong format (starts with: {OPENAI_API_KEY[:5]})")

        # Check for common issues
        if '\n' in OPENAI_API_KEY or '\r' in OPENAI_API_KEY:
            errors.append("OPENAI_API_KEY contains newline characters")
        if ' ' in OPENAI_API_KEY:
            errors.append("OPENAI_API_KEY contains spaces")

    # Groq Key validation
    if not GROQ_API_KEY:
        errors.append("GROQ_API_KEY not found or empty")
    else:
        logger.info(f"Groq key loaded: {len(GROQ_API_KEY)} chars")
        logger.info(f"Groq key format: {GROQ_API_KEY[:15]}...{GROQ_API_KEY[-10:]}")

        if len(GROQ_API_KEY) < 40:
            warnings.append(f"GROQ_API_KEY appears short (length: {len(GROQ_API_KEY)})")

    # Telegram Token validation
    if not TELEGRAM_TOKEN:
        errors.append("TELEGRAM_TOKEN not found or empty")
    else:
        logger.info(f"Telegram token loaded: {len(TELEGRAM_TOKEN)} chars")

    # Report issues
    for warning in warnings:
        logger.warning(f"Configuration warning: {warning}")

    if errors:
        for error in errors:
            logger.error(f"Configuration error: {error}")
        raise ValueError("Configuration validation failed. Check .env file for:\n" + "\n".join(errors))

    logger.info("Configuration validated successfully")


def main():
    """Run the integrated bot"""
    global rag_vector_store, rag_service, vision_service

    logger.info("=" * 60)
    logger.info("Starting Marine Edge Integrated Bot")
    logger.info("=" * 60)

    # Validate configuration FIRST
    validate_config()

    # NOW initialize services with validated keys
    logger.info("Initializing services...")

    try:
        # Initialize RAG
        rag_vector_store = RAGVectorStore()
        rag_service = RAGService(rag_vector_store)

        # Initialize Vision
        vision_service = VisionService()

        logger.info("All services initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize services: {str(e)}")
        raise

    # Log system info
    logger.info(f"Vision Model: {OPENAI_MODEL}")
    logger.info(f"RAG Model: {GROQ_MODEL}")
    logger.info(f"RAG Documents: {len(rag_vector_store.documents)}")

    # Create Telegram application
    app = Application.builder().token(TELEGRAM_TOKEN).build()

    # Add handlers
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(CommandHandler("status", status_command))
    app.add_handler(CommandHandler("clear", clear_command))
    app.add_handler(MessageHandler(filters.PHOTO, handle_photo))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))

    logger.info("Bot started. Press Ctrl+C to stop.")
    logger.info("Default mode: RAG | Image upload switches to: Vision")
    logger.info("=" * 60)

    # Run
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
