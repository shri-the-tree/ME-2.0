"""
Marine Edge Vision Doubt Clarification Bot - Dual System
Primary: OpenAI GPT-4o | Backup: Groq Llama Vision
Optimized for performance and reliability
"""

import logging
import os
import base64
from io import BytesIO
from typing import Dict, List, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import asyncio
from collections import deque

# Telegram imports
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes

# AI APIs
from openai import OpenAI
from groq import Groq

# Environment setup
from dotenv import load_dotenv

from dotenv import load_dotenv
load_dotenv(override=True)

# Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN_VISION")

# Model configuration
OPENAI_MODEL = "gpt-4o"
GROQ_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"

# Rate limiting (conservative for OpenAI)
MAX_REQUESTS_PER_MINUTE = 30
MAX_TOKENS_PER_MINUTE = 8000
ESTIMATED_TOKENS_PER_REQUEST = 600

# Session configuration
MAX_IMAGES_PER_USER = 5
IMAGE_EXPIRY_HOURS = 24

# Configure logging (UTF-8 encoding for Windows)
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO,
    handlers=[
        logging.FileHandler("vision_bot_dual.log", encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Force UTF-8 output for Windows console
import sys
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

# Optimized system prompt (reduced tokens)
SYSTEM_PROMPT = """You're a friendly Marine Edge tutor for IMUCET prep. Help students understand concepts clearly.

**Style:**
- Warm, encouraging, patient
- Break down complex ideas simply
- Ask guiding questions
- Provide hints or full solutions as needed
- Connect to real applications

**Response format:**
- Identify subject/topic
- Explain core concept if needed
- Step-by-step when helpful
- Check understanding

Be concise but thorough. Make learning positive!"""


@dataclass
class ImageContext:
    """Stores uploaded image info"""
    id: int
    base64_data: str
    upload_time: datetime
    brief_description: str = ""

    def to_message_format(self) -> dict:
        """Convert to API image format (works for both OpenAI and Groq)"""
        return {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{self.base64_data}"
            }
        }


@dataclass
class UserSession:
    """Manages user session with images and conversation"""
    user_id: str
    images: List[ImageContext] = field(default_factory=list)
    conversation_history: List[dict] = field(default_factory=list)
    last_activity: datetime = field(default_factory=datetime.now)
    model_stats: Dict[str, int] = field(default_factory=lambda: {"openai": 0, "groq": 0, "failed": 0})

    def add_image(self, base64_data: str) -> int:
        """Add image and return ID"""
        image_id = len(self.images) + 1
        self.images.append(ImageContext(
            id=image_id,
            base64_data=base64_data,
            upload_time=datetime.now()
        ))
        self.last_activity = datetime.now()
        return image_id

    def get_images_for_api(self, limit: int = 3) -> List[dict]:
        """Get recent images in API format (limit for token optimization)"""
        return [img.to_message_format() for img in self.images[-limit:]]

    def add_to_history(self, role: str, content: str):
        """Add to conversation history (keep last 6 messages for context)"""
        self.conversation_history.append({"role": role, "content": content})
        if len(self.conversation_history) > 6:
            self.conversation_history = self.conversation_history[-6:]
        self.last_activity = datetime.now()

    def clear_images(self):
        """Clear session"""
        self.images.clear()
        self.conversation_history.clear()
        self.last_activity = datetime.now()

    def is_expired(self) -> bool:
        """Check expiry"""
        return datetime.now() - self.last_activity > timedelta(hours=IMAGE_EXPIRY_HOURS)


# ============================================================================
# RATE LIMITER
# ============================================================================

class GlobalRateLimiter:
    def __init__(self):
        self.request_timestamps = deque()
        self.token_usage = deque()
        self.lock = asyncio.Lock()

    async def wait_if_needed(self):
        """Wait if at rate limit"""
        async with self.lock:
            now = datetime.now()
            self._clean_old_data(now)

            while (len(self.request_timestamps) >= MAX_REQUESTS_PER_MINUTE or
                   sum(usage for _, usage in self.token_usage) >= MAX_TOKENS_PER_MINUTE):
                await asyncio.sleep(2)
                now = datetime.now()
                self._clean_old_data(now)

            self.request_timestamps.append(now)
            self.token_usage.append((now, ESTIMATED_TOKENS_PER_REQUEST))

    def _clean_old_data(self, now: datetime):
        """Remove old timestamps"""
        cutoff = now - timedelta(minutes=1)
        while self.request_timestamps and self.request_timestamps[0] < cutoff:
            self.request_timestamps.popleft()
        while self.token_usage and self.token_usage[0][0] < cutoff:
            self.token_usage.popleft()


rate_limiter = GlobalRateLimiter()
user_sessions: Dict[str, UserSession] = {}


# ============================================================================
# DUAL VISION SERVICE (OpenAI Primary, Groq Backup)
# ============================================================================

class DualVisionService:
    def __init__(self):
        # Initialize both clients
        self.openai_client = None
        self.groq_client = None
        self.openai_available = False
        self.groq_available = False

        if OPENAI_API_KEY:
            self.openai_client = OpenAI(api_key=OPENAI_API_KEY)
            logger.info("OpenAI client initialized")
        else:
            logger.warning("OPENAI_API_KEY not found")

        if GROQ_API_KEY:
            self.groq_client = Groq(api_key=GROQ_API_KEY)
            logger.info("Groq client initialized")
        else:
            logger.warning("GROQ_API_KEY not found")

        if not self.openai_client and not self.groq_client:
            raise ValueError("No API keys found! Need at least one: OPENAI_API_KEY or GROQ_API_KEY")

    async def health_check(self):
        """Test both APIs to verify they work"""
        logger.info("Running health checks...")

        # Test OpenAI
        if self.openai_client:
            try:
                await asyncio.to_thread(
                    self.openai_client.chat.completions.create,
                    model=OPENAI_MODEL,
                    messages=[{"role": "user", "content": "Hi"}],
                    max_tokens=5
                )
                self.openai_available = True
                logger.info("✓ OpenAI: Healthy")
            except Exception as e:
                logger.error(f"✗ OpenAI: Failed - {str(e)[:100]}")

        # Test Groq
        if self.groq_client:
            try:
                await asyncio.to_thread(
                    self.groq_client.chat.completions.create,
                    model=GROQ_MODEL,
                    messages=[{"role": "user", "content": "Hi"}],
                    max_completion_tokens=5
                )
                self.groq_available = True
                logger.info("✓ Groq: Healthy")
            except Exception as e:
                logger.error(f"✗ Groq: Failed - {str(e)[:100]}")

        if not self.openai_available and not self.groq_available:
            raise RuntimeError("No AI services available! Check API keys and credits.")

        # Log primary service
        if self.openai_available:
            logger.info("Primary: OpenAI ✓")
        if self.groq_available:
            logger.info(f"Backup: Groq {'✓' if not self.openai_available else '(standby)'}")

        return self.openai_available or self.groq_available

    async def _call_openai(self, messages: List[dict], max_tokens: int = 800) -> tuple[str, bool]:
        """Call OpenAI API - returns (response, success)"""
        try:
            response = await asyncio.to_thread(
                self.openai_client.chat.completions.create,
                model=OPENAI_MODEL,
                messages=messages,
                max_tokens=max_tokens,
                temperature=0.7
            )
            return response.choices[0].message.content, True
        except Exception as e:
            logger.error(f"OpenAI API error: {str(e)}")
            return str(e), False

    async def _call_groq(self, messages: List[dict], max_tokens: int = 800) -> tuple[str, bool]:
        """Call Groq API - returns (response, success)"""
        try:
            response = await asyncio.to_thread(
                self.groq_client.chat.completions.create,
                model=GROQ_MODEL,
                messages=messages,
                max_completion_tokens=max_tokens,
                temperature=0.7
            )
            return response.choices[0].message.content, True
        except Exception as e:
            logger.error(f"Groq API error: {str(e)}")
            return str(e), False

    async def generate_response(self, messages: List[dict], session: UserSession, max_tokens: int = 800) -> str:
        """
        Generate response with fallback logic
        Try OpenAI first, fallback to Groq if fails
        """
        await rate_limiter.wait_if_needed()

        # Try OpenAI first
        if self.openai_client:
            logger.info("Attempting OpenAI...")
            response, success = await self._call_openai(messages, max_tokens)
            if success:
                session.model_stats["openai"] += 1
                logger.info("✓ OpenAI success")
                return response
            logger.warning("✗ OpenAI failed, trying Groq...")

        # Fallback to Groq
        if self.groq_client:
            logger.info("Attempting Groq...")
            response, success = await self._call_groq(messages, max_tokens)
            if success:
                session.model_stats["groq"] += 1
                logger.info("✓ Groq success")
                return response
            logger.error("✗ Groq also failed")

        # Both failed
        session.model_stats["failed"] += 1
        return "I'm having trouble connecting right now. Please try again in a moment!"

    async def analyze_image_initial(self, base64_image: str, session: UserSession) -> str:
        """Initial image analysis - optimized prompt"""
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Analyze this question briefly: subject, topic, key challenge. Then warmly greet and ask what I need help with. Be concise!"
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
                    }
                ]
            }
        ]

        return await self.generate_response(messages, session, max_tokens=512)

    async def answer_doubt(self, user_session: UserSession, user_query: str) -> str:
        """Answer doubt with conversation context"""
        # Build content with images (limit to 3 most recent for token optimization)
        content = [{"type": "text", "text": user_query}]
        content.extend(user_session.get_images_for_api(limit=3))

        # Build messages with limited history
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]

        # Add conversation history (already limited to 6 in UserSession)
        messages.extend(user_session.conversation_history[-4:])

        # Add current query
        messages.append({"role": "user", "content": content})

        response = await self.generate_response(messages, user_session, max_tokens=1024)

        # Update conversation history
        user_session.add_to_history("user", user_query)
        user_session.add_to_history("assistant", response)

        return response


vision_service = DualVisionService()


# ============================================================================
# SESSION MANAGEMENT
# ============================================================================

def get_or_create_session(user_id: str) -> UserSession:
    """Get or create user session"""
    user_id = str(user_id)
    if user_id not in user_sessions or user_sessions[user_id].is_expired():
        user_sessions[user_id] = UserSession(user_id=user_id)
        logger.info(f"New session for user {user_id}")
    return user_sessions[user_id]


# ============================================================================
# TELEGRAM HANDLERS
# ============================================================================

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /start"""
    await update.message.reply_text(
        "👋 Hey! I'm your Marine Edge Doubt Clarification Assistant!\n\n"
        "📸 **How to use:**\n"
        "1. Upload question image\n"
        "2. I'll analyze it\n"
        "3. Ask anything!\n\n"
        "💡 **I help with:**\n"
        "• Concept explanations\n"
        "• Step-by-step solutions\n"
        "• Hints & clarifications\n\n"
        "📚 Perfect for IMUCET prep!\n\n"
        "Commands: /help /clear /stats"
    )


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /help"""
    await update.message.reply_text(
        "🆘 **Quick Guide:**\n\n"
        "📸 Upload question images (up to 5)\n"
        "💬 Ask questions about them\n"
        "🎯 Compare multiple images\n\n"
        "**Commands:**\n"
        "/clear - Fresh start\n"
        "/stats - See your session info\n"
        "/help - This message\n\n"
        "Ask away! 😊"
    )


async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle photo uploads"""
    user_id = str(update.effective_user.id)
    username = update.effective_user.username or "Unknown"

    logger.info(f"Photo from {user_id} ({username})")

    try:
        await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")

        session = get_or_create_session(user_id)

        # Check limit
        if len(session.images) >= MAX_IMAGES_PER_USER:
            await update.message.reply_text(
                f"⚠️ Max {MAX_IMAGES_PER_USER} images reached.\n"
                "Use /clear to start fresh!"
            )
            return

        # Download and encode
        photo = update.message.photo[-1]
        file = await context.bot.get_file(photo.file_id)
        buffer = BytesIO()
        await file.download_to_memory(buffer)
        buffer.seek(0)
        base64_image = base64.b64encode(buffer.read()).decode('utf-8')

        logger.info(f"Image encoded for {user_id}")

        # Analyze with AI
        await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")

        initial_response = await vision_service.analyze_image_initial(base64_image, session)
        image_id = session.add_image(base64_image)

        logger.info(f"Image {image_id} added for {user_id}")

        await update.message.reply_text(f"📸 **Image {image_id}**\n\n{initial_response}")

    except Exception as e:
        logger.error(f"Photo error for {user_id}: {str(e)}")
        await update.message.reply_text(
            "😅 Had trouble with that image. Try uploading again?"
        )


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle text messages"""
    user_id = str(update.effective_user.id)
    user_input = update.message.text
    username = update.effective_user.username or "Unknown"

    logger.info(f"Message from {user_id} ({username}): {user_input[:50]}")

    try:
        await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")

        session = get_or_create_session(user_id)

        # Check for images
        if not session.images:
            await update.message.reply_text(
                "👋 Hey! Upload a question image first, then I can help! 📸"
            )
            return

        # Generate response
        response = await vision_service.answer_doubt(session, user_input)
        await update.message.reply_text(response)

        logger.info(f"Response sent to {user_id}")

    except Exception as e:
        logger.error(f"Message error for {user_id}: {str(e)}")
        await update.message.reply_text(
            "Error processing your question. Try rephrasing?"
        )


async def clear_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /clear"""
    user_id = str(update.effective_user.id)
    session = get_or_create_session(user_id)
    session.clear_images()

    logger.info(f"Session cleared for {user_id}")

    await update.message.reply_text(
        "✨ **Fresh start!**\n\n"
        "All cleared. Upload new questions! 📸"
    )


async def stats_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /stats"""
    user_id = str(update.effective_user.id)
    session = get_or_create_session(user_id)

    if not session.images:
        await update.message.reply_text(
            "📊 **Session Status**\n\n"
            "No images yet. Send one to start!"
        )
    else:
        stats = session.model_stats
        total = sum(stats.values())

        image_info = "\n".join([
            f"• Image {img.id} ({(datetime.now() - img.upload_time).seconds // 60}m ago)"
            for img in session.images
        ])

        await update.message.reply_text(
            f"📊 **Session Status**\n\n"
            f"**Images:** {len(session.images)}/{MAX_IMAGES_PER_USER}\n"
            f"{image_info}\n\n"
            f"**Messages:** {len(session.conversation_history)}\n"
            f"**Last activity:** {(datetime.now() - session.last_activity).seconds // 60}m ago\n\n"
            f"**AI Usage:**\n"
            f"• OpenAI: {stats.get('openai', 0)}\n"
            f"• Groq: {stats.get('groq', 0)}\n"
            f"• Failed: {stats.get('failed', 0)}"
        )


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Run the bot"""
    if not TELEGRAM_TOKEN:
        logger.error("TELEGRAM_TOKEN_VISION not found!")
        return

    logger.info("Starting Marine Edge Vision Bot (Dual System)")
    logger.info(f"Primary: OpenAI {OPENAI_MODEL}")
    logger.info(f"Backup: Groq {GROQ_MODEL}")
    logger.info(f"Rate limit: {MAX_REQUESTS_PER_MINUTE} req/min")

    # Create application
    app = Application.builder().token(TELEGRAM_TOKEN).build()

    # Add handlers
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(CommandHandler("clear", clear_command))
    app.add_handler(CommandHandler("stats", stats_command))
    app.add_handler(MessageHandler(filters.PHOTO, handle_photo))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    # Run
    logger.info("Bot running. Press Ctrl+C to stop.")
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()