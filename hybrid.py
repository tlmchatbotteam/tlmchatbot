# Optimized Rule-Based Chatbot with GPT Keyword-Only Integration
# MAJOR UPDATE: GPT returns ONLY keywords from dataset (no free-form rewriting)
# This ensures 100% accuracy - GPT can only return existing keywords

# --- SỬA LỖI (User 29/10/2025):
# 1. Sửa hàm detect_needs_normalization: len(t) <= 2 (thay vì 3) để tránh false positive
# 2. Sửa hàm normalize_query_with_gpt_keyword_only: Cho phép GPT trả về "KHÔNG_PHÙ_HỢP"

import json
import os
import re
import torch
import unicodedata
import logging
import hashlib
import sys
from difflib import SequenceMatcher
from types import SimpleNamespace
from flask import Flask, request, jsonify, send_from_directory, render_template
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import Optional, List, Tuple, Dict
from functools import lru_cache
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables
load_dotenv()

# --- SỬA LỖI ĐƯỜNG DẪN (DEPLOYMENT) ---
# Lấy đường dẫn tuyệt đối đến thư mục chứa file hybrid.py
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# --- KẾT THÚC SỬA LỖI ĐƯỜNG DẪN ---

# FIX: Force UTF-8 encoding on Windows console
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except AttributeError:
        import io

        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# Setup logging with UTF-8 encoding
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('chatbot.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Initialize OpenAI client with error handling
try:
    client = OpenAI(
        api_key=os.getenv('OPENAI_API_KEY'),
        timeout=int(os.getenv('OPENAI_TIMEOUT', '10')),
        max_retries=int(os.getenv('OPENAI_MAX_RETRIES', '3'))
    )
    logger.info("OpenAI client initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize OpenAI client: {e}")
    client = None

# NEW: fast fuzzy matching with graceful fallback
try:
    from rapidfuzz import fuzz, process

    logger.info("RapidFuzz loaded successfully")
except Exception as e:
    logger.warning(f"RapidFuzz not available, using difflib fallback: {e}")
    fuzz = None
    process = None


# --- CONFIGURATION ---
@dataclass
class ChatbotConfig:
    """Centralized configuration"""
    # --- SỬA LỖI ĐƯỜNG DẪN (DEPLOYMENT) ---
    # Sử dụng đường dẫn tuyệt đối để đảm bảo server luôn tìm thấy file
    DATA_PATH: str = os.getenv('DATA_PATH', os.path.join(BASE_DIR, 'admissions_data.json'))
    IMAGES_DIR: str = os.getenv('IMAGES_DIR', os.path.join(BASE_DIR, 'images'))
    # --- KẾT THÚC SỬA LỖI ĐƯỜNG DẪN ---

    # GPT API settings
    USE_GPT: bool = os.getenv('USE_GPT', 'false').lower() == 'true'
    GPT_MODEL: str = os.getenv('OPENAI_MODEL', 'gpt-4o-mini')

    # NEW: Separate settings for keyword extraction
    GPT_KEYWORD_MAX_TOKENS: int = int(os.getenv('GPT_KEYWORD_MAX_TOKENS', '50'))
    GPT_KEYWORD_TEMPERATURE: float = float(os.getenv('GPT_KEYWORD_TEMPERATURE', '0'))

    # GPT Normalization settings (now returns keyword only)
    USE_GPT_NORMALIZATION: bool = os.getenv('USE_GPT_NORMALIZATION', 'true').lower() == 'true'
    NORMALIZATION_TEMPERATURE: float = 0  # Deterministic for keyword selection
    NORMALIZATION_MAX_TOKENS: int = 50  # Shorter - just need 1 keyword

    # Thresholds
    FUZZY_STRONG: float = 0.88
    FUZZY_MIN: float = 0.60
    EMBED_STRONG: float = 0.72
    EMBED_MIN: float = 0.60

    # Performance
    MAX_CANDIDATES: int = 8
    TOP_K_RETRIEVAL: int = 10
    HISTORY_LIMIT: int = 10
    CACHE_SIZE: int = 1000
    SESSION_EXPIRATION_HOURS: int = 24

    # NEW: Pre-filtering for GPT
    GPT_PREFILTER_CANDIDATES: int = 30  # Send only top 30 candidates to GPT

    # Model names
    SBERT_MODEL: str = "paraphrase-multilingual-mpnet-base-v2"
    CROSS_ENCODER_MODEL: str = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"


config = ChatbotConfig()


# --- TRIE DATA STRUCTURE FOR OPTIMIZED MATCHING ---
class TrieNode:
    """Node của cây Trie"""

    def __init__(self):
        self.children = {}
        self.is_end = False
        self.data = None


class KeywordTrie:
    """Cây Trie để tìm kiếm nhanh keywords"""

    def __init__(self):
        self.root = TrieNode()
        self.max_phrase_length = 0

    def insert(self, phrase: str, data: dict):
        """Thêm keyword vào Trie"""
        try:
            tokens = phrase.split()
            self.max_phrase_length = max(self.max_phrase_length, len(tokens))
            node = self.root
            for token in tokens:
                if token not in node.children:
                    node.children[token] = TrieNode()
                node = node.children[token]
            node.is_end = True
            node.data = data
        except Exception as e:
            logger.error(f"Error inserting phrase '{phrase}' into Trie: {e}")

    def search_all_matches(self, question: str) -> List[Tuple[str, dict, int]]:
        """Tìm tất cả keyword matches trong câu hỏi"""
        try:
            tokens = question.split()
            matches = []
            for start_idx in range(len(tokens)):
                node = self.root
                for end_idx in range(start_idx, min(start_idx + self.max_phrase_length, len(tokens))):
                    token = tokens[end_idx]
                    if token not in node.children:
                        break
                    node = node.children[token]
                    if node.is_end:
                        phrase = ' '.join(tokens[start_idx:end_idx + 1])
                        phrase_length = end_idx - start_idx + 1
                        matches.append((phrase, node.data, phrase_length))
            return matches
        except Exception as e:
            logger.error(f"Error searching matches for '{question}': {e}")
            return []

    def find_best_match(self, question: str) -> Optional[dict]:
        """Tìm match dài nhất"""
        try:
            matches = self.search_all_matches(question)
            if not matches:
                return None
            matches.sort(key=lambda x: (-x[2], -len(x[0])))
            return matches[0][1]
        except Exception as e:
            logger.error(f"Error finding best match: {e}")
            return None


# --- HYBRID MATCHER ---
class HybridMatcher:
    """Kết hợp exact, Trie, và fuzzy matching"""

    def __init__(self, keywords_map: dict):
        self.exact_map = keywords_map
        self.trie = KeywordTrie()
        self.all_tokens = set()

        logger.info(f"Building Trie index for {len(keywords_map)} keywords...")
        for kw, data in keywords_map.items():
            self.trie.insert(kw, data)
            self.all_tokens.update(kw.split())
        logger.info("Trie index built successfully")

    def find_match(self, question: str) -> Optional[dict]:
        """Tìm match theo thứ tự: exact -> trie -> token check"""
        try:
            if question in self.exact_map:
                return self.exact_map[question]
            trie_result = self.trie.find_best_match(question)
            if trie_result:
                return trie_result
            tokens = set(question.split())
            if not tokens.intersection(self.all_tokens):
                return None
            return None
        except Exception as e:
            logger.error(f"Error in hybrid matching: {e}")
            return None


# --- RESPONSE CACHE ---
class ResponseCache:
    """LRU cache cho responses"""

    def __init__(self, max_size: int = 1000):
        self.cache = {}
        self.max_size = max_size
        self.access_order = []
        self.hits = 0
        self.misses = 0

    def _make_key(self, text: str) -> str:
        return hashlib.md5(text.encode()).hexdigest()

    def get(self, text: str) -> Optional[dict]:
        try:
            key = self._make_key(text)
            if key in self.cache:
                self.access_order.remove(key)
                self.access_order.append(key)
                self.hits += 1
                return self.cache[key]
            self.misses += 1
            return None
        except Exception as e:
            logger.error(f"Cache get error: {e}")
            return None

    def set(self, text: str, response: dict):
        try:
            key = self._make_key(text)
            if len(self.cache) >= self.max_size:
                oldest = self.access_order.pop(0)
                del self.cache[oldest]
            self.cache[key] = response
            self.access_order.append(key)
        except Exception as e:
            logger.error(f"Cache set error: {e}")

    def clear(self):
        self.cache.clear()
        self.access_order.clear()
        self.hits = 0
        self.misses = 0


response_cache = ResponseCache(max_size=config.CACHE_SIZE)


# --- SESSION STORE ---
class SessionStore:
    """Session management với expiration"""

    def __init__(self, expiration_hours: int = 24):
        self.sessions = {}
        self.expiration_delta = timedelta(hours=expiration_hours)

    def get_history(self, session_id: str) -> list:
        try:
            self._cleanup_expired()
            session = self.sessions.get(session_id)
            if session:
                session['last_access'] = datetime.now()
                return session['history']
            return []
        except Exception as e:
            logger.error(f"Error getting history: {e}")
            return []

    def append_message(self, session_id: str, role: str, text: str):
        try:
            self._cleanup_expired()
            if session_id not in self.sessions:
                self.sessions[session_id] = {
                    'history': [],
                    'created': datetime.now(),
                    'last_access': datetime.now()
                }
            session = self.sessions[session_id]
            session['history'].append({
                'role': role,
                'text': text[:2000],
                'timestamp': datetime.now().isoformat()
            })
            session['last_access'] = datetime.now()
            if len(session['history']) > config.HISTORY_LIMIT:
                session['history'] = session['history'][-config.HISTORY_LIMIT:]
        except Exception as e:
            logger.error(f"Error appending message: {e}")

    def _cleanup_expired(self):
        try:
            now = datetime.now()
            expired = [
                sid for sid, data in self.sessions.items()
                if now - data['last_access'] > self.expiration_delta
            ]
            for sid in expired:
                del self.sessions[sid]
            if expired:
                logger.info(f"Cleaned up {len(expired)} expired sessions")
        except Exception as e:
            logger.error(f"Error cleaning up sessions: {e}")


session_store = SessionStore(expiration_hours=config.SESSION_EXPIRATION_HOURS)

# --- FLASK APP ---
app = Flask(__name__)
if not hasattr(app, 'session_state'):
    app.session_state = SimpleNamespace(
        question_embeddings=None,
        question_texts=[],
        question_data_map={},
        faiss_index=None,
        knn_index=None
    )


# --- LOAD DATA ---
def load_admissions_data(file_path: str) -> dict:
    """Load data với error handling"""
    try:
        if not os.path.exists(file_path):
            logger.error(f"Data file not found: {file_path}")
            return {"questions": []}
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if 'questions' not in data:
            logger.error("Invalid data format: 'questions' key missing")
            return {"questions": []}
        logger.info(f"Loaded {len(data['questions'])} questions from {file_path}")
        return data
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON format: {e}")
        return {"questions": []}
    except Exception as e:
        logger.error(f"Unexpected error loading data: {e}")
        return {"questions": []}


try:
    admissions_data = load_admissions_data(config.DATA_PATH)
except Exception as e:
    logger.critical(f"Failed to load admissions data: {e}")
    admissions_data = {"questions": []}


# --- UTILITY FUNCTIONS ---
def remove_vietnamese_accents(text):
    """Loại bỏ dấu tiếng Việt"""
    try:
        return "".join(c for c in unicodedata.normalize('NFD', text)
                       if unicodedata.category(c) != 'Mn')
    except Exception as e:
        logger.error(f"Error removing accents: {e}")
        return text


def normalize_text(text):
    """Chuẩn hóa text"""
    try:
        text = text.lower().strip()
        text = re.sub(r'\s+', ' ', text)
        return text
    except Exception as e:
        logger.error(f"Error normalizing text: {e}")
        return text


@lru_cache(maxsize=10000)
def normalize_and_unaccent(text: str) -> str:
    """Cached version - chuẩn hóa và bỏ dấu"""
    try:
        norm = remove_vietnamese_accents(normalize_text(text))
        norm = re.sub(r'\bly\b', 'li', norm)
        return norm
    except Exception as e:
        logger.error(f"Error in normalize_and_unaccent: {e}")
        return text


# --- NEW: IMPROVED GPT KEYWORD-ONLY NORMALIZER ---
@lru_cache(maxsize=1000)
def detect_needs_normalization(text: str) -> bool:
    """
    Phát hiện câu hỏi cần chuẩn hóa bởi GPT

    Returns:
        True nếu cần GPT xử lý (không dấu, viết tắt, typo)
    """
    try:
        norm = normalize_text(text)

        # 1. Check không dấu
        has_vietnamese_chars = bool(
            re.search(r'[àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ]', norm))
        if not has_vietnamese_chars and len(norm) > 5:
            logger.info(f"[Detector] No Vietnamese accents detected: {text[:50]}")
            return True

        # 2. Check viết tắt (nhiều từ ngắn liên tiếp)
        tokens = norm.split()
        # SỬA LỖI: Chỉ coi từ <= 2 ký tự là viết tắt (tránh "nhà", "học"...)
        short_tokens = [t for t in tokens if len(t) <= 2 and t.isalpha()]
        if len(short_tokens) >= 2:
            logger.info(f"[Detector] Abbreviations detected: {text[:50]}")
            return True

        # 3. Check typo patterns
        invalid_patterns = [
            r'[bcdfghjklmnpqrstvwxyz]{4,}',  # 4+ phụ âm liên tiếp
            r'\b[a-z]{1,2}\b.*\b[a-z]{1,2}\b',  # nhiều từ 1-2 ký tự
        ]
        for pattern in invalid_patterns:
            if re.search(pattern, norm):
                logger.info(f"[Detector] Invalid pattern detected: {text[:50]}")
                return True

        # 4. Check từ không tồn tại trong keywords (will be initialized later)
        if 'ALL_KEYWORDS_SET' in globals():
            unaccented = normalize_and_unaccent(text)
            found_tokens = set(unaccented.split()) & ALL_KEYWORDS_SET
            if len(tokens) >= 3 and len(found_tokens) == 0:
                logger.info(f"[Detector] No keyword match: {text[:50]}")
                return True

        return False

    except Exception as e:
        logger.error(f"[Detector] Error: {e}")
        return False


def pre_filter_keywords(question: str, all_keywords: list, top_k: int = 50) -> list:
    """
    Lọc trước keywords bằng fuzzy matching để giảm số lượng gửi GPT

    Args:
        question: Câu hỏi người dùng
        all_keywords: Tất cả keywords từ dataset
        top_k: Số lượng candidates tối đa

    Returns:
        List keywords phù hợp nhất
    """
    try:
        norm_q = normalize_and_unaccent(question)

        if fuzz and process:
            # Use RapidFuzz for fast matching
            matches = process.extract(
                norm_q,
                all_keywords,
                scorer=fuzz.token_set_ratio,
                limit=top_k
            )
            candidates = [m[0] for m in matches if m[1] > 30]
            logger.info(f"[Pre-filter] Found {len(candidates)} candidates using RapidFuzz")
            return candidates
        else:
            # Fallback: token overlap
            q_tokens = set(norm_q.split())
            candidates = []
            for kw in all_keywords:
                kw_tokens = set(kw.split())
                if q_tokens & kw_tokens:
                    candidates.append(kw)
            result = candidates[:top_k]
            logger.info(f"[Pre-filter] Found {len(result)} candidates using token overlap")
            return result

    except Exception as e:
        logger.error(f"Error pre-filtering keywords: {e}")
        return all_keywords[:top_k]


@lru_cache(maxsize=500)
def normalize_query_with_gpt_keyword_only(query: str, keywords_hash: str) -> Optional[str]:
    """
    🆕 GPT CHỈ TRẢ VỀ KEYWORD TỪ DATASET - KHÔNG TỰ DO VIẾT LẠI CÂU

    Đảm bảo 100% keyword thuộc dataset, tránh GPT sáng tạo từ không có sẵn.

    Args:
        query: Câu hỏi người dùng (có thể không dấu/viết tắt/lỗi)
        keywords_hash: Hash của danh sách keywords (để cache)

    Returns:
        Keyword từ dataset hoặc None

    Examples:
        >>> normalize_query_with_gpt_keyword_only("hoc phi truong", "...")
        "học phí"

        >>> normalize_query_with_gpt_keyword_only("ts lop 10", "...")
        "tuyển sinh lớp 10"
    """
    try:
        if not config.USE_GPT_NORMALIZATION or client is None:
            logger.debug("[GPT Keyword] Disabled or client not available")
            return None

        # Lấy tất cả keywords từ dataset
        all_keywords = list(KEYWORD_TO_ITEM_MAP.keys())

        # Pre-filter để giảm tokens (quan trọng để tiết kiệm cost!)
        candidates = pre_filter_keywords(
            query,
            all_keywords,
            top_k=config.GPT_PREFILTER_CANDIDATES
        )

        if not candidates:
            logger.warning(f"[GPT Keyword] No candidates found for: {query}")
            return None

        # SỬA LỖI: PROMPT MỚI: Cho phép GPT từ chối
        prompt = f"""Bạn là hệ thống matching câu hỏi với dataset keywords.

DANH SÁCH KEYWORDS CÓ SẴN (chọn ĐÚNG 1 phù hợp nhất):
{chr(10).join([f"- {kw}" for kw in candidates])}

CÂU HỎI NGƯỜI DÙNG (có thể không dấu/viết tắt/lỗi chính tả):
"{query}"

YÊU CẦU:
1. Phân tích ý định của câu hỏi
2. Chọn ĐÚNG 1 keyword phù hợp nhất từ danh sách trên
3. NẾU không có keyword nào phù hợp (ví dụ: câu hỏi về "nhà vệ sinh" nhưng danh sách chỉ có "học phí", "điểm chuẩn"), hãy trả về "KHÔNG_PHÙ_HỢP"
4. CHỈ trả về keyword đó hoặc "KHÔNG_PHÙ_HỢP", KHÔNG giải thích, KHÔNG thêm bớt gì

VÍ DỤ CHUẨN:
- Input: "hoc phi truong" → Output: học phí
- Input: "ts lop 10" → Output: tuyển sinh lớp 10  
- Input: "diem chuan ntn" → Output: điểm chuẩn
- Input: "hp vb2" → Output: học phí văn bằng 2
- Input: "nhà vệ sinh" (và "nhà vệ sinh" không có trong danh sách) → Output: KHÔNG_PHÙ_HỢP

Keyword phù hợp:"""

        response = client.chat.completions.create(
            model=config.GPT_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "Bạn là chuyên gia matching keyword. LUÔN LUÔN chỉ trả về ĐÚNG 1 keyword có trong danh sách đã cho, hoặc trả về 'KHÔNG_PHÙ_HỢP'."
                },
                {"role": "user", "content": prompt}
            ],
            temperature=config.NORMALIZATION_TEMPERATURE,  # 0 = deterministic
            max_tokens=config.NORMALIZATION_MAX_TOKENS  # 50 = đủ cho 1 keyword
        )

        keyword = response.choices[0].message.content.strip().lower()

        # Làm sạch output (GPT đôi khi thêm quotes/prefix)
        keyword = re.sub(r'^["\'\-→•]|["\']$', '', keyword)
        keyword = keyword.replace('output:', '').replace('→', '').strip()

        # SỬA LỖI: Check nếu GPT trả về "không phù hợp"
        if "không_phù_hợp" in keyword or "khong_phu_hop" in keyword:
            logger.info(f"[GPT Keyword] ✗ GPT judged as not suitable: '{query}'")
            return None

        # 🔒 VALIDATE: Keyword PHẢI có trong dataset
        if keyword in all_keywords:
            logger.info(f"[GPT Keyword] ✓ '{query}' → '{keyword}'")
            return keyword

        # Thử normalize và check lại
        keyword_normalized = normalize_and_unaccent(keyword)
        if keyword_normalized in all_keywords:
            logger.info(f"[GPT Keyword] ✓ '{query}' → '{keyword_normalized}' (normalized)")
            return keyword_normalized

        # Thử fuzzy match với candidates (GPT có thể sai chính tả nhẹ)
        if fuzz and process:
            match = process.extractOne(
                keyword,
                all_keywords,
                scorer=fuzz.ratio,
                score_cutoff=85
            )
            if match:
                matched_keyword = match[0]
                logger.warning(f"[GPT Keyword] ⚠ '{keyword}' fuzzy matched to '{matched_keyword}'")
                return matched_keyword

        # ❌ REJECT: Keyword không hợp lệ
        logger.warning(f"[GPT Keyword] ✗ Invalid: '{keyword}' not in dataset (query: '{query}')")
        return None

    except Exception as e:
        logger.error(f"[GPT Keyword] Error: {e}")
        return None


def normalize_query_with_gpt_keyword_wrapper(query: str) -> Optional[str]:
    """
    Wrapper để làm keywords_hash có thể cache được
    """
    all_keywords = list(KEYWORD_TO_ITEM_MAP.keys())
    keywords_hash = hashlib.md5(','.join(sorted(all_keywords)).encode()).hexdigest()
    return normalize_query_with_gpt_keyword_only(query, keywords_hash)


def strip_leadin_phrases(text: str) -> str:
    """Loại bỏ cụm dẫn nhập"""
    try:
        norm = normalize_and_unaccent(text)
        leadins = [
            r'^toi\s*muon\s*biet\s*ve\s*',
            r'^toi\s*muon\s*biet\s*',
            r'^toi\s*muon\s*hoi\s*',
            r'^cho\s*t\s*oi\s*biet\s*ve\s*',
            r'^cho\s*toi\s*biet\s*ve\s*',
            r'^thong\s*tin\s*ve\s*',
            r'^gioi\s*thieu\s*ve\s*',
            r'^ve\s*',
            r'^truong\s*co\s*',
            r'^truong\s*co\s*cac\s*',
            r'^truong\s*co\s*nhung\s*'
        ]
        for pat in leadins:
            norm = re.sub(pat, '', norm)
        norm = re.sub(r'\b(cua|ve)\s*$', '', norm).strip()
        norm = re.sub(r'\b(bao\s*nhieu|nao|khong|khong\s*\?|gi|gi\s*\?)\s*$', '', norm).strip()
        norm = re.sub(r'\bcua\s*(truong|thpt|trg|truong\s*thpt|trung\s*hoc\s*pho\s*thong)\b.*$', '', norm).strip()
        return norm
    except Exception as e:
        logger.error(f"Error stripping phrases: {e}")
        return text


# --- SESSION HELPERS ---
def get_session_history(session_id: str):
    return session_store.get_history(session_id)


def append_history(session_id: str, role: str, text: str):
    session_store.append_message(session_id, role, text)


def last_user_turn(session_id: str) -> str:
    try:
        hist = get_session_history(session_id)
        for m in reversed(hist):
            if m.get('role') == 'user' and m.get('text'):
                return m.get('text')
        return ""
    except Exception as e:
        logger.error(f"Error getting last user turn: {e}")
        return ""


def looks_context_dependent(q: str) -> bool:
    try:
        nq = normalize_and_unaccent(q)
        if len(nq) <= 25:
            return True
        patterns = [
            r'^(con|the\s*con|va\s*con)\b',
            r'^(vay|the\s*nao|the\s*thi|con\s*gi)\b',
            r'\b(o\s*dau|khi\s*nao|bao\s*gio|bao\s*nhieu|ten\s*gi)\b',
            r'^(cai\s*do|nguoi\s*do|no|the)\b',
        ]
        for p in patterns:
            if re.search(p, nq):
                return True
        return False
    except Exception as e:
        logger.error(f"Error checking context dependency: {e}")
        return False


def augment_with_context(session_id: str, q: str) -> str:
    """Ghép với ngữ cảnh nếu cần"""
    try:
        norm_q_unaccented = normalize_and_unaccent(q)
        if 'KEYWORD_TO_ITEM_MAP' in globals() and norm_q_unaccented in KEYWORD_TO_ITEM_MAP:
            return q
        if re.fullmatch(r"hieu\s*truong", norm_q_unaccented):
            return q
        prev = last_user_turn(session_id)
        if not prev:
            return q
        interrogatives = [
            r"\bai\b", r"\bgi\b", r"\bgi\s*\?", r"o\s*dau", r"khi\s*nao",
            r"bao\s*gio", r"bao\s*nhieu", r"the\s*nao", r"nao", r"khong", r"sao"
        ]
        for pat in interrogatives:
            if re.search(pat, norm_q_unaccented):
                return f"{prev} ; {q}"
        return q
    except Exception as e:
        logger.error(f"Error augmenting context: {e}")
        return q


# --- BUILD KEYWORD MAPS ---
def get_all_question_keywords():
    """Trích xuất keywords"""
    try:
        keywords = set()
        for item in admissions_data.get('questions', []):
            questions = item.get('question', [])
            if isinstance(questions, str):
                questions = [questions]
            for q in questions:
                norm_q = normalize_text(q)
                unaccented_q = remove_vietnamese_accents(norm_q)
                if len(norm_q) > 2:
                    keywords.add(norm_q)
                if len(unaccented_q) > 2:
                    keywords.add(unaccented_q)
        return sorted(keywords, key=lambda x: -len(x))
    except Exception as e:
        logger.error(f"Error getting keywords: {e}")
        return []


QUESTION_KEYWORDS = get_all_question_keywords()

# Build keyword maps
KEYWORD_ANSWER_MAP = {}
ALL_KEYWORDS_SET = set()
KEYWORD_TO_ITEM_MAP = {}
KEYWORD_TO_ITEMS_MAP = {}

try:
    for item in admissions_data.get('questions', []):
        questions = item.get('question', [])
        if isinstance(questions, str):
            questions = [questions]
        for q in questions:
            key = normalize_and_unaccent(q)
            KEYWORD_ANSWER_MAP[key] = item
            KEYWORD_TO_ITEM_MAP[key] = item
            ALL_KEYWORDS_SET.add(key)
            lst = KEYWORD_TO_ITEMS_MAP.setdefault(key, [])
            if item not in lst:
                lst.append(item)
    logger.info(f"Built keyword maps with {len(KEYWORD_TO_ITEM_MAP)} entries")
except Exception as e:
    logger.error(f"Error building keyword maps: {e}")

# Cached constants
try:
    MAX_KEY_LEN = max((len(k.split()) for k in KEYWORD_TO_ITEM_MAP.keys()), default=6)
except Exception:
    MAX_KEY_LEN = 6

COMMON_STOP_TOKENS = {"truong", "co", "cua", "ve", "la", "nao", "gi", "cai", "cac", "nhung", "o", "dau", "ai"}

KEY_HEADS_2 = set()
try:
    for _k in KEYWORD_TO_ITEM_MAP.keys():
        _ks = _k.split()
        if len(_ks) >= 2:
            KEY_HEADS_2.add(' '.join(_ks[:2]))
except Exception as e:
    logger.error(f"Error building KEY_HEADS_2: {e}")

# --- INITIALIZE HYBRID MATCHER ---
HYBRID_MATCHER = None
try:
    if KEYWORD_TO_ITEM_MAP:
        HYBRID_MATCHER = HybridMatcher(KEYWORD_TO_ITEM_MAP)
        logger.info("Hybrid matcher initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize hybrid matcher: {e}")


# --- MODEL MANAGEMENT ---
class ModelManager:
    """Singleton pattern cho models"""
    _instance = None
    _sbert_model = None
    _cross_encoder_model = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @property
    def sbert_model(self):
        if self._sbert_model is None:
            try:
                from sentence_transformers import SentenceTransformer
                logger.info("Loading SBERT model...")
                self._sbert_model = SentenceTransformer(config.SBERT_MODEL)
                logger.info("SBERT model loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load SBERT model: {e}")
                self._sbert_model = None
        return self._sbert_model

    @property
    def cross_encoder_model(self):
        if self._cross_encoder_model is None:
            try:
                from sentence_transformers import CrossEncoder
                logger.info("Loading Cross-Encoder model...")
                self._cross_encoder_model = CrossEncoder(config.CROSS_ENCODER_MODEL)
                logger.info("Cross-Encoder model loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load Cross-Encoder model: {e}")
                self._cross_encoder_model = None
        return self._cross_encoder_model


model_manager = ModelManager()

# --- DEVICE ---
device = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Using device: {device}")


# --- ENCODING FUNCTIONS ---
def encode_question_embedding(inputs):
    """Mã hóa câu hỏi thành embedding"""
    try:
        sbert_model = model_manager.sbert_model
        if sbert_model is not None:
            if isinstance(inputs, list):
                embs = sbert_model.encode(inputs, batch_size=32, normalize_embeddings=True, convert_to_tensor=True)
                return embs
            else:
                embs = sbert_model.encode([inputs], batch_size=32, normalize_embeddings=True, convert_to_tensor=True)
                return embs
        return None
    except Exception as e:
        logger.error(f"Error encoding embedding: {e}")
        return None


def build_question_embeddings_and_maps(admissions_data_local):
    """Xây dựng embeddings cho toàn bộ câu hỏi"""
    try:
        question_texts = []
        question_items = []
        for item in admissions_data_local.get('questions', []):
            questions = item.get('question', [])
            if isinstance(questions, str):
                questions = [questions]
            for q in questions:
                qn_embed = normalize_text(q)
                if len(qn_embed) < 2:
                    continue
                question_texts.append(qn_embed)
                question_items.append(item)

        if not question_texts:
            return [], None, {}

        embeddings = encode_question_embedding(question_texts)

        text_to_item = {}
        for t, it in zip(question_texts, question_items):
            if t not in text_to_item:
                text_to_item[t] = it

        logger.info(f"Built embeddings for {len(question_texts)} questions")
        return question_texts, embeddings, text_to_item
    except Exception as e:
        logger.error(f"Error building embeddings: {e}")
        return [], None, {}


# --- INITIALIZE SEMANTIC RESOURCES ---
def initialize_semantic_resources():
    """Build embeddings và lookup maps"""
    try:
        logger.info("Initializing semantic resources...")
        question_texts, embeddings, text_to_item = build_question_embeddings_and_maps(admissions_data)

        app.session_state.question_texts = question_texts or []
        app.session_state.question_embeddings = embeddings
        app.session_state.question_data_map = text_to_item or {}

        # Build KNN index
        try:
            if embeddings is not None and embeddings.shape[0] > 0:
                from sklearn.neighbors import NearestNeighbors
                nn = NearestNeighbors(metric='cosine', algorithm='auto')
                emb_np = embeddings.detach().cpu().numpy()
                nn.fit(emb_np)
                app.session_state.knn_index = nn
                logger.info("KNN index built successfully")
        except Exception as e:
            logger.warning(f"KNN index initialization failed: {e}")

        # Build FAISS index
        try:
            if embeddings is not None and embeddings.shape[0] > 0:
                import faiss
                emb_np = embeddings.detach().cpu().numpy().astype('float32')
                try:
                    faiss.normalize_L2(emb_np)
                except Exception:
                    pass
                d = emb_np.shape[1]
                index = faiss.IndexFlatIP(d)
                index.add(emb_np)
                app.session_state.faiss_index = index
                logger.info("FAISS index built successfully")
        except Exception as e:
            logger.warning(f"FAISS index initialization failed: {e}")

        logger.info("Semantic resources initialized")
    except Exception as e:
        logger.error(f"Error initializing semantic resources: {e}")


# --- MATCHING FUNCTIONS ---
def fuzzy_best_item(question_text: str):
    """Fuzzy matching với rapidfuzz hoặc difflib"""
    try:
        user_q = normalize_and_unaccent(question_text)
        keys = list(KEYWORD_TO_ITEM_MAP.keys())
        if not keys:
            return None, 0.0

        if process is not None and fuzz is not None:
            match = process.extractOne(user_q, keys, scorer=fuzz.token_set_ratio)
            if not match:
                return None, 0.0
            best_key, score, _ = match
            return KEYWORD_TO_ITEM_MAP.get(best_key), float(score) / 100.0

        # Fallback: difflib
        best_item = None
        best_ratio = 0.0
        for k in keys:
            ratio = SequenceMatcher(None, user_q, k).ratio()
            if ratio > best_ratio:
                best_ratio = ratio
                best_item = KEYWORD_TO_ITEM_MAP.get(k)
        return best_item, float(best_ratio)
    except Exception as e:
        logger.error(f"Error in fuzzy matching: {e}")
        return None, 0.0


def retrieve_topk_embeddings(query_text: str, top_k: int = 10):
    """Retrieve top-k candidates bằng embeddings"""
    try:
        sbert_model = model_manager.sbert_model
        if sbert_model is None:
            return []

        if (not hasattr(app.session_state, 'question_embeddings') or
                app.session_state.question_embeddings is None or
                not hasattr(app.session_state, 'question_texts') or
                len(app.session_state.question_texts) == 0):
            return []

        # Try FAISS first
        index = getattr(app.session_state, 'faiss_index', None)
        if index is not None:
            try:
                q_emb = encode_question_embedding(normalize_text(query_text))
                q_norm = torch.nn.functional.normalize(q_emb, dim=1)
                q_np = q_norm.detach().cpu().numpy().astype('float32')
                k = min(top_k, app.session_state.question_embeddings.shape[0])
                D, I = index.search(q_np, k)
                sims = D[0].tolist()
                inds = I[0].tolist()
                return [(int(i), float(s)) for i, s in zip(inds, sims) if i >= 0]
            except Exception as e:
                logger.warning(f"FAISS search failed: {e}")

        # Try KNN
        knn = getattr(app.session_state, 'knn_index', None)
        if knn is not None:
            try:
                q_emb = encode_question_embedding(normalize_text(query_text))
                if q_emb is None:
                    return []
                q_np = torch.nn.functional.normalize(q_emb, dim=1).detach().cpu().numpy()
                k = min(top_k, app.session_state.question_embeddings.shape[0])
                distances, indices = knn.kneighbors(q_np, n_neighbors=k, return_distance=True)
                inds = indices[0].tolist()
                dists = distances[0].tolist()
                sims = [1.0 - float(d) for d in dists]
                pairs = sorted([(int(i), float(s)) for i, s in zip(inds, sims)],
                               key=lambda x: x[1], reverse=True)
                return pairs
            except Exception as e:
                logger.warning(f"KNN search failed: {e}")

        # Fallback: torch cosine
        try:
            q_emb = encode_question_embedding(normalize_text(query_text))
            q = torch.nn.functional.normalize(q_emb, dim=1)
            c = torch.nn.functional.normalize(app.session_state.question_embeddings, dim=1)
            sims = torch.mm(q, c.t()).squeeze(0)
            values, indices = torch.topk(sims, k=min(top_k, sims.shape[0]))
            return [(int(idx.item()), float(val.item())) for val, idx in zip(values, indices)]
        except Exception as e:
            logger.error(f"Torch cosine search failed: {e}")
            return []
    except Exception as e:
        logger.error(f"Error retrieving embeddings: {e}")
        return []


def rerank_with_cross_encoder(query_text: str, candidate_indices):
    """Rerank với cross-encoder"""
    try:
        cross_encoder = model_manager.cross_encoder_model
        if cross_encoder is None or not candidate_indices:
            return None, 0.0

        max_candidates = config.MAX_CANDIDATES
        candidate_indices = list(candidate_indices)[:max_candidates]
        pairs = []
        qt = normalize_text(query_text)
        for idx in candidate_indices:
            cand_text = app.session_state.question_texts[idx]
            pairs.append((qt, normalize_text(cand_text)))
        scores = cross_encoder.predict(pairs)
        try:
            scores = scores.tolist()
        except Exception:
            pass
        if not scores:
            return None, 0.0
        best_pos = max(range(len(scores)), key=lambda i: scores[i])
        return candidate_indices[best_pos], float(scores[best_pos])
    except Exception as e:
        logger.error(f"Error in cross-encoder reranking: {e}")
        return None, 0.0


def contains_dataset_keyword(text: str) -> bool:
    """Kiểm tra xem text có chứa keyword từ dataset không"""
    try:
        if not text or not KEYWORD_TO_ITEM_MAP:
            return False
        nq = normalize_and_unaccent(text)
        if nq in KEYWORD_TO_ITEM_MAP:
            return True

        def _wb_contains(hay: str, needle: str) -> bool:
            if not hay or not needle:
                return False
            H = f" {hay.strip()} ".replace("  ", " ")
            N = f" {needle.strip()} ".replace("  ", " ")
            return N in H

        np = re.sub(r"\W+", " ", nq).strip()
        for key in KEYWORD_TO_ITEM_MAP.keys():
            if len(key) < 3:
                continue
            if _wb_contains(nq, key) or _wb_contains(key, nq):
                if len(nq.split()) >= len(key.split()):
                    return True
            kp = re.sub(r"\W+", " ", key).strip()
            if (kp and np) and (_wb_contains(np, kp) or _wb_contains(kp, np)):
                return True
        return False
    except Exception as e:
        logger.error(f"Error checking dataset keyword: {e}")
        return False


def find_multi_keyword_spans(norm_text: str):
    """Tìm các keyword spans trong text - SỬ DỤNG HYBRID MATCHER"""
    try:
        if HYBRID_MATCHER:
            matches = HYBRID_MATCHER.trie.search_all_matches(norm_text)
            return [phrase for phrase, _, _ in matches]
        return []
    except Exception as e:
        logger.error(f"Error finding keyword spans: {e}")
        return []


def make_clarifying_question(user_query: str, candidates: list) -> dict:
    """Sinh câu hỏi gợi ý khi có nhiều ứng viên"""
    try:
        options = []
        data_questions = admissions_data.get('questions', [])

        for c in candidates:
            q = c.get("question", [])
            if isinstance(q, str):
                q = [q]
            if not q:
                continue
            label = q[0]
            label = re.sub(r"^làm thế nào để\s*", "", label, flags=re.IGNORECASE)
            label = re.sub(r"^cách\s*", "", label, flags=re.IGNORECASE)
            label = re.sub(r"\?$", "", label).strip()
            if len(label) > 40:
                label = " ".join(label.split()[:6])
            if label not in options:
                options.append(label)

        if not options:
            return {"text": "Xin lỗi, tôi chưa tìm thấy thông tin phù hợp.", "media_type": "text",
                    "media_content": None}

        opts_text = " , ".join(options[:5])
        question = f"Bạn muốn hỏi cụ thể về: {opts_text}?"

        opts_objs = []
        for lab, cand in zip(options[:5], candidates[:5]):
            idx = None
            try:
                cand_q = cand.get('question', [])
                if isinstance(cand_q, list) and cand_q:
                    cand_key = normalize_and_unaccent(cand_q[0])
                elif isinstance(cand_q, str):
                    cand_key = normalize_and_unaccent(cand_q)
                else:
                    cand_key = None

                if cand_key:
                    for i, it in enumerate(data_questions):
                        its_q = it.get('question', [])
                        if isinstance(its_q, list) and its_q:
                            its_key = normalize_and_unaccent(its_q[0])
                        elif isinstance(its_q, str):
                            its_key = normalize_and_unaccent(its_q)
                        else:
                            its_key = None
                        if its_key and its_key == cand_key:
                            idx = i
                            break
            except Exception:
                idx = None

            if idx is None:
                opts_objs.append({"id": lab, "label": lab})
            else:
                opts_objs.append({"id": idx, "label": lab})

        return {
            "text": question,
            "media_type": "text",
            "media_content": None,
            "action": "clarification",
            "options": opts_objs
        }
    except Exception as e:
        logger.error(f"Error making clarifying question: {e}")
        return {"text": "Xin lỗi, tôi chưa tìm thấy thông tin phù hợp.", "media_type": "text", "media_content": None}


# --- 🆕 NEW: GPT KEYWORD-ONLY PIPELINE ---
def get_answer_with_gpt_normalization(question: str, session_id: Optional[str] = None):
    """
    🆕 Pipeline mới: GPT chỉ trả về keyword từ dataset

    Flow:
    1. Detect nếu câu hỏi cần normalize (không dấu/viết tắt/typo)
    2. GPT chọn keyword từ dataset → 100% đảm bảo có trong data
    3. Lấy đáp án trực tiếp từ keyword
    4. Fallback về pipeline cũ nếu GPT fail

    Args:
        question: Câu hỏi gốc
        session_id: Session ID (optional)

    Returns:
        List of response dictionaries
    """
    try:
        # Bước 1: Detect cần normalize?
        needs_norm = detect_needs_normalization(question)

        if needs_norm and config.USE_GPT_NORMALIZATION:
            logger.info(f"[Pipeline] Query needs normalization: {question[:50]}")

            # Bước 2: GPT chọn keyword từ dataset
            matched_keyword = normalize_query_with_gpt_keyword_wrapper(question)

            if matched_keyword:
                logger.info(f"[Pipeline] ✓ GPT matched keyword: '{matched_keyword}'")

                # Bước 3: Lấy đáp án trực tiếp từ keyword
                if matched_keyword in KEYWORD_TO_ITEM_MAP:
                    item = KEYWORD_TO_ITEM_MAP[matched_keyword]

                    ans = item.get('answer', "Xin lỗi, tôi chưa tìm thấy thông tin phù hợp.")
                    media_type = "text"
                    media_content = None
                    images = item.get('images')
                    captions = item.get('captions')
                    video_url = item.get('video_url')

                    if images and isinstance(images, str):
                        images = [images]
                    if video_url:
                        media_type = "video"
                        media_content = video_url
                    elif images:
                        media_type = "image"
                        media_content = (images, captions)

                    logger.info(f"[Pipeline] ✓ Returning answer from GPT-matched keyword")
                    return [{"text": ans, "media_type": media_type, "media_content": media_content}]
                else:
                    logger.warning(f"[Pipeline] Keyword '{matched_keyword}' not found in map")
            else:
                logger.info(f"[Pipeline] GPT didn't return valid keyword — responding with no info.")
                return [{"text": "Xin lỗi, tôi không có thông tin về câu hỏi của bạn.", "media_type": "text",
                         "media_content": None}]


    except Exception as e:
        logger.error(f"[Pipeline] Error: {e}")
        return get_answer(question, skip_gpt=False)


# --- MAIN ANSWER FUNCTIONS ---
def get_answer(question, skip_gpt: bool = False):
    """Xử lý câu hỏi với pipeline hybrid"""
    try:
        def _build_response_from_item(item):
            if not item:
                return {"text": "Xin lỗi, tôi chưa tìm thấy thông tin phù hợp.", "media_type": "text",
                        "media_content": None}

            ans = item.get('answer', "Xin lỗi, tôi chưa tìm thấy thông tin phù hợp.")
            media_type = "text"
            media_content = None
            images = item.get('images')
            captions = item.get('captions')
            video_url = item.get('video_url')

            if images and isinstance(images, str):
                images = [images]
            if video_url:
                media_type = "video"
                media_content = video_url
            elif images:
                media_type = "image"
                media_content = (images, captions)

            return {"text": ans, "media_type": media_type, "media_content": media_content}

        # Step 1: Normalization
        norm_question = normalize_text(question)
        norm_unaccent_question = normalize_and_unaccent(question)

        # Handle special case: hiệu trưởng
        if re.fullmatch(r"\s*hiệu\s*trưởng\s*[?.!]*\s*", norm_question) or \
                re.fullmatch(r"\s*hieu\s*truong\s*[?.!]*\s*", norm_unaccent_question):
            hardcoded_response = "Bạn muốn biết về hiệu trưởng hiện tại hay hiệu trưởng qua từng thời kỳ?"
            return [{"text": hardcoded_response, "media_type": "text", "media_content": None,
                     "action": "hieutruong_choices"}]

        core_question = strip_leadin_phrases(norm_unaccent_question)

        # Step 2: Exact multi-answer path
        if core_question in KEYWORD_TO_ITEMS_MAP:
            items = KEYWORD_TO_ITEMS_MAP[core_question]
            if len(items) > 0:
                results = [_build_response_from_item(item) for item in items]
                unique_results = []
                seen_keys = set()
                for r in results:
                    key = r.get("text", "")
                    if key not in seen_keys:
                        unique_results.append(r)
                        seen_keys.add(key)
                if unique_results:
                    return unique_results

        # Step 3: Multi-intent span-based path - SỬ DỤNG TRIE
        sub_questions = find_multi_keyword_spans(core_question)
        if len(sub_questions) > 1:
            results = []
            for subq in sub_questions:
                items = KEYWORD_TO_ITEMS_MAP.get(subq, [])
                for item in items:
                    if item:
                        results.append(_build_response_from_item(item))

            if results:
                unique_results = []
                seen_keys = set()
                for r in results:
                    key = r.get("text", "")
                    if key not in seen_keys:
                        unique_results.append(r)
                        seen_keys.add(key)
                if unique_results:
                    return unique_results

        # Step 4: High-confidence fast path
        try:
            fuzzy_item, fuzzy_score = fuzzy_best_item(core_question)
            if fuzzy_score > 0.95 and fuzzy_item:
                return [_build_response_from_item(fuzzy_item)]
        except Exception as e:
            logger.error(f"Error in high-confidence path: {e}")

        # Step 5: Gather evidence
        near_candidates = []
        try:
            fuzzy_item, fuzzy_score = fuzzy_best_item(core_question)
            embed_candidates = retrieve_topk_embeddings(core_question, top_k=3)

            if fuzzy_item and fuzzy_score > 0.70:
                near_candidates.append(fuzzy_item)

            if embed_candidates:
                for idx, sim in embed_candidates:
                    if sim > 0.68:
                        q_text = app.session_state.question_texts[idx]
                        cand_item = app.session_state.question_data_map.get(q_text)
                        if cand_item and cand_item not in near_candidates:
                            near_candidates.append(cand_item)
        except Exception as e:
            logger.error(f"Error gathering evidence: {e}")

        # Step 6: Decision logic
        if len(near_candidates) > 1:
            return [make_clarifying_question(core_question, near_candidates)]

        if len(near_candidates) == 1:
            return [_build_response_from_item(near_candidates[0])]

        # Step 7: Final fallback
        final_ans, media_type, media_content = find_answer_and_media(question)
        final_response = {
            "text": final_ans,
            "media_type": media_type,
            "media_content": media_content
        }

        if not contains_dataset_keyword(core_question):
            final_response["text"] = "Xin lỗi, tôi không có thông tin về nội dung này."

        return [final_response]
    except Exception as e:
        logger.error(f"Error in get_answer: {e}", exc_info=True)
        return [{"text": "Xin lỗi, đã có lỗi xảy ra.", "media_type": "text", "media_content": None}]


def find_answer_and_media(question):
    """Tìm câu trả lời và media - SỬ DỤNG HYBRID MATCHER"""
    try:
        norm_question = normalize_and_unaccent(question)

        # 0) Direct match trong data gốc
        for item in admissions_data.get('questions', []):
            questions = item.get('question', [])
            if isinstance(questions, str):
                questions = [questions]
            for q in questions:
                if normalize_and_unaccent(q) == norm_question:
                    answer = item.get('answer', "Không có câu trả lời.")
                    images = item.get('images')
                    captions = item.get('captions')
                    video_url = item.get('video_url')
                    if images and isinstance(images, str):
                        images = [images]
                    if video_url:
                        return answer, "video", video_url
                    if images:
                        return answer, "image", (images, captions)
                    return answer, "text", None

        # 1) HYBRID MATCHER
        if HYBRID_MATCHER:
            matched_item = HYBRID_MATCHER.find_match(norm_question)
            if matched_item:
                answer = matched_item.get('answer', "Không có câu trả lời.")
                images = matched_item.get('images')
                captions = matched_item.get('captions')
                video_url = matched_item.get('video_url')
                if images and isinstance(images, str):
                    images = [images]
                if video_url:
                    return answer, "video", video_url
                if images:
                    return answer, "image", (images, captions)
                return answer, "text", None

        # 2) Adaptive routing
        if not contains_dataset_keyword(question):
            return "Xin lỗi, tôi không có thông tin về nội dung này.", "text", None

        FUZZY_STRONG = config.FUZZY_STRONG
        EMBED_STRONG = config.EMBED_STRONG
        FUZZY_MIN = config.FUZZY_MIN
        EMBED_MIN = config.EMBED_MIN

        best_fuzzy_item, best_fuzzy_ratio = fuzzy_best_item(question)

        best_embed_index = None
        best_embed_sim = 0.0
        embed_candidates = []
        if hasattr(app.session_state, 'question_embeddings') and \
                app.session_state.question_embeddings is not None and \
                len(app.session_state.question_texts) > 0:
            embed_candidates = retrieve_topk_embeddings(question, top_k=10)
            if embed_candidates:
                best_embed_index, best_embed_sim = embed_candidates[0]
                candidate_indices = [idx for idx, _ in embed_candidates]
                ce_idx, ce_score = rerank_with_cross_encoder(question, candidate_indices)
                if ce_idx is not None and ce_score >= EMBED_MIN and ce_score >= best_embed_sim:
                    best_embed_index = ce_idx
                    best_embed_sim = ce_score

        chosen_item = None
        if best_fuzzy_ratio >= FUZZY_STRONG and best_fuzzy_ratio >= (best_embed_sim + 0.10):
            chosen_item = best_fuzzy_item
        elif best_embed_index is not None and best_embed_sim >= EMBED_STRONG:
            matched_question = app.session_state.question_texts[best_embed_index]
            chosen_item = app.session_state.question_data_map.get(matched_question)
        else:
            if best_fuzzy_ratio >= FUZZY_MIN or (best_embed_index is not None and best_embed_sim >= EMBED_MIN):
                if (best_embed_index is not None and best_embed_sim >= EMBED_MIN) and (
                        best_embed_sim >= best_fuzzy_ratio):
                    matched_question = app.session_state.question_texts[best_embed_index]
                    chosen_item = app.session_state.question_data_map.get(matched_question)
                else:
                    chosen_item = best_fuzzy_item

        if chosen_item:
            answer = chosen_item.get('answer', "Không có câu trả lời.")
            images = chosen_item.get('images')
            captions = chosen_item.get('captions')
            video_url = chosen_item.get('video_url')
            if images and isinstance(images, str):
                images = [images]
            if video_url:
                return answer, "video", video_url
            if images:
                return answer, "image", (images, captions)
            return answer, "text", None

        return "Xin lỗi, tôi chưa tìm thấy thông tin phù hợp.", "text", None
    except Exception as e:
        logger.error(f"Error in find_answer_and_media: {e}", exc_info=True)
        return "Xin lỗi, đã có lỗi xảy ra.", "text", None


def fuzzy_match_question(question, admissions_data, min_ratio=0.6):
    """Fuzzy match question"""
    try:
        user_q = normalize_and_unaccent(question)
        keys = list(KEYWORD_TO_ITEM_MAP.keys())
        if not keys:
            return None

        if process is not None and fuzz is not None:
            match = process.extractOne(user_q, keys, scorer=fuzz.token_set_ratio)
            if not match:
                return None
            key, score, _ = match
            if float(score) / 100.0 >= float(min_ratio):
                item = KEYWORD_TO_ITEM_MAP.get(key)
                if not item:
                    return None
                answer = item.get('answer', "Không có câu trả lời.")
                images = item.get('images')
                captions = item.get('captions')
                return answer, images, captions
            return None

        # Fallback difflib
        best_key = None
        best_ratio = 0.0
        for k in keys:
            ratio = SequenceMatcher(None, user_q, k).ratio()
            if ratio > best_ratio:
                best_ratio = ratio
                best_key = k

        if best_key is not None and best_ratio >= float(min_ratio):
            item = KEYWORD_TO_ITEM_MAP.get(best_key)
            if not item:
                return None
            answer = item.get('answer', "Không có câu trả lời.")
            images = item.get('images')
            captions = item.get('captions')
            return answer, images, captions
        return None
    except Exception as e:
        logger.error(f"Error in fuzzy_match_question: {e}")
        return None


# --- FLASK ENDPOINTS ---
@app.route('/ask', methods=['POST'])
def ask():
    try:
        data = request.get_json()
        if not data or ('question' not in data and 'choice_id' not in data):
            return jsonify({"error": "Vui lòng cung cấp 'question' hoặc 'choice_id' trong JSON"}), 400

        if getattr(app.session_state, 'question_embeddings', None) is None or \
                not getattr(app.session_state, 'question_texts', []):
            initialize_semantic_resources()

        # Handle choice_id
        if 'choice_id' in data:
            try:
                cid = data.get('choice_id')
                idx = int(cid)
                items = admissions_data.get('questions', [])
                if idx < 0 or idx >= len(items):
                    return jsonify({"error": "Invalid choice_id"}), 400

                item = items[idx]
                ans = item.get('answer', "Xin lỗi, tôi chưa tìm thấy thông tin phù hợp.")
                media_type = "text"
                media_content = None
                images = item.get('images')
                captions = item.get('captions')
                video_url = item.get('video_url')
                if images and isinstance(images, str):
                    images = [images]
                if video_url:
                    media_type = "video"
                    media_content = video_url
                elif images:
                    media_type = "image"
                    media_content = (images, captions)

                session_id = data.get('session_id', None)
                if session_id:
                    append_history(session_id, 'user', f"(chose option) {idx}")
                    append_history(session_id, 'bot', ans)

                # Đây là một lựa chọn, không cần cache
                # Nhưng để nhất quán, chúng ta trả về một danh sách (list)

                # --- SỬA LỖI ĐỊNH DẠNG TRẢ VỀ CHO CHOICE_ID ---
                # Đảm bảo nó trả về đúng định dạng frontend mong đợi
                result = [{
                    "text": ans,
                    "media_type": media_type,
                    "media_content": None,
                    "images": [],
                    "captions": [],
                    "video_url": None
                }]
                if media_type == "video":
                    result[0]["video_url"] = media_content
                elif media_type == "image":
                    images, captions = media_content
                    result[0]["images"] = [f"/images/{os.path.basename(img)}" for img in images if
                                           isinstance(img, str) and img.strip()]
                    result[0]["captions"] = captions if captions else []

                return jsonify(result), 200
                # --- KẾT THÚC SỬA LỖI ---

            except ValueError:
                return jsonify({"error": "choice_id must be an integer index"}), 400
            except Exception as e:
                logger.error(f"Error handling choice: {e}")
                return jsonify({"error": "Lỗi khi xử lý lựa chọn"}), 500

        question = data['question']
        session_id = data.get('session_id', None)

        # Check cache first
        cached_response = response_cache.get(question)
        if cached_response:
            logger.info(f"Cache hit for: {question[:50]}")
            if session_id:
                append_history(session_id, 'user', question)
                try:
                    if cached_response and isinstance(cached_response, list):
                        first_text = cached_response[0].get('text') if isinstance(cached_response[0], dict) else None
                        if first_text:
                            append_history(session_id, 'bot', first_text)
                except Exception:
                    pass
            # --- SỬA LỖI CACHE: Dữ liệu trong cache đã được định dạng ---
            return jsonify(cached_response), 200

        # Process question
        if session_id:
            append_history(session_id, 'user', question)
            question_for_answer = augment_with_context(session_id, question)
        else:
            question_for_answer = question

        # Call the GPT normalization pipeline
        responses = get_answer_with_gpt_normalization(question_for_answer, session_id)

        # --- SỬA LỖI CACHE: Dòng cache cũ đã bị xóa khỏi đây ---

        # Store bot turn
        try:
            if session_id and responses and isinstance(responses, list):
                first_text = responses[0].get('text') if isinstance(responses[0], dict) else None
                if first_text:
                    append_history(session_id, 'bot', first_text)
        except Exception as e:
            logger.error(f"Error storing bot turn: {e}")

        # Build response for frontend
        result = []
        for resp in responses:
            entry = {
                "text": resp["text"],
                "media_type": resp["media_type"],
                "media_content": None,
                "images": [],
                "captions": [],
                "video_url": None
            }
            if isinstance(resp, dict) and "action" in resp:
                entry["action"] = resp["action"]
            if "options" in resp:
                entry["options"] = resp["options"]
            if resp["media_type"] == "video" and resp["media_content"]:
                entry["video_url"] = resp["media_content"]
            elif resp["media_type"] == "image" and resp["media_content"]:
                images, captions = resp["media_content"]
                entry["images"] = [f"/images/{os.path.basename(img)}" for img in images if
                                   isinstance(img, str) and img.strip()]
                entry["captions"] = captions if captions else []
            result.append(entry)

        # Remove duplicates
        try:
            unique = []
            seen = set()
            for r in result:
                t = (r.get('text') or '').strip()
                if not t:
                    key = '__empty__'
                else:
                    key = t
                if key in seen:
                    continue
                seen.add(key)
                unique.append(r)

            # --- SỬA LỖI CACHE: Cache đối tượng `unique` (đã định dạng) ---
            response_cache.set(question, unique)

            return jsonify(unique), 200
        except Exception:
            # --- SỬA LỖI CACHE: Cache `result` nếu lọc `unique` thất bại ---
            response_cache.set(question, result)
            return jsonify(result), 200
    except Exception as e:
        logger.error(f"Error in /ask endpoint: {e}", exc_info=True)
        return jsonify({"error": "Internal server error"}), 500


@app.route('/images/<path:filename>')
def serve_image(filename):
    try:
        # Debug: Log request details
        logger.info(f"[IMAGE REQUEST] Requested: {filename}")
        logger.info(f"[IMAGE REQUEST] Images dir: {config.IMAGES_DIR}")
        logger.info(f"[IMAGE REQUEST] Absolute path: {os.path.abspath(config.IMAGES_DIR)}")

        # Check if images directory exists
        if not os.path.exists(config.IMAGES_DIR):
            logger.error(f"[IMAGE ERROR] Images directory does not exist: {config.IMAGES_DIR}")
            return jsonify({"error": "Images directory not found"}), 404

        # Check if file exists
        file_path = os.path.join(config.IMAGES_DIR, filename)
        if not os.path.exists(file_path):
            logger.error(f"[IMAGE ERROR] File not found: {file_path}")
            # List available files for debugging
            try:
                available_files = os.listdir(config.IMAGES_DIR)
                logger.error(f"[IMAGE ERROR] Available files: {available_files[:10]}")  # First 10 files
            except Exception as list_err:
                logger.error(f"[IMAGE ERROR] Cannot list directory: {list_err}")
            return jsonify({"error": f"Image not found: {filename}"}), 404

        # Check file permissions
        if not os.access(file_path, os.R_OK):
            logger.error(f"[IMAGE ERROR] No read permission for: {file_path}")
            return jsonify({"error": "Permission denied"}), 403

        logger.info(f"[IMAGE SUCCESS] Serving: {file_path}")
        return send_from_directory(config.IMAGES_DIR, filename)

    except Exception as e:
        logger.error(f"[IMAGE ERROR] Unexpected error serving {filename}: {e}", exc_info=True)
        return jsonify({"error": f"Error: {str(e)}"}), 500


@app.route('/')
def index():
    try:
        return render_template('index.html')
    except Exception as e:
        logger.error(f"Error rendering index: {e}")
        return "Error loading page", 500


@app.route('/status', methods=['GET'])
def status():
    try:
        n_items = len(admissions_data.get('questions', []))
        n_texts = len(getattr(app.session_state, 'question_texts', []) or [])
        has_emb = app.session_state.question_embeddings is not None
        has_knn = getattr(app.session_state, 'knn_index', None) is not None
        has_faiss = getattr(app.session_state, 'faiss_index', None) is not None
        has_ce = model_manager.cross_encoder_model is not None
        has_sbert = model_manager.sbert_model is not None
        has_hybrid = HYBRID_MATCHER is not None
        has_gpt = client is not None and config.USE_GPT

        return jsonify({
            "status": "healthy",
            "items": n_items,
            "questions_indexed": n_texts,
            "embeddings": bool(has_emb),
            "faiss_index": bool(has_faiss),
            "knn_index": bool(has_knn),
            "cross_encoder": bool(has_ce),
            "sbert": bool(has_sbert),
            "hybrid_matcher": bool(has_hybrid),
            "gpt_enabled": bool(has_gpt),
            "gpt_normalization": config.USE_GPT_NORMALIZATION,
            "cache": {
                "size": len(response_cache.cache),
                "hits": response_cache.hits,
                "misses": response_cache.misses,
                "hit_rate": f"{response_cache.hits / max(1, response_cache.hits + response_cache.misses) * 100:.1f}%"
            },
            "sessions": {
                "active": len(session_store.sessions)
            }
        }), 200
    except Exception as e:
        logger.error(f"Error in status endpoint: {e}")
        return jsonify({"status": "unhealthy", "error": str(e)}), 500


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint for monitoring"""
    try:
        return jsonify({
            "status": "healthy",
            "timestamp": datetime.now().isoformat()
        }), 200
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return jsonify({"status": "unhealthy", "error": str(e)}), 500


@app.route('/clear_cache', methods=['POST'])
def clear_cache():
    """Clear response cache (admin endpoint)"""
    try:
        response_cache.clear()
        logger.info("Cache cleared manually")
        return jsonify({"message": "Cache cleared successfully"}), 200
    except Exception as e:
        logger.error(f"Error clearing cache: {e}")
        return jsonify({"error": "Failed to clear cache"}), 500


# --- GRACEFUL SHUTDOWN ---
import signal


def graceful_shutdown(signum, frame):
    """Handle graceful shutdown"""
    logger.info("Received shutdown signal, cleaning up...")
    try:
        response_cache.clear()
        logger.info("Cache cleared")
    except Exception as e:
        logger.error(f"Error during cleanup: {e}")
    logger.info("Shutdown complete")
    sys.exit(0)


signal.signal(signal.SIGINT, graceful_shutdown)
signal.signal(signal.SIGTERM, graceful_shutdown)

# --- MAIN ---
if __name__ == "__main__":
    try:
        logger.info("=" * 60)
        logger.info("STARTING OPTIMIZED CHATBOT SERVER")
        logger.info("=" * 60)

        # Log configuration
        logger.info(f"Configuration:")
        logger.info(f"  Data path: {config.DATA_PATH}")
        logger.info(f"  Images dir: {config.IMAGES_DIR}")
        logger.info(f"  Cache size: {config.CACHE_SIZE}")
        logger.info(f"  GPT enabled: {config.USE_GPT}")
        logger.info(f"  GPT normalization: {config.USE_GPT_NORMALIZATION}")
        logger.info(f"  Thresholds: FUZZY_STRONG={config.FUZZY_STRONG}, FUZZY_MIN={config.FUZZY_MIN}")
        logger.info(f"              EMBED_STRONG={config.EMBED_STRONG}, EMBED_MIN={config.EMBED_MIN}")

        # Initialize semantic resources
        logger.info("Initializing semantic resources...")
        initialize_semantic_resources()

        # Log statistics
        logger.info(f"Statistics:")
        logger.info(f"  Total questions: {len(admissions_data.get('questions', []))}")
        logger.info(f"  Keyword mappings: {len(KEYWORD_TO_ITEM_MAP)}")
        logger.info(f"  Hybrid matcher: {'Ready' if HYBRID_MATCHER else 'Not available'}")
        logger.info(f"  SBERT model: {'Loaded' if model_manager.sbert_model else 'Not loaded'}")
        logger.info(f"  Cross-encoder: {'Loaded' if model_manager.cross_encoder_model else 'Not loaded'}")
        logger.info(f"  OpenAI client: {'Ready' if client else 'Not available'}")

        logger.info("=" * 60)
        logger.info("Server ready! Starting Flask...")
        logger.info("=" * 60)

        # Start Flask server
        port = int(os.getenv('PORT', 8080))
        debug = os.getenv('DEBUG', 'false').lower() == 'true'

        app.run(
            host='0.0.0.0',
            port=port,
            debug=debug
        )

    except Exception as e:
        logger.critical(f"Failed to start server: {e}", exc_info=True)
        sys.exit(1)