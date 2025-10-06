# filepath: rule-based-fixed.py
# Copyright (c) [2025] [Nguyễn Minh Tấn Phúc]. Bảo lưu mọi quyền.
# Nguồn: https://tlmchattest.streamlit.app/
import json
import os
import re
import torch
import unicodedata# --- GPT-4 Turbo Keyword Extractor ---
from openai import OpenAI
client = OpenAI()
from difflib import SequenceMatcher
from types import SimpleNamespace
from flask import Flask, request, jsonify, send_from_directory, render_template

# NEW: fast fuzzy matching with graceful fallback
try:
    from rapidfuzz import fuzz, process  # type: ignore
except Exception:
    fuzz = None  # type: ignore
    process = None  # type: ignore

# --- KHỞI TẠO FLASK APP ---
app = Flask(__name__)
# Provide a lightweight session_state to avoid AttributeError in optional features
if not hasattr(app, 'session_state'):
    app.session_state = SimpleNamespace(
        question_embeddings=None,
        question_texts=[],
        question_data_map={},
        faiss_index=None,
        knn_index=None,  # sklearn NearestNeighbors (cosine) fallback
        sessions={}  # in-memory conversation store: {session_id: [{role, text}]}
    )


# Conversation helpers for lightweight per-session memory

def get_session_history(session_id: str):
    try:
        if not session_id:
            return []
        return app.session_state.sessions.get(session_id, [])
    except Exception:
        return []


def append_history(session_id: str, role: str, text: str, limit: int = 10):
    try:
        if not session_id:
            return
        hist = app.session_state.sessions.setdefault(session_id, [])
        hist.append({"role": role, "text": (text or "")[:2000]})
        # keep only last N turns
        if len(hist) > limit:
            app.session_state.sessions[session_id] = hist[-limit:]
    except Exception:
        pass


def last_user_turn(session_id: str) -> str:
    try:
        hist = get_session_history(session_id)
        for m in reversed(hist):
            if m.get('role') == 'user' and m.get('text'):
                return m.get('text')
        return ""
    except Exception:
        return ""


def looks_context_dependent(q: str) -> bool:
    # Heuristics to detect short/elliptical follow-ups likely needing context
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
    except Exception:
        return False


def augment_with_context(session_id: str, q: str) -> str:
    """Nếu câu hỏi là keyword exact match thì giữ nguyên,
    nếu có từ nghi vấn thì ghép với ngữ cảnh trước, còn lại trả nguyên văn."""
    try:
        norm_q_unaccented = normalize_and_unaccent(q)

        # 1. Nếu là keyword exact match trong dataset -> giữ nguyên
        if norm_q_unaccented in KEYWORD_TO_ITEM_MAP:
            return q

        # 2. Nếu là trường hợp đặc biệt 'hiệu trưởng' -> giữ nguyên
        if re.fullmatch(r"hieu\s*truong", norm_q_unaccented):
            return q

        prev = last_user_turn(session_id)
        if not prev:
            return q

        # 3. Nếu câu hỏi chứa từ nghi vấn -> ghép với câu trước
        interrogatives = [
            r"\bai\b", r"\bgi\b", r"\bgi\s*\?", r"o\s*dau", r"khi\s*nao",
            r"bao\s*gio", r"bao\s*nhieu", r"the\s*nao", r"nao", r"khong", r"sao"
        ]
        for pat in interrogatives:
            if re.search(pat, norm_q_unaccented):
                return f"{prev} ; {q}"

        # 4. Mặc định: trả về nguyên văn (không ghép)
        return q
    except Exception:
        return q


# --- CẤU HÌNH VÀ TẢI DỮ LIỆU ---
try:
    with open(os.path.join(os.path.dirname(__file__), 'admissions_data.json'), 'r', encoding='utf-8') as f:
        admissions_data = json.load(f)
except Exception as e:
    print(f"Lỗi khi tải admissions_data.json: {e}")
    admissions_data = {"questions": []}


# --- CÁC HÀM TIỆN ÍCH ---

def remove_vietnamese_accents(text):
    """
    Hàm này nhận một chuỗi văn bản tiếng Việt và trả về chuỗi đó không có dấu.
    """
    return "".join(c for c in unicodedata.normalize('NFD', text)
                   if unicodedata.category(c) != 'Mn')


def remove_vietnamese_stopwords(tokenized_text):
    stopwords = [
        'là', 'và', 'có', 'của', 'trong', 'được', 'cho', 'với', 'tại', 'từ',
        'bởi', 'để', 'như', 'thì', 'mà', 'này', 'kia', 'đó', 'nào', 'cái',
        'những', 'một', 'các', 'đã', 'lại', 'còn', 'nếu', 'vì', 'do', 'bị',
        'về'
    ]
    tokens = tokenized_text.split() if isinstance(tokenized_text, str) else tokenized_text
    return [token for token in tokens if token not in stopwords]


def normalize_text(text):
    text = text.lower().strip()
    text = re.sub(r'\s+', ' ', text)
    return text

# Inserted: GPT-assisted keyword extractor (defensive about missing client)
def extract_keyword_with_gpt_turbo(question, all_keywords, skip_gpt: bool = False):
    """
    Sử dụng GPT-4 Turbo để chọn keyword có trong danh sách all_keywords
    phù hợp nhất với câu hỏi người dùng.
    """
    try:
        # If caller requested to skip GPT (e.g., UI button action), avoid calling the API
        if skip_gpt:
            # Short-circuit: do not call the OpenAI API for UI/button-triggered requests
            print("[GPT] Skipped by skip_gpt flag (UI/button request)")
            return None
        # Defensive: require an API client named `client` to exist in globals()
        client = globals().get('client')
        if client is None:
            print("[GPT] No 'client' available in globals() - skipping GPT extraction")
            return None

        prompt = f"""
        Dưới đây là danh sách keyword có sẵn:
        {', '.join(all_keywords[:500])}
        Hãy chọn từ hoặc cụm từ trong danh sách trên phù hợp nhất với câu hỏi sau:
        "{question}"
        Chỉ trả về đúng keyword (phải có trong danh sách), không thêm ký tự khác.
        """
        resp = client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[
                {"role": "system", "content": "Bạn là hệ thống trích xuất keyword chính xác."},
                {"role": "user", "content": prompt},
            ],
            temperature=0,
            max_tokens=32,
        )
        kw = resp.choices[0].message.content.strip().lower()
        if kw in all_keywords:
            print(f"[GPT] Keyword chọn: {kw}")
            return kw
        else:
            print(f"[GPT] Keyword không trùng: {kw}")
            return None
    except Exception as e:
        print(f"[GPT ERROR] {e}")
        return None


def add_li_ly_variants(keyword):
    # Thêm cả hai biến thể 'li' và 'ly' cho chính tả tiếng Việt
    variants = {keyword}
    if ' li ' in f' {keyword} ':
        variants.add(keyword.replace(' li ', ' ly '))
    if ' ly ' in f' {keyword} ':
        variants.add(keyword.replace(' ly ', ' li '))
    # Xử lý ở đầu/cuối từ
    if keyword.endswith('li'):
        variants.add(keyword[:-2] + 'ly')
    if keyword.endswith('ly'):
        variants.add(keyword[:-2] + 'li')
    if keyword.startswith('li '):
        variants.add('ly ' + keyword[3:])
    if keyword.startswith('ly '):
        variants.add('li ' + keyword[3:])
    return variants


def get_all_question_keywords():
    # Trích xuất tất cả từ khóa câu hỏi từ admissions_data.json, chuẩn hóa và loại bỏ dấu
    keywords = set()
    for item in admissions_data.get('questions', []):
        questions = item.get('question', [])
        if isinstance(questions, str):
            questions = [questions]
        for q in questions:
            norm_q = normalize_text(q)
            unaccented_q = remove_vietnamese_accents(norm_q)
            # Thêm cả dạng có dấu và không dấu
            if len(norm_q) > 2:
                for v in add_li_ly_variants(norm_q):
                    keywords.add(v)
            if len(unaccented_q) > 2:
                for v in add_li_ly_variants(unaccented_q):
                    keywords.add(v)
    # Sắp xếp theo độ dài giảm dần để tránh trùng lặp một phần
    return sorted(keywords, key=lambda x: -len(x))


QUESTION_KEYWORDS = get_all_question_keywords()


# Hàm phụ: chuẩn hóa và loại bỏ dấu cho tất cả thao tác so khớp
def normalize_and_unaccent(text):
    norm = remove_vietnamese_accents(normalize_text(text))
    # Chuyển 'ly' thành 'li' để so khớp
    norm = re.sub(r'\bly\b', 'li', norm)
    return norm


# Loại bỏ các cụm dẫn nhập thường gặp (không mang ý nghĩa nội dung chính)
def strip_leadin_phrases(text: str) -> str:
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
        # NEW: common school/generic prefixes
        r'^truong\s*co\s*',
        r'^truong\s*co\s*cac\s*',
        r'^truong\s*co\s*nhung\s*'
    ]
    for pat in leadins:
        norm = re.sub(pat, '', norm)
    # Trim trailing filler words like "của"/"về"
    norm = re.sub(r'\b(cua|ve)\s*$', '', norm).strip()
    # NEW: trim common Vietnamese interrogatives at tail
    norm = re.sub(r'\b(bao\s*nhieu|nao|khong|khong\s*\?|gi|gi\s*\?)\s*$', '', norm).strip()
    # Also trim trailing "cua truong ..." to expose core keyword
    norm = re.sub(r'\bcua\s*(truong|thpt|trg|truong\s*thpt|trung\s*hoc\s*pho\s*thong)\b.*$', '', norm).strip()
    return norm


# Tách ý nhỏ cho câu hỏi nhiều ý, không dấu

def split_subquestions(text):
    norm_text = normalize_and_unaccent(text)
    # NEW GUARD: nếu toàn bộ chuỗi khớp 1 từ khóa đã biết, không tách
    try:
        if KEYWORD_TO_ITEM_MAP and norm_text in KEYWORD_TO_ITEM_MAP:
            return [norm_text]
    except Exception:
        pass
    # Chỉ tách theo các liên từ rõ ràng, dùng non-capturing group để không giữ lại từ nối
    conjunctions = [
        r'va', r'và', r'hoac', r'hay',
        r'voi', r'với', r'voi\s*lai', r'với\s*lại', 'vs',
        r'cung', r'cùng', r'cung\s*voi', r'cùng\s*với',
        r'roi', r'rồi', r'xong', r'tiep', r'tiep\s*theo', r'sau\s*do', r'sau\s*do',
        r'con', r'nua', r'kèm', 'kem'
    ]
    # Thêm các ký tự phân tách phổ biến: ; , / + & |
    sep_chars = r'[;,/\+&|]'
    pattern = sep_chars + r'|\b(?:' + '|'.join(conjunctions) + r')\b'
    parts = re.split(pattern, norm_text)
    subqs = [p.strip() for p in parts if p and len(p.strip()) > 2]

    # NEW: tách theo đánh số liệt kê (1., 2), i), a), - ...) nếu chưa tách được
    if len(subqs) <= 1:
        numbered = re.split(
            r'(?:\b\d+[).]|\bthu\s*(?:nhat|hai|ba|tu|nam|sau|bay|tam|chin)\b|\b(?:i|ii|iii|iv|v|vi|vii|viii|ix|x)\))',
            norm_text)
        numbered = [s.strip() for s in numbered if s and len(s.strip()) > 2 and not re.fullmatch(
            r'(?:i|ii|iii|iv|v|vi|vii|viii|ix|x|nhat|hai|ba|tu|nam|sau|bay|tam|chin)', s)]
        if len(numbered) > 1:
            subqs = numbered

    # NEW: nếu không tách được nhưng có nhiều danh xưng (thay/co), tách theo mốc danh xưng
    if len(subqs) <= 1:
        tokens = norm_text.split()
        honorifics = {"thay", "co"}
        segments = []
        current = []
        for t in tokens:
            if t in honorifics:
                if current:
                    segments.append(' '.join(current).strip())
                    current = []
                current = [t]
            else:
                if current:
                    current.append(t)
        if current:
            segments.append(' '.join(current).strip())
        # Giữ các đoạn có ít nhất 2 token (danh xưng + tên)
        segments = [s for s in segments if len(s.split()) >= 2]
        if len(segments) > 1:
            return segments

    # NEW: nếu vẫn không tách được, thử tách theo các từ khóa đã biết trong kho dữ liệu (khớp liên tiếp)
    if len(subqs) <= 1:
        try:
            # Sanitize punctuation to avoid breaking keyword spans (e.g., hoc phi' diem chuan)
            tokens = [t for t in re.split(r'\s+', re.sub(r'[^\w]+', ' ', norm_text)) if t]
            n = len(tokens)
            if n >= 2 and KEYWORD_TO_ITEM_MAP:
                # Dùng giá trị cache MAX_KEY_LEN
                LMAX = MAX_KEY_LEN if 'MAX_KEY_LEN' in globals() else 6
                STOP_TOKENS = COMMON_STOP_TOKENS if 'COMMON_STOP_TOKENS' in globals() else {"truong", "co", "cua", "ve",
                                                                                            "la", "nao", "gi", "cai",
                                                                                            "cac", "nhung", "o", "dau"}
                i = 0
                found_spans = []
                while i < n:
                    matched = False
                    Lmax = min(LMAX, n - i)
                    for L in range(Lmax, 0, -1):
                        phrase = ' '.join(tokens[i:i + L])
                        if phrase in KEYWORD_TO_ITEM_MAP:
                            # Bỏ qua match quá yếu (toàn stop token 1-2 từ)
                            toks = phrase.split()
                            if len(toks) == 1 and toks[0] in STOP_TOKENS:
                                continue
                            if len(toks) == 2 and all(t in STOP_TOKENS for t in toks):
                                continue
                            found_spans.append((i, i + L, phrase))
                            i += L
                            matched = True
                            break
                    if not matched:
                        i += 1
                if len(found_spans) > 1:
                    # Hợp nhất các span tách biệt để tạo các tiểu ý
                    result = [p for (_, _, p) in found_spans]
                    # Loại bỏ trùng lặp liên tiếp
                    dedup = []
                    for p in result:
                        if not dedup or dedup[-1] != p:
                            dedup.append(p)
                    if len(dedup) > 1:
                        return dedup
        except Exception:
            pass

    # NEW: nếu vẫn không tách được, thử bỏ từ đệm (gap-tolerant) rồi quét từ khóa (bỏ dấu câu)
    if len(subqs) <= 1:
        try:
            tokens = [t for t in re.split(r'\s+', re.sub(r'[^\w]+', ' ', norm_text)) if t]
            if tokens and KEYWORD_TO_ITEM_MAP:
                FILLERS = {
                    'la', 'thi', 'thoi', 'nhe', 'nha', 'voi', 'va', 'và', 'vs', 'lai', 'nua', 'cai', 'cua', 've', 'la',
                    'cac', 'nhung', 'o', 'dau', 'roi', 'xong', 'tiep', 'sau', 'do', 'tiep', 'theo'
                }
                filtered = [t for t in tokens if t not in FILLERS]
                if len(filtered) >= 2:
                    LMAX = MAX_KEY_LEN if 'MAX_KEY_LEN' in globals() else 6
                    i = 0
                    n = len(filtered)
                    spans = []
                    while i < n:
                        matched = False
                        Lmax = min(LMAX, n - i)
                        for L in range(Lmax, 0, -1):
                            phrase = ' '.join(filtered[i:i + L])
                            if phrase in KEYWORD_TO_ITEM_MAP:
                                spans.append(phrase)
                                i += L
                                matched = True
                                break
                        if not matched:
                            i += 1
                    if len(spans) > 1:
                        # Loại trùng lặp liên tiếp
                        dedup2 = []
                        for p in spans:
                            if not dedup2 or dedup2[-1] != p:
                                dedup2.append(p)
                        if len(dedup2) > 1:
                            return dedup2
        except Exception:
            pass

    # NEW: nếu vẫn không tách được, thử nhận diện "đầu từ khóa" (keyword heads) 2 từ
    # Ví dụ: 'diem chuan' là tiền tố của nhiều key như 'diem chuan nam nay', 'diem chuan cua truong', ...
    if len(subqs) <= 1 and KEYWORD_TO_ITEM_MAP:
        try:
            tokens = [t for t in re.split(r'\s+', re.sub(r'[^\w]+', ' ', norm_text)) if t]
            if len(tokens) >= 2:
                # Sử dụng cache KEY_HEADS_2
                heads = KEY_HEADS_2 if 'KEY_HEADS_2' in globals() else set()
                # Quét theo cửa sổ 2 từ để tìm các đầu-key xuất hiện theo thứ tự
                i = 0
                found_heads = []
                while i <= len(tokens) - 2:
                    candidate = ' '.join(tokens[i:i + 2])
                    if candidate in heads:
                        if not found_heads or found_heads[-1] != candidate:
                            found_heads.append(candidate)
                        i += 2
                    else:
                        i += 1
                if len(found_heads) > 1:
                    return found_heads
        except Exception:
            pass

    return subqs


def find_multi_keyword_spans(norm_text: str):
    """Return a list of non-overlapping keyword phrases found in norm_text using KEYWORD_TO_ITEM_MAP.
    Greedy left-to-right longest-match; phrases are normalized (unaccented, lower).
    """
    try:
        if not norm_text or not KEYWORD_TO_ITEM_MAP:
            return []
        # sanitize punctuation, split to tokens
        tokens = [t for t in re.split(r"\s+", re.sub(r"[^\w]+", " ", norm_text)) if t]
        if not tokens:
            return []
        LMAX = MAX_KEY_LEN if 'MAX_KEY_LEN' in globals() else 6
        i = 0
        n = len(tokens)
        spans = []
        while i < n:
            matched = False
            Lmax = min(LMAX, n - i)
            for L in range(Lmax, 0, -1):
                phrase = ' '.join(tokens[i:i + L])
                if phrase in KEYWORD_TO_ITEM_MAP:
                    # Avoid extremely weak matches consisting solely of common stop tokens
                    toks = phrase.split()
                    STOP_TOKENS = COMMON_STOP_TOKENS if 'COMMON_STOP_TOKENS' in globals() else {"truong", "co", "cua",
                                                                                                "ve", "la", "nao", "gi",
                                                                                                "cai", "cac", "nhung",
                                                                                                "o", "dau"}
                    if len(toks) == 1 and toks[0] in STOP_TOKENS:
                        continue
                    if len(toks) == 2 and all(t in STOP_TOKENS for t in toks):
                        continue
                    spans.append(phrase)
                    i += L
                    matched = True
                    break
            if not matched:
                i += 1
        # Dedup consecutive
        dedup = []
        for p in spans:
            if not dedup or dedup[-1] != p:
                dedup.append(p)
        return dedup
    except Exception:
        return []


# NEW: helper to detect whether a user query contains any keyword/phrase from the dataset
# Uses normalized (lower, unaccented) matching with span scan and boundary-based contains

def contains_dataset_keyword(text: str) -> bool:
    try:
        if not text or not KEYWORD_TO_ITEM_MAP:
            return False
        nq = normalize_and_unaccent(text)
        # Direct key
        if nq in KEYWORD_TO_ITEM_MAP:
            return True
        # Span scan
        spans = find_multi_keyword_spans(nq)
        if spans:
            return True

        # Boundary-based contains check (both directions), punctuation-stripped variants
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
                return True
            kp = re.sub(r"\W+", " ", key).strip()
            if (kp and np) and (_wb_contains(np, kp) or _wb_contains(kp, np)):
                return True
        return False
    except Exception:
        return False


# =========================================================================
# START: REFACTORED get_answer FUNCTION FOR RELIABILITY
# =========================================================================
def get_answer(question, skip_gpt: bool = False):
    """
    Handles user questions with a clear, structured logic flow.
    1. Prioritizes exact and near-perfect matches for immediate, accurate answers.
    2. Gathers evidence from multiple sources (fuzzy, semantic) if the answer isn't obvious.
    3. Asks for clarification only when there is genuine ambiguity between strong candidates.
    4. Falls back to complex parsing for multi-intent questions as a last resort.
    """

    # Helper function to build a standard response object from a data item
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

    # --- Step 1: Normalization and Basic Checks ---
    norm_question = normalize_text(question)
    norm_unaccent_question = normalize_and_unaccent(question)

    # Handle special cases like "hiệu trưởng"
    if re.fullmatch(r"\s*hiệu\s*trưởng\s*[?.!]*\s*", norm_question) or re.fullmatch(r"\s*hieu\s*truong\s*[?.!]*\s*",
                                                                                    norm_unaccent_question):
        hardcoded_response = "Bạn muốn biết về hiệu trưởng hiện tại hay hiệu trưởng qua từng thời kỳ?"
        return [
            {"text": hardcoded_response, "media_type": "text", "media_content": None, "action": "hieutruong_choices"}]

    # Clean the question for matching
    SCHOOL_NAME_VARIANTS = [
        "trường thpt", "thpt", "trường trung học phổ thông", "trung học phổ thông", "ten lơ men", "ten lơ man",
        "ernst thälmann", "ernst thalmann", "trường cấp 3", "tlm", "trường ten lơ man"
    ]
    school_pattern = r"\b(" + r"|".join([re.escape(v) for v in SCHOOL_NAME_VARIANTS]) + r")\b"
    core_question = re.sub(school_pattern, "", norm_question, flags=re.IGNORECASE).strip()
    if not core_question:  # If question was only the school name
        core_question = norm_question
    # Normalize away common lead-in/trailing filler like 'cua truong', 've', etc.
    # This helps queries such as "học phí của trường" -> expose core "học phí" to matcher.
    try:
        core_question = strip_leadin_phrases(core_question)
        # strip_leadin_phrases returns a normalized (unaccented) string; keep as-is since
        # downstream matchers re-normalize inputs internally.
    except Exception:
        pass

    # --- GPT keyword extractor ---
    try:
        # Only call GPT keyword extractor when not explicitly skipped (UI buttons)
        gpt_kw = extract_keyword_with_gpt_turbo(question, list(KEYWORD_TO_ITEM_MAP.keys()), skip_gpt=skip_gpt)
        if gpt_kw and gpt_kw in KEYWORD_TO_ITEM_MAP:
            item = KEYWORD_TO_ITEM_MAP[gpt_kw]
            # Previously returned text-only here which caused images/media to be dropped
            # when the GPT keyword extractor matched. Use the helper to preserve media.
            return [_build_response_from_item(item)]
    except Exception as e:
        # Fail silently and continue to fuzzy/semantic pipeline
        print(f"[GPT] extract_keyword_with_gpt_turbo error: {e}")

    # --- Step 2: High-Confidence Fast Path ---
    # This is the most important fix: check for a near-perfect match FIRST and return immediately.
    try:
        fuzzy_item, fuzzy_score = fuzzy_best_item(core_question)
        if fuzzy_score > 0.95 and fuzzy_item:
            return [_build_response_from_item(fuzzy_item)]
    except Exception:
        pass  # If this fails, we proceed to more complex logic

    # --- Step 3: Gather Evidence for Ambiguous Cases ---
    # Only run if the fast path didn't produce a clear winner.
    near_candidates = []
    try:
        # Get fuzzy and semantic candidates
        fuzzy_item, fuzzy_score = fuzzy_best_item(core_question)
        embed_candidates = retrieve_topk_embeddings(core_question, top_k=3)

        # Add fuzzy candidate if it's reasonably good
        if fuzzy_item and fuzzy_score > 0.70:
            near_candidates.append(fuzzy_item)

        # Add semantic candidates if they are good and not already present
        if embed_candidates:
            for idx, sim in embed_candidates:
                if sim > 0.68:  # Confidence threshold for semantic match
                    q_text = app.session_state.question_texts[idx]
                    cand_item = app.session_state.question_data_map.get(q_text)
                    if cand_item and cand_item not in near_candidates:
                        near_candidates.append(cand_item)
    except Exception:
        pass

    # --- Step 4: Decision Logic ---
    # If we have multiple strong, distinct candidates, ask for clarification.
    if len(near_candidates) > 1:
        return [make_clarifying_question(core_question, near_candidates)]

    # If we have exactly one strong candidate, answer with it.
    if len(near_candidates) == 1:
        return [_build_response_from_item(near_candidates[0])]

    # --- Step 5: Fallback to Multi-Intent Parsing ---
    # This logic runs only if no single clear answer was found above.
    sub_questions = find_multi_keyword_spans(normalize_and_unaccent(core_question))
    if len(sub_questions) <= 1:
        sub_questions = split_subquestions(core_question)

    if len(sub_questions) > 1:
        results = []
        for subq in sub_questions:
            # For sub-questions, we want a direct answer, not more ambiguity checks.
            # So we use a direct lookup.
            item = KEYWORD_TO_ITEM_MAP.get(normalize_and_unaccent(subq))
            if item:
                results.append(_build_response_from_item(item))

        if results:
            # Deduplicate identical results
            unique_results = []
            seen_keys = set()
            for r in results:
                key = r.get("text", "")
                if key not in seen_keys:
                    unique_results.append(r)
                    seen_keys.add(key)
            return unique_results

    # --- Step 6: Final Fallback ---
    # If all else fails, use the generic find_answer_and_media on the original question.
    # This acts as a catch-all for complex phrasing the above logic might miss.
    final_ans, media_type, media_content = find_answer_and_media(question)
    final_response = {
        "text": final_ans,
        "media_type": media_type,
        "media_content": media_content
    }

    # Gate the final fallback if no keyword is present at all
    if not contains_dataset_keyword(core_question):
        final_response["text"] = "Xin lỗi, tôi không có thông tin về nội dung này."

    return [final_response]


# =========================================================================
# END: REFACTORED get_answer FUNCTION
# =========================================================================


def get_school_info_answer():
    for item in admissions_data.get('questions', []):
        questions = item.get('question', [])
        if isinstance(questions, str):
            questions = [questions]
        for q in questions:
            norm_q = normalize_text(q)
            if norm_q in ["khái quát", "sơ lược", "khái quát về trường", "sơ lược về trường",
                          "cho tôi thông tin sơ lược và khái quát về trường"]:
                return item.get('answer', "Thông tin về trường THPT Ten Lơ Man...")
    return "Thông tin về trường THPT Ten Lơ Man..."


def remove_school_name(question):
    pattern = r"(trường\s+thpt\s+ten\s+lơ\s+man|thpt\s+ten\s+lơ\s+man|ernst\s+thälmann|ernst\s+thalmann)"
    return re.sub(pattern, "", question, flags=re.IGNORECASE).strip()


def find_answer(core_question):
    return find_answer_and_media(core_question)[0]


def split_sticky_words(text):
    """Word-segment Vietnamese using pyvi if available; fallback to original text."""
    try:
        from pyvi import ViTokenizer
        return ViTokenizer.tokenize(text)
    except Exception:
        return text


# --- CẤU HÌNH VÀ TẢI DỮ LIỆU ---
device = "cuda" if torch.cuda.is_available() else "cpu"

QUESTION_KEYWORDS = get_all_question_keywords()

# Tạo từ điển tra cứu cho so khớp từ khóa trực tiếp và n-gram
KEYWORD_ANSWER_MAP = {}
ALL_KEYWORDS_SET = set()
KEYWORD_TO_ITEM_MAP = {}
KEYWORD_TO_ITEMS_MAP = {}
for item in admissions_data.get('questions', []):
    questions = item.get('question', [])
    if isinstance(questions, str):
        questions = [questions]
    for q in questions:
        key = normalize_and_unaccent(q)
        KEYWORD_ANSWER_MAP[key] = item
        KEYWORD_TO_ITEM_MAP[key] = item
        ALL_KEYWORDS_SET.add(key)
        # Multi-map for duplicates
        lst = KEYWORD_TO_ITEMS_MAP.setdefault(key, [])
        if item not in lst:
            lst.append(item)

# NEW: Cached constants for performance-sensitive scans
try:
    MAX_KEY_LEN = max((len(k.split()) for k in KEYWORD_TO_ITEM_MAP.keys()), default=6)
except Exception:
    MAX_KEY_LEN = 6
COMMON_STOP_TOKENS = {"truong", "co", "cua", "ve", "la", "nao", "gi", "cai", "cac", "nhung", "o", "dau", "ai"}
KEY_HEADS_2 = set()
for _k in KEYWORD_TO_ITEM_MAP.keys():
    _ks = _k.split()
    if len(_ks) >= 2:
        KEY_HEADS_2.add(' '.join(_ks[:2]))


# NEW: Lazy-load models

def load_cross_encoder_model():
    try:
        from sentence_transformers import CrossEncoder
        return CrossEncoder("cross-encoder/stsb-xlm-r-multilingual")
    except Exception:
        return None


cross_encoder_model = load_cross_encoder_model()


def load_sentence_transformer_model():
    try:
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer("paraphrase-multilingual-mpnet-base-v2")
    except Exception:
        return None


sbert_model = load_sentence_transformer_model()


# --- Initialize semantic resources (embeddings + maps) once ---

def initialize_semantic_resources():
    """Build embeddings and lookup maps so semantic search works for full sentences."""
    try:
        question_texts, embeddings, text_to_item = build_question_embeddings_and_maps(admissions_data)
        # Store in app.session_state for reuse
        app.session_state.question_texts = question_texts or []
        app.session_state.question_embeddings = embeddings
        app.session_state.question_data_map = text_to_item or {}
        # Build a fast cosine KNN index (fallback when FAISS isn't available)
        try:
            if embeddings is not None and embeddings.shape[0] > 0:
                from sklearn.neighbors import NearestNeighbors
                # embeddings are already normalized; cosine distance ~ 1 - cosine sim
                nn = NearestNeighbors(metric='cosine', algorithm='auto')
                # sklearn expects numpy arrays
                emb_np = embeddings.detach().cpu().numpy()
                nn.fit(emb_np)
                app.session_state.knn_index = nn
        except Exception as e:
            # Keep running even if KNN init fails
            print(f"Khởi tạo KNN index thất bại: {e}")
        # NEW: Build FAISS index for faster top-K retrieval (cosine via inner product)
        try:
            if embeddings is not None and embeddings.shape[0] > 0:
                import faiss  # type: ignore
                emb_np = embeddings.detach().cpu().numpy().astype('float32')
                try:
                    faiss.normalize_L2(emb_np)
                except Exception:
                    pass
                d = emb_np.shape[1]
                index = faiss.IndexFlatIP(d)
                index.add(emb_np)
                app.session_state.faiss_index = index
        except Exception as e:
            # Optional; continue gracefully if FAISS not available
            print(f"Khởi tạo FAISS index thất bại: {e}")
    except Exception as e:
        print(f"Lỗi khởi tạo embeddings: {e}")


# Hàm mã hóa thống nhất: chỉ SBERT; trả torch.Tensor (N, d) hoặc (1, d)
def encode_question_embedding(inputs):
    try:
        if sbert_model is not None:
            if isinstance(inputs, list):
                embs = sbert_model.encode(inputs, batch_size=32, normalize_embeddings=True, convert_to_tensor=True)
                return embs  # (N, d)
            else:
                embs = sbert_model.encode([inputs], batch_size=32, normalize_embeddings=True, convert_to_tensor=True)
                return embs  # (1, d)
        else:
            return None
    except Exception:
        return None


# Xây dựng embeddings cho toàn bộ câu hỏi trong dữ liệu
def build_question_embeddings_and_maps(admissions_data_local):
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
    # Mã hóa theo batch (ưu tiên SBERT)
    embeddings = encode_question_embedding(question_texts)
    # Lưu map từ text -> item (ưu tiên danh sách nếu trùng)
    text_to_item = {}
    for t, it in zip(question_texts, question_items):
        if t in text_to_item:
            pass
        else:
            text_to_item[t] = it
    return question_texts, embeddings, text_to_item


# --- Adaptive routing helpers ---

def fuzzy_best_item(question_text: str):
    """Trả về (item_tốt_nhất, tỷ_lệ_fuzzy). Nếu không có, trả (None, 0)."""
    # Normalize both user question and dataset questions to unaccented lowercase
    user_q = normalize_and_unaccent(question_text)
    try:
        keys = list(KEYWORD_TO_ITEM_MAP.keys())
        if not keys:
            return None, 0.0
        # Prefer RapidFuzz if available
        if process is not None and fuzz is not None:
            match = process.extractOne(user_q, keys, scorer=fuzz.ratio)
            if not match:
                return None, 0.0
            best_key, score, _ = match
            return KEYWORD_TO_ITEM_MAP.get(best_key), float(score) / 100.0
        # Fallback: difflib scan
        best_item = None
        best_ratio = 0.0
        for k in keys:
            ratio = SequenceMatcher(None, user_q, k).ratio()
            if ratio > best_ratio:
                best_ratio = ratio
                best_item = KEYWORD_TO_ITEM_MAP.get(k)
        return best_item, float(best_ratio)
    except Exception:
        return None, 0.0


def retrieve_topk_embeddings(query_text: str, top_k: int = 10):
    """Trả về danh sách [(index, cosine)] giảm dần theo cosine, tối đa top_k. Nếu không sẵn sàng, trả []."""
    if sbert_model is None:
        return []
    # Fix: use hasattr instead of membership test on SimpleNamespace
    if (not hasattr(app.session_state, 'question_embeddings') or app.session_state.question_embeddings is None
            or not hasattr(app.session_state, 'question_texts') or len(app.session_state.question_texts) == 0):
        return []
    # Ưu tiên FAISS nếu có
    index = getattr(app.session_state, 'faiss_index', None)
    if index is not None:
        try:
            # Chuẩn hóa truy vấn và tìm top-k bằng FAISS (inner product)
            q_emb = encode_question_embedding(normalize_text(query_text))  # (1, d)
            q_norm = torch.nn.functional.normalize(q_emb, dim=1)
            q_np = q_norm.detach().cpu().numpy().astype('float32')
            k = min(top_k, app.session_state.question_embeddings.shape[0])
            D, I = index.search(q_np, k)
            sims = D[0].tolist();
            inds = I[0].tolist()
            return [(int(i), float(s)) for i, s in zip(inds, sims) if i >= 0]
        except Exception:
            pass
    # Ưu tiên KNN sklearn nếu có
    knn = getattr(app.session_state, 'knn_index', None)
    if knn is not None:
        try:
            q_emb = encode_question_embedding(normalize_text(query_text))  # (1, d)
            if q_emb is None:
                return []
            # embeddings were fit as normalized tensors; convert to numpy
            q_np = torch.nn.functional.normalize(q_emb, dim=1).detach().cpu().numpy()
            k = min(top_k, app.session_state.question_embeddings.shape[0])
            distances, indices = knn.kneighbors(q_np, n_neighbors=k, return_distance=True)
            inds = indices[0].tolist()
            dists = distances[0].tolist()
            # Convert cosine distances to cosine similarity
            sims = [1.0 - float(d) for d in dists]
            # Pair and sort by similarity desc (robustness)
            pairs = sorted([(int(i), float(s)) for i, s in zip(inds, sims)], key=lambda x: x[1], reverse=True)
            return pairs
        except Exception:
            pass
    try:
        # Fallback tính cosine bằng torch
        q_emb = encode_question_embedding(normalize_text(query_text))  # (1, d)
        q = torch.nn.functional.normalize(q_emb, dim=1)
        c = torch.nn.functional.normalize(app.session_state.question_embeddings, dim=1)
        sims = torch.mm(q, c.t()).squeeze(0)  # (N,)
        values, indices = torch.topk(sims, k=min(top_k, sims.shape[0]))
        return [(int(idx.item()), float(val.item())) for val, idx in zip(values, indices)]
    except Exception:
        return []


def rerank_with_cross_encoder(query_text: str, candidate_indices):
    """Dùng cross-encoder để chấm điểm (query, candidate_text) và trả về (best_idx, best_score)."""
    if cross_encoder_model is None or not candidate_indices:
        return None, 0.0
    try:
        # NEW: cap number of candidates for latency
        max_candidates = 8
        candidate_indices = list(candidate_indices)[:max_candidates]
        pairs = []
        qt = normalize_text(query_text)
        for idx in candidate_indices:
            cand_text = app.session_state.question_texts[idx]
            pairs.append((qt, normalize_text(cand_text)))
        scores = cross_encoder_model.predict(pairs)
        try:
            scores = scores.tolist()
        except Exception:
            pass
        if not scores:
            return None, 0.0
        best_pos = max(range(len(scores)), key=lambda i: scores[i])
        return candidate_indices[best_pos], float(scores[best_pos])
    except Exception:
        return None, 0.0


def find_answer_and_media(question):
    # Chuẩn hóa (bỏ dấu) cho các bước đối sánh từ khóa/chuỗi
    norm_question = normalize_and_unaccent(question)

    # 0) Đối sánh trực tiếp với dữ liệu gốc
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

    # 1) Bản đồ từ khóa trực tiếp
    tokens = norm_question.split()
    num_tokens = len(tokens)

    direct_item = KEYWORD_TO_ITEM_MAP.get(norm_question)
    if direct_item:
        answer = direct_item.get('answer', "Không có câu trả lời.")
        images = direct_item.get('images')
        captions = direct_item.get('captions')
        video_url = direct_item.get('video_url')
        if images and isinstance(images, str):
            images = [images]
        if video_url:
            return answer, "video", video_url
        if images:
            return answer, "image", (images, captions)
        return answer, "text", None

    # 1b) Prefix/contains fallback to catch partial phrases (e.g., 'tien ich' -> 'tien ich xung quanh truong')
    phrase_matches = []  # keep earlier variable name context
    contains_candidates = []
    nq = norm_question
    if len(nq) >= 3:
        # punctuation-stripped versions to improve partial matching
        def _pun(s: str) -> str:
            try:
                return re.sub(r"\W+", " ", s).strip()
            except Exception:
                return s

        def _wb_contains(hay: str, needle: str) -> bool:
            # word-boundary containment: sequence of tokens, not substring of a token
            if not hay or not needle:
                return False
            H = f" {hay.strip()} ".replace("  ", " ")
            N = f" {needle.strip()} ".replace("  ", " ")
            return N in H

        np = _pun(nq)
        STOP_TOKENS = COMMON_STOP_TOKENS if 'COMMON_STOP_TOKENS' in globals() else {"truong", "co", "cua", "ve", "la",
                                                                                    "nao", "gi", "cai", "nhung", "o",
                                                                                    "dau", "ai"}
        for key, it in KEYWORD_TO_ITEM_MAP.items():
            # Skip trivially short or stop-only keys
            if len(key) < 3:
                continue
            key_tokens = key.split()
            if key_tokens and all((t in STOP_TOKENS or len(t) <= 2) for t in key_tokens):
                continue
            # Direct boundary-based checks on normalized strings
            if _wb_contains(key, nq) or _wb_contains(nq, key):
                contains_candidates.append((key, it))
                continue
            # Also compare punctuation-stripped versions
            kp = _pun(key)
            if kp and np and (_wb_contains(kp, np) or _wb_contains(np, kp)):
                contains_candidates.append((key, it))
    if contains_candidates:
        best_key, best_item = max(contains_candidates, key=lambda x: len(x[0]))
        answer = best_item.get('answer', "Không có câu trả lời.")
        images = best_item.get('images')
        captions = best_item.get('captions')
        video_url = best_item.get('video_url')
        if images and isinstance(images, str):
            images = [images]
        if video_url:
            return answer, "video", video_url
        if images:
            return answer, "image", (images, captions)
        return answer, "text", None

    # 2) N-gram cụm từ dài nhất
    phrase_matches = []
    for length in range(num_tokens, 1, -1):
        for i in range(num_tokens - length + 1):
            phrase = ' '.join(tokens[i:i + length])
            item = KEYWORD_TO_ITEM_MAP.get(phrase)
            if item:
                phrase_matches.append((phrase, item, length))
    if phrase_matches:
        phrase_matches.sort(key=lambda x: (-x[2], -len(x[0])))
        _, best_item, _ = phrase_matches[0]
        answer = best_item.get('answer', "Không có câu trả lời.")
        images = best_item.get('images')
        captions = best_item.get('captions')
        video_url = best_item.get('video_url')
        if images and isinstance(images, str):
            images = [images]
        if video_url:
            return answer, "video", video_url
        if images:
            return answer, "image", (images, captions)
        return answer, "text", None

    # 3) Xử lý danh xưng 2 từ (thay/co + tên)
    if num_tokens == 2:
        phrase = ' '.join(tokens)
        item = KEYWORD_TO_ITEM_MAP.get(phrase)
        if item:
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
        honorifics = {"thay", "co"}
        t0, t1 = tokens[0], tokens[1]
        if t0 in honorifics:
            direct = KEYWORD_TO_ITEM_MAP.get(t1)
            if direct:
                answer = direct.get('answer', "Không có câu trả lời.")
                images = direct.get('images')
                captions = direct.get('captions')
                video_url = direct.get('video_url')
                if images and isinstance(images, str):
                    images = [images]
                if video_url:
                    return answer, "video", video_url
                if images:
                    return answer, "image", (images, captions)
                return answer, "text", None
            # Fuzzy match chỉ trên tên sau danh xưng
            fuzzy_res = fuzzy_match_question(t1, admissions_data, min_ratio=0.6)
            if fuzzy_res:
                answer, images, captions = fuzzy_res
                if images and isinstance(images, str):
                    images = [images]
                if images:
                    return answer, "image", (images, captions)
                return answer, "text", None
            # NEW: tìm key chứa cả danh xưng và tên (token) trong bản đồ từ khóa đã chuẩn hóa
            try:
                candidates = []
                for key, it in KEYWORD_TO_ITEM_MAP.items():
                    ktoks = key.split()
                    if ktoks and ktoks[0] in honorifics and t1 in ktoks:
                        candidates.append((key, it))
                if candidates:
                    # Ưu tiên key dài hơn (đủ họ tên) để chính xác hơn
                    best_key, best_item = max(candidates, key=lambda x: len(x[0]))
                    answer = best_item.get('answer', "Không có câu trả lời.")
                    images = best_item.get('images')
                    captions = best_item.get('captions')
                    video_url = best_item.get('video_url')
                    if images and isinstance(images, str):
                        images = [images]
                    if video_url:
                        return answer, "video", video_url
                    if images:
                        return answer, "image", (images, captions)
                    return answer, "text", None
            except Exception:
                pass

    # 3b) Truy vấn 1 từ (tên ngắn...)
    if num_tokens == 1:
        single = tokens[0]
        direct = KEYWORD_TO_ITEM_MAP.get(single)
        if direct:
            answer = direct.get('answer', "Không có câu trả lời.")
            images = direct.get('images')
            captions = direct.get('captions')
            video_url = direct.get('video_url')
            if images and isinstance(images, str):
                images = [images]
            if video_url:
                return answer, "video", video_url
            if images:
                return answer, "image", (images, captions)
            return answer, "text", None
        fuzzy_res = fuzzy_match_question(single, admissions_data, min_ratio=0.6)
        if fuzzy_res:
            answer, images, captions = fuzzy_res
            if images and isinstance(images, str):
                images = [images]
            if images:
                return answer, "image", (images, captions)
            return answer, "text", None

    # 4) Truy vấn dài: chọn span tốt nhất hoặc token phù hợp nhất
    matched_items = []
    matched_tokens = []
    if num_tokens > 2 and not phrase_matches:
        best_token_item = None
        best_token_length = 0
        for i in range(num_tokens):
            for j in range(i + 1, num_tokens + 1):
                phrase = ' '.join(tokens[i:j])
                item = KEYWORD_TO_ITEM_MAP.get(phrase)
                if item and (j - i) > best_token_length:
                    best_token_length = (j - i)
                    best_token_item = item
        if best_token_item:
            answer = best_token_item.get('answer', "Không có câu trả lời.")
            images = best_token_item.get('images')
            captions = best_token_item.get('captions')
            video_url = best_token_item.get('video_url')
            if images and isinstance(images, str):
                images = [images]
            if video_url:
                return answer, "video", video_url
            if images:
                return answer, "image", (images, captions)
            return answer, "text", None
        # NEW: ignore stopword/short tokens to avoid mapping "co" (có/cô), etc.
        STOP_TOKENS = COMMON_STOP_TOKENS if 'COMMON_STOP_TOKENS' in globals() else {"truong", "co", "cua", "ve", "la",
                                                                                    "nao", "gi", "cai", "cac", "nhung",
                                                                                    "o", "dau"}
        for token in tokens:
            if len(token) < 3 or token in STOP_TOKENS:
                continue
            item = KEYWORD_TO_ITEM_MAP.get(token)
            if item and item not in matched_items:
                matched_items.append(item)
                matched_tokens.append(token)
        if matched_items:
            best_item = None
            best_length = 0
            user_input = norm_question
            for idx, item in enumerate(matched_items):
                questions = item.get('question', [])
                if isinstance(questions, str):
                    questions = [questions]
                for q in questions:
                    norm_q = normalize_and_unaccent(q)
                    if norm_q in user_input and len(norm_q) > best_length:
                        best_length = len(norm_q)
                        best_item = item
            if not best_item:
                for idx, item in enumerate(matched_items):
                    questions = item.get('question', [])
                    if isinstance(questions, str):
                        questions = [questions]
                    for q in questions:
                        norm_q = normalize_and_unaccent(q)
                        if norm_q == matched_tokens[idx] and len(norm_q) > best_length:
                            best_length = len(norm_q)
                            best_item = item
            if not best_item:
                best_item = matched_items[0]
            answer = best_item.get('answer', "Không có câu trả lời.")
            images = best_item.get('images')
            captions = best_item.get('captions')
            video_url = best_item.get('video_url')
            if images and isinstance(images, str):
                images = [images]
            if video_url:
                return answer, "video", video_url
            if images:
                return answer, "image", (images, captions)
            return answer, "text", None

    # 5 & 6) Adaptive routing giữa fuzzy và embeddings (+ cross-encoder rerank nếu sẵn)
    # NEW: If no dataset keyword is present at all, stop here with default unknown instead of fuzzy/semantic guesses
    if not contains_dataset_keyword(question):
        return "Xin lỗi, tôi không có thông tin về nội dung này.", "text", None

    FUZZY_STRONG = 0.88
    EMBED_STRONG = 0.72
    FUZZY_MIN = 0.60
    EMBED_MIN = 0.60

    best_fuzzy_item, best_fuzzy_ratio = fuzzy_best_item(question)

    best_embed_index = None
    best_embed_sim = 0.0
    embed_candidates = []
    if hasattr(app.session_state, 'question_embeddings') and app.session_state.question_embeddings is not None and len(
            app.session_state.question_texts) > 0:
        embed_candidates = retrieve_topk_embeddings(question, top_k=10)
        if embed_candidates:
            best_embed_index, best_embed_sim = embed_candidates[0]
            # Cross-encoder rerank trên cùng danh sách
            candidate_indices = [idx for idx, _ in embed_candidates]
            ce_idx, ce_score = rerank_with_cross_encoder(question, candidate_indices)
            # Nếu cross-encoder hoạt động và điểm không kém, dùng nó
            if ce_idx is not None and ce_score >= EMBED_MIN and ce_score >= best_embed_sim:
                best_embed_index = ce_idx
                best_embed_sim = ce_score

    # Quyết định
    chosen_item = None
    if best_fuzzy_ratio >= FUZZY_STRONG and best_fuzzy_ratio >= (best_embed_sim + 0.10):
        chosen_item = best_fuzzy_item
    elif best_embed_index is not None and best_embed_sim >= EMBED_STRONG:
        matched_question = app.session_state.question_texts[best_embed_index]
        chosen_item = app.session_state.question_data_map.get(matched_question)
    else:
        # Nếu cả hai không mạnh, nhưng một trong hai vượt tối thiểu, chọn cái cao hơn
        if best_fuzzy_ratio >= FUZZY_MIN or (best_embed_index is not None and best_embed_sim >= EMBED_MIN):
            if (best_embed_index is not None and best_embed_sim >= EMBED_MIN) and (best_embed_sim >= best_fuzzy_ratio):
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


def fuzzy_match_question(question, admissions_data, min_ratio=0.6):
    # Normalize both sides for robust Vietnamese fuzzy matching
    user_q = normalize_and_unaccent(question)
    try:
        keys = list(KEYWORD_TO_ITEM_MAP.keys())
        if not keys:
            return None
        if process is not None and fuzz is not None:
            match = process.extractOne(user_q, keys, scorer=fuzz.ratio)
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
    except Exception:
        return None


# --- FLASK ENDPOINTS ---
@app.route('/ask', methods=['POST'])
def ask():
    data = request.get_json()
    # Accept either a textual question or a choice_id (sent when user clicks an id-based option)
    if not data or ('question' not in data and 'choice_id' not in data):
        return jsonify({"error": "Vui lòng cung cấp 'question' hoặc 'choice_id' trong JSON"}), 400
    # Ensure semantic resources exist even if warmup didn’t run yet (e.g., in some WSGI setups)
    if getattr(app.session_state, 'question_embeddings', None) is None or not getattr(app.session_state,
                                                                                      'question_texts', []):
        initialize_semantic_resources()

    # Handle choice_id (id-based option chosen from frontend)
    if 'choice_id' in data:
        try:
            cid = data.get('choice_id')
            # Expect integer index into admissions_data['questions'] when we emit options
            idx = int(cid)
            items = admissions_data.get('questions', [])
            if idx < 0 or idx >= len(items):
                return jsonify({"error": "Invalid choice_id"}), 400
            item = items[idx]
            # Build a single response for the chosen item
            ans = item.get('answer', "Xin lỗi, tôi chưa tìm thấy thông tin phù hợp.")
            media_type = "text"
            media_content = None
            images = item.get('images')
            captions = item.get('captions')
            video_url = item.get('video_url')
            if images and isinstance(images, str):
                images = [images]
            if video_url:
                media_type = "video";
                media_content = video_url
            elif images:
                media_type = "image";
                media_content = (images, captions)
            # Optionally append to session history
            session_id = data.get('session_id', None)
            if session_id:
                append_history(session_id, 'user', f"(chose option) {idx}")
                append_history(session_id, 'bot', ans)
            return jsonify([{"text": ans, "media_type": media_type, "media_content": media_content}]), 200
        except ValueError:
            return jsonify({"error": "choice_id must be an integer index"}), 400
        except Exception as _e:
            return jsonify({"error": "Lỗi khi xử lý lựa chọn"}), 500

    question = data['question']
    session_id = data.get('session_id', None)

    # Append user turn to history
    if session_id:
        append_history(session_id, 'user', question)
        # Augment question with short context when appropriate
        question_for_answer = augment_with_context(session_id, question)
    else:
        question_for_answer = question

    # Detect UI/button-originated requests and avoid calling GPT for them.
    # Frontend can pass `via_button: true` or `trigger: 'button'` or `source: 'ui'` to indicate a UI button action.
    is_ui_button = False
    try:
        if isinstance(data, dict):
            if data.get('via_button') or data.get('trigger') in ('button', 'ui', 'click') or data.get('source') in ('ui', 'button'):
                is_ui_button = True
    except Exception:
        is_ui_button = False

    responses = get_answer(question_for_answer, skip_gpt=is_ui_button)

    # Store bot turn (text only, first response) for lightweight history
    try:
        if session_id and responses and isinstance(responses, list):
            first_text = responses[0].get('text') if isinstance(responses[0], dict) else None
            if first_text:
                append_history(session_id, 'bot', first_text)
    except Exception:
        pass

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
        # pass-through action for frontend UI hints
        if isinstance(resp, dict) and "action" in resp:
            entry["action"] = resp["action"]
        if resp["media_type"] == "video" and resp["media_content"]:
            entry["video_url"] = resp["media_content"]
        elif resp["media_type"] == "image" and resp["media_content"]:
            images, captions = resp["media_content"]
            entry["images"] = [f"/images/{os.path.basename(img)}" for img in images if
                               isinstance(img, str) and img.strip()]
            entry["captions"] = captions if captions else []
        result.append(entry)
    # Defensive: remove duplicate responses by text to avoid client showing repeated answers
    try:
        unique = []
        seen = set()
        for r in result:
            t = (r.get('text') or '').strip()
            if not t:
                # include empty/unnamed entries once
                key = '__empty__'
            else:
                key = t
            if key in seen:
                continue
            seen.add(key)
            unique.append(r)
        return jsonify(unique), 200
    except Exception:
        return jsonify(result), 200


@app.route('/images/<path:filename>')
def serve_image(filename):
    return send_from_directory('images', filename)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/status', methods=['GET'])
def status():
    try:
        n_items = len(admissions_data.get('questions', []))
        n_texts = len(getattr(app.session_state, 'question_texts', []) or [])
        has_emb = app.session_state.question_embeddings is not None
        has_knn = getattr(app.session_state, 'knn_index', None) is not None
        has_faiss = getattr(app.session_state, 'faiss_index', None) is not None
        has_ce = cross_encoder_model is not None
        has_sbert = sbert_model is not None
        return jsonify({
            "items": n_items,
            "questions_indexed": n_texts,
            "embeddings": bool(has_emb),
            "faiss_index": bool(has_faiss),
            "knn_index": bool(has_knn),
            "cross_encoder": bool(has_ce),
            "sbert": bool(has_sbert)
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


def make_clarifying_question(user_query: str, candidates: list) -> dict:
    """Sinh câu hỏi gợi ý rule-based khi có nhiều ứng viên phù hợp."""
    try:
        options = []
        for c in candidates:
            q = c.get("question", [])
            if isinstance(q, str):
                q = [q]
            if not q:
                continue
            label = q[0]
            # rút gọn nhãn cho gọn gàng
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
        # Use the frontend-compatible action name expected by templates/index.html
        # Previous value 'clarify_options' wasn't handled by the UI, causing the client
        # to ignore the provided options and ask again. The UI supports 'clarification'
        # with an 'options' list (backwards-compatible), so return that.
        # Prefer id-based options so frontend sends choice_id (more robust than free-text clicks)
        opts_objs = []
        # Map each candidate back to its index in admissions_data for id-based choices.
        # Use normalized comparison of the candidate's primary question to be robust
        data_questions = admissions_data.get('questions', [])
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
                # fallback: emit label-only option (frontend will send label string)
                opts_objs.append({"id": lab, "label": lab})
            else:
                opts_objs.append({"id": idx, "label": lab})
        return {"text": question, "media_type": "text", "media_content": None, "action": "clarification",
                "options": opts_objs}
    except Exception as e:
        return {"text": "Xin lỗi, tôi chưa tìm thấy thông tin phù hợp.", "media_type": "text", "media_content": None}


if __name__ == "__main__":
    # Warm up semantic resources on startup to reduce first-request latency
    try:
        initialize_semantic_resources()
    except Exception as _e:
        # Still start the server even if warmup fails
        pass
    app.run(host='0.0.0.0', port=8080)
