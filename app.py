# Copyright (c) [2025] [Nguyễn Minh Tấn Phúc]. Bảo lưu mọi quyền.
# Nguồn: https://tlmchattest.streamlit.app/
import json
import os
import re
import torch
import unicodedata
from difflib import SequenceMatcher
from flask import Flask, request, jsonify, send_from_directory

# --- KHỞI TẠO FLASK APP ---
app = Flask(__name__)

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
        r'^toi\s*muon\s*biet\s*',  # new: handle sentences without "về"
        r'^toi\s*muon\s*hoi\s*',   # new: handle "tôi muốn hỏi"
        r'^cho\s*t\s*oi\s*biet\s*ve\s*',
        r'^cho\s*toi\s*biet\s*ve\s*',
        r'^thong\s*tin\s*ve\s*',
        r'^gioi\s*thieu\s*ve\s*',
        r'^ve\s*'
    ]
    for pat in leadins:
        norm = re.sub(pat, '', norm)
    # Trim trailing filler words like "của"/"về"
    norm = re.sub(r'\b(cua|ve)\s*$', '', norm).strip()
    # Also trim trailing "cua truong ..." (generic school tail) to expose core keyword
    norm = re.sub(r'\bcua\s*(truong|thpt|trg|truong\s*thpt|trung\s*hoc\s*pho\s*thong)\b.*$', '', norm).strip()
    return norm

# Tách ý nhỏ cho câu hỏi nhiều ý, không dấu
def split_subquestions(text):
    norm_text = normalize_and_unaccent(text)
    # Chỉ tách theo các liên từ rõ ràng, dùng non-capturing group để không giữ lại từ nối
    conjunctions = [
        r'va', r'và', r'hoac', r'hay',
        r'voi', r'với',
        r'cung', r'cùng', r'cung\s*voi', r'cùng\s*với',
        r'roi', r'rồi', r'sau\s*do'
    ]
    pattern = r'[;,]|\b(?:' + '|'.join(conjunctions) + r')\b'
    parts = re.split(pattern, norm_text)
    subqs = [p.strip() for p in parts if p and len(p.strip()) > 2]
    return subqs

def get_answer(question):
    norm_question = normalize_text(question)
    # Only return early for exact question matches (collect all duplicates)
    norm_unaccent_question = normalize_and_unaccent(question)
    exact_matches = []
    for item in admissions_data.get('questions', []):
        questions = item.get('question', [])
        if isinstance(questions, str):
            questions = [questions]
        for q in questions:
            if normalize_and_unaccent(q) == norm_unaccent_question:
                if item not in exact_matches:
                    exact_matches.append(item)
                break  # avoid adding same item multiple times
    if exact_matches:
        responses = []
        for item in exact_matches:
            ans = item.get('answer', "Xin lỗi, tôi chưa tìm thấy thông tin phù hợp.")
            media_type = "text"
            media_content = None
            images = item.get('images')
            captions = item.get('captions')
            video_url = item.get('video_url')
            if images and isinstance(images, str):
                images = [images]
            if video_url:
                media_type = "video"; media_content = video_url
            elif images:
                media_type = "image"; media_content = (images, captions)
            responses.append({"text": ans, "media_type": media_type, "media_content": media_content})
        return responses

    # Fallback for "hiệu trưởng" keyword
    if "hiệu trưởng" in norm_question:
        hardcoded_response = "Bạn muốn biết về hiệu trưởng hiện tại hay hiệu trưởng qua từng thời kỳ?"
        return [{"text": hardcoded_response, "media_type": "text", "media_content": None}]

    SCHOOL_NAME_VARIANTS = [
        "trường thpt", "thpt", "trường trung học phổ thông", "trung học phổ thông",
        "ten lơ men", "ten lơ man", "ten-lơ-man", "ten-lơ-men", "trường cấp 3", "cấp 3", "cấp ba", "trường cấp ba",
        "ernst thälmann", "ernst thalmann", "trường công lập", "công lập", "tlm", "t.l.m", "t l m",
        "trường ten lơ man", "trường ten lơ men", "truong thpt", "truong trung hoc pho thong", "truong cap 3",
        "truong cap ba", "truong cong lap", "truong ten lo man", "truong ten lo men", "trường ernst",
        "trường ernst thälmann", "trường ernst thalmann", "ernst", "trường tlm", "trường t.l.m", "trường t l m",
        "school", "high school", "secondary school", "tenlo man", "tenlo men", "tenloman", "tenlomen",
        "trường tenlo man", "trường tenlo men", "trường tenloman", "trường tenlomen"
    ]
    school_pattern = r"(" + r"|".join(
        [re.escape(variant).replace(" ", r"\\s*") for variant in SCHOOL_NAME_VARIANTS]) + r")"
    if re.fullmatch(rf"(\s*{school_pattern}\s*)+", norm_question, flags=re.IGNORECASE):
        ans, media_type, media_content = find_answer_and_media(norm_question)
        return [{"text": ans, "media_type": media_type, "media_content": media_content}]

    core_question = re.sub(school_pattern, "", norm_question, flags=re.IGNORECASE).strip()
    if not core_question or core_question in ["", "về", "của"]:
        ans, media_type, media_content = find_answer_and_media(norm_question)
        return [{"text": ans, "media_type": media_type, "media_content": media_content}]

    # TÁCH Ý NHỎ
    core_question = strip_leadin_phrases(core_question)
    sub_questions = split_subquestions(core_question)
    # If single sub-question, try multi-hit return via map before falling back
    if len(sub_questions) <= 1:
        norm_core = normalize_and_unaccent(core_question)
        multi = KEYWORD_TO_ITEMS_MAP.get(norm_core, [])
        if len(multi) > 1:
            results = []
            for item in multi:
                ans = item.get('answer', "Không có câu trả lời.")
                images = item.get('images')
                captions = item.get('captions')
                video_url = item.get('video_url')
                media_type = "text"; media_content = None
                if images and isinstance(images, str):
                    images = [images]
                if video_url:
                    media_type = "video"; media_content = video_url
                elif images:
                    media_type = "image"; media_content = (images, captions)
                results.append({"text": ans, "media_type": media_type, "media_content": media_content})
            return results
        # NEW: Search substrings for duplicate keyword mappings (e.g., "hieu pho")
        tokens = norm_core.split()
        for length in range(len(tokens), 1, -1):
            for i in range(len(tokens) - length + 1):
                phrase = ' '.join(tokens[i:i+length])
                multi_sub = KEYWORD_TO_ITEMS_MAP.get(phrase, [])
                if len(multi_sub) > 1:
                    results = []
                    for item in multi_sub:
                        ans = item.get('answer', "Không có câu trả lời.")
                        images = item.get('images')
                        captions = item.get('captions')
                        video_url = item.get('video_url')
                        media_type = "text"; media_content = None
                        if images and isinstance(images, str):
                            images = [images]
                        if video_url:
                            media_type = "video"; media_content = video_url
                        elif images:
                            media_type = "image"; media_content = (images, captions)
                        results.append({"text": ans, "media_type": media_type, "media_content": media_content})
                    return results
        # Prefer original text for semantic/fuzzy, then fallback to normalized
        ans, media_type, media_content = find_answer_and_media(core_question)
        if ans and ans.strip() and ans != "Xin lỗi, tôi chưa tìm thấy thông tin phù hợp.":
            return [{"text": ans, "media_type": media_type, "media_content": media_content}]
        ans, media_type, media_content = find_answer_and_media(norm_core)
        return [{"text": ans, "media_type": media_type, "media_content": media_content}]

    # Nếu có nhiều ý nhỏ, trả về từng câu trả lời
    results = []
    for subq in sub_questions:
        # Prefer original sub-question first (retains dấu cho embedding/fuzzy)
        ans, media_type, media_content = find_answer_and_media(subq)
        if (not ans or not ans.strip() or ans == "Xin lỗi, tôi chưa tìm thấy thông tin phù hợp."):
            sub_norm = normalize_and_unaccent(subq)
            ans, media_type, media_content = find_answer_and_media(sub_norm)
        if ans and ans.strip():
            results.append({"text": ans, "media_type": media_type, "media_content": media_content})
    if results:
        return results
    return [{"text": "Xin lỗi, tôi chưa tìm thấy thông tin phù hợp cho các ý bạn hỏi.", "media_type": "text", "media_content": None}]

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

try:
    with open(os.path.join(os.path.dirname(__file__), 'admissions_data.json'), 'r', encoding='utf-8') as f:
        admissions_data = json.load(f)
except Exception as e:
    print(f"Lỗi khi tải admissions_data.json: {e}")
    admissions_data = {"questions": []}

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

# Lazy-load cross-encoder cho rerank (đa ngôn ngữ dựa trên XLM-R)
def load_cross_encoder_model():
    try:
        from sentence_transformers import CrossEncoder
        return CrossEncoder("cross-encoder/stsb-xlm-r-multilingual")
    except Exception:
        return None

cross_encoder_model = load_cross_encoder_model()

# Lazy-load SBERT/E5 bi-encoder (ưu tiên cho tiếng Việt)
def load_sentence_transformer_model():
    try:
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer("paraphrase-multilingual-mpnet-base-v2")
    except Exception:
        return None

sbert_model = load_sentence_transformer_model()

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
    best_item = None
    best_ratio = 0.0
    for item in admissions_data.get('questions', []):
        questions = item.get('question', [])
        if isinstance(questions, str):
            questions = [questions]
        for q in questions:
            ratio = SequenceMatcher(None, question_text, q).ratio()
            if ratio > best_ratio:
                best_ratio = ratio
                best_item = item
    return best_item, best_ratio

def retrieve_topk_embeddings(query_text: str, top_k: int = 10):
    """Trả về danh sách [(index, cosine)] giảm dần theo cosine, tối đa top_k. Nếu không sẵn sàng, trả []."""
    if sbert_model is None:
        return []
    if 'question_embeddings' not in app.session_state or app.session_state.question_embeddings is None or len(app.session_state.question_texts) == 0:
        return []
    # Ưu tiên FAISS nếu có
    index = app.session_state.get('faiss_index')
    if index is not None:
        try:
            # Chuẩn hóa truy vấn và tìm top-k bằng FAISS (inner product)
            q_emb = encode_question_embedding(normalize_text(query_text))  # (1, d)
            q_norm = torch.nn.functional.normalize(q_emb, dim=1)
            q_np = q_norm.detach().cpu().numpy().astype('float32')
            k = min(top_k, app.session_state.question_embeddings.shape[0])
            D, I = index.search(q_np, k)
            sims = D[0].tolist(); inds = I[0].tolist()
            return [(int(i), float(s)) for i, s in zip(inds, sims) if i >= 0]
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
    contains_candidates = []
    nq = norm_question
    if len(nq) >= 3:
        # punctuation-stripped versions to improve partial matching
        def _pun(s: str) -> str:
            try:
                return re.sub(r"\W+", " ", s).strip()
            except Exception:
                return s
        np = _pun(nq)
        for key, it in KEYWORD_TO_ITEM_MAP.items():
            if key.startswith(nq) or nq.startswith(key) or (nq in key) or (key in nq):
                contains_candidates.append((key, it))
                continue
            kp = _pun(key)
            if kp and np and (kp.startswith(np) or np.startswith(kp) or (np in kp) or (kp in np)):
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
            phrase = ' '.join(tokens[i:i+length])
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
            fuzzy_res = fuzzy_match_question(t1, admissions_data, min_ratio=0.6)
            if fuzzy_res:
                answer, images, captions = fuzzy_res
                if images and isinstance(images, str):
                    images = [images]
                if images:
                    return answer, "image", (images, captions)
                return answer, "text", None

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
            for j in range(i+1, num_tokens+1):
                phrase = ' '.join(tokens[i:j])
                item = KEYWORD_TO_ITEM_MAP.get(phrase)
                if item and (j-i) > best_token_length:
                    best_token_length = (j-i)
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
        for token in tokens:
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
    FUZZY_STRONG = 0.88
    EMBED_STRONG = 0.72
    FUZZY_MIN = 0.60
    EMBED_MIN = 0.60

    best_fuzzy_item, best_fuzzy_ratio = fuzzy_best_item(question)

    best_embed_index = None
    best_embed_sim = 0.0
    embed_candidates = []
    if hasattr(app.session_state, 'question_embeddings') and app.session_state.question_embeddings is not None and len(app.session_state.question_texts) > 0:
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
    best_match = None
    best_ratio = 0
    for item in admissions_data.get('questions', []):
        questions = item.get('question', [])
        if isinstance(questions, str):
            questions = [questions]
        for q in questions:
            ratio = SequenceMatcher(None, question, q).ratio()
            if ratio > best_ratio and ratio >= min_ratio:
                best_ratio = ratio
                best_match = item
    if best_match:
        answer = best_match.get('answer', "Không có câu trả lời.")
        images = best_match.get('images')
        captions = best_match.get('captions')
        return answer, images, captions
    return None

# --- FLASK ENDPOINTS ---
@app.route('/ask', methods=['POST'])
def ask():
    data = request.get_json()
    if not data or 'question' not in data:
        return jsonify({"error": "Vui lòng cung cấp câu hỏi trong JSON (key: 'question')"}), 400
    question = data['question']
    responses = get_answer(question)
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
        if resp["media_type"] == "video" and resp["media_content"]:
            entry["video_url"] = resp["media_content"]
        elif resp["media_type"] == "image" and resp["media_content"]:
            images, captions = resp["media_content"]
            entry["images"] = [f"/images/{os.path.basename(img)}" for img in images if isinstance(img, str) and img.strip()]
            entry["captions"] = captions if captions else []
        result.append(entry)
    return jsonify(result), 200

@app.route('/images/<path:filename>')
def serve_image(filename):
    return send_from_directory('images', filename)

@app.route('/')
def index():
    return "Chatbot Tuyển sinh Flask API. Sử dụng endpoint /ask để hỏi."

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080)