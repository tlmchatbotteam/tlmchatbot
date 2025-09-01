# Copyright (c) [2025] [Nguyễn Minh Tấn Phúc]. Bảo lưu mọi quyền.
# Nguồn: https://tlmchattest.streamlit.app/
import json
import streamlit as st
import os
import re
import torch
import unicodedata
import py_vncorenlp
from transformers import AutoTokenizer, AutoModel
from difflib import SequenceMatcher

# --- CẤU HÌNH VÀ TẢI DỮ LIỆU ---
try:
    with open(os.path.join(os.path.dirname(__file__), 'admissions_data.json'), 'r', encoding='utf-8') as f:
        admissions_data = json.load(f)
except Exception as e:
    st.error(f"Lỗi khi tải admissions_data.json: {e}")
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
        'v���'
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
        r'^cho\s*t\s*oi\s*biet\s*ve\s*',
        r'^cho\s*toi\s*biet\s*ve\s*',
        r'^thong\s*tin\s*ve\s*',
        r'^gioi\s*thieu\s*ve\s*',
        r'^ve\s*'
    ]
    for pat in leadins:
        norm = re.sub(pat, '', norm)
    return norm.strip()

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
        "trường ernst thälmann", "trường ernst thalmann", "ernst", "trư���ng tlm", "trường t.l.m", "trường t l m",
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
        # Otherwise, return best single match
        ans, media_type, media_content = find_answer_and_media(core_question)
        return [{"text": ans, "media_type": media_type, "media_content": media_content}]

    # Nếu có nhiều ý nhỏ, trả về từng câu trả lời
    results = []
    for subq in sub_questions:
        ans, media_type, media_content = find_answer_and_media(subq)
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

# Khởi tạo VnCoreNLP cho tách từ (dùng trong split_sticky_words)
@st.cache_resource
def load_vncorenlp_model():
    return py_vncorenlp.VnCoreNLP(save_dir=os.path.join(os.path.dirname(__file__), 'vncorenlp'))

vncorenlp_model = load_vncorenlp_model()


def split_sticky_words(text):
    # Sử dụng VnCoreNLP để tách từ
    segments = vncorenlp_model.word_segment(text)
    return ' '.join(segments)


# --- CẤU HÌNH VÀ TẢI DỮ LIỆU ---
device = "cuda" if torch.cuda.is_available() else "cpu"

try:
    with open(os.path.join(os.path.dirname(__file__), 'admissions_data.json'), 'r', encoding='utf-8') as f:
        admissions_data = json.load(f)
except Exception as e:
    st.error(f"Lỗi khi tải admissions_data.json: {e}")
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

# Tải mô hình XLM-RoBERTa
@st.cache_resource
def load_xlm_roberta_model():
    tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
    model = AutoModel.from_pretrained("xlm-roberta-base")
    return tokenizer, model

try:
    tokenizer, xlm_roberta_model = load_xlm_roberta_model()
except Exception as e:
    st.error(f"Lỗi khi tải mô hình XLM-RoBERTa: {e}")
    tokenizer, xlm_roberta_model = None, None

# Hàm mã hóa câu hỏi bằng XLM-RoBERTa
def encode_question_xlm_roberta(question):
    inputs = tokenizer(question, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = xlm_roberta_model(**inputs)
    return outputs.last_hidden_state.mean(dim=1)  # Sử dụng trung bình các vector ẩn

# Xây dựng embeddings cho toàn bộ câu hỏi trong dữ liệu
@st.cache_resource
def build_question_embeddings_and_maps(admissions_data_local):
    question_texts = []
    question_items = []
    for item in admissions_data_local.get('questions', []):
        questions = item.get('question', [])
        if isinstance(questions, str):
            questions = [questions]
        for q in questions:
            # Dùng văn bản gốc (giữ dấu) nhưng đã lower/trim để mô hình ngữ nghĩa hiểu tốt hơn
            qn_embed = normalize_text(q)
            if len(qn_embed) < 2:
                continue
            question_texts.append(qn_embed)
            question_items.append(item)
    if not question_texts:
        return [], None, {}
    # Mã hóa theo batch
    embeddings = encode_question_xlm_roberta(question_texts)
    # Lưu map từ text -> item (ưu tiên danh sách nếu trùng)
    text_to_item = {}
    for t, it in zip(question_texts, question_items):
        if t in text_to_item:
            pass
        else:
            text_to_item[t] = it
    return question_texts, embeddings, text_to_item


def ensure_embeddings_ready():
    if tokenizer is None or xlm_roberta_model is None:
        return
    if 'question_embeddings' not in st.session_state or st.session_state.question_embeddings is None or len(st.session_state.question_texts) == 0:
        q_texts, q_embeds, q_map = build_question_embeddings_and_maps(admissions_data)
        st.session_state.question_texts = q_texts
        st.session_state.question_embeddings = q_embeds
        st.session_state.question_data_map = q_map


def find_best_match_with_embeddings(query_embedding, corpus_embeddings, min_similarity: float = 0.6):
    """Trả về index có cosine similarity cao nhất nếu >= ngưỡng, ngược lại None."""
    try:
        if query_embedding is None or corpus_embeddings is None:
            return None
        if corpus_embeddings.ndim != 2:
            return None
        # Chuẩn hóa L2 rồi tính tích ma trận để ra cosine
        q = torch.nn.functional.normalize(query_embedding, dim=1)  # (1, d)
        c = torch.nn.functional.normalize(corpus_embeddings, dim=1)  # (N, d)
        sims = torch.mm(q, c.t()).squeeze(0)  # (N,)
        best_val, best_idx = torch.max(sims, dim=0)
        if float(best_val.item()) >= min_similarity:
            return int(best_idx.item())
        return None
    except Exception:
        return None

# Khớp và truy xuất câu trả lời (có lớp đồng nghĩa và ngữ nghĩa)

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

    # 5) Fuzzy cuối cùng
    fuzzy_result = fuzzy_match_question(question, admissions_data, min_ratio=0.6)
    if fuzzy_result:
        answer, images, captions = fuzzy_result
        if images and isinstance(images, str):
            images = [images]
        if images:
            return answer, "image", (images, captions)
        return answer, "text", None
    fuzzy_result = fuzzy_match_question(norm_question, admissions_data, min_ratio=0.6)
    if fuzzy_result:
        answer, images, captions = fuzzy_result
        if images and isinstance(images, str):
            images = [images]
        if images:
            return answer, "image", (images, captions)
        return answer, "text", None

    # 6) Ngữ nghĩa (embeddings) nếu sẵn sàng
    if tokenizer is not None and xlm_roberta_model is not None and hasattr(st.session_state, 'question_embeddings') and st.session_state.question_embeddings is not None and len(st.session_state.question_texts) > 0:
        try:
            question_embedding = encode_question_xlm_roberta(question)
            best_index = find_best_match_with_embeddings(
                question_embedding, st.session_state.question_embeddings
            )
            if best_index is not None:
                matched_question = st.session_state.question_texts[best_index]
                matched_item = st.session_state.question_data_map.get(matched_question)
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
        except Exception:
            pass

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

# --- GIAO DIỆN STREAMLIT ---

def main():
    st.title("Chatbot Tư vấn Tuyển sinh")
    st.markdown("Hỏi về thông tin tuyển sinh và xem hình ảnh hoặc video liên quan!")

    if 'messages' not in st.session_state:
        st.session_state.messages = []

    if 'question_embeddings' not in st.session_state or st.session_state.question_embeddings is None:
        st.session_state.question_embeddings = []

    if 'question_texts' not in st.session_state or not st.session_state.question_texts:
        st.session_state.question_texts = []

    if 'question_data_map' not in st.session_state or not st.session_state.question_data_map:
        st.session_state.question_data_map = {}

    if xlm_roberta_model is None:
        st.error("Mô hình XLM-RoBERTa chưa được tải. Vui lòng kiểm tra lại cấu hình.")
    else:
        # Chuẩn bị embeddings để hỗ trợ tìm kiếm ngữ nghĩa (đồng nghĩa chưa thấy trong kho)
        ensure_embeddings_ready()

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if "text" in message:
                st.markdown(message["text"])
            if "video" in message:
                st.video(message["video"])
            if "images" in message:
                valid_images_paths = [img_path for img_path in message["images"] if
                                      isinstance(img_path, str) and os.path.exists(img_path) and img_path.strip() != ""]
                if valid_images_paths:
                    num_cols = min(len(valid_images_paths), 3)
                    cols = st.columns(num_cols)
                    captions = message.get("captions", [])
                    if not isinstance(captions, list):
                        captions = [captions] if captions else [f"Ảnh {i + 1}" for i in range(len(valid_images_paths))]
                    captions = captions[:len(valid_images_paths)]
                    for i, img_path in enumerate(valid_images_paths):
                        with cols[i % num_cols]:
                            st.image(img_path,
                                     caption=captions[i] if i < len(captions) else f"Ảnh {i + 1}")

    if prompt := st.chat_input("Câu hỏi của bạn:"):
        # Pass raw prompt to get_answer, let get_answer handle normalization
        st.session_state.messages.append({"role": "user", "text": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.spinner("Đang xử lý câu hỏi..."):
            responses = get_answer(prompt)

        with st.chat_message("assistant"):
            for resp in responses:
                st.markdown(resp["text"])
                if resp["media_type"] == "video" and resp["media_content"]:
                    st.video(resp["media_content"])
                elif resp["media_type"] == "image" and resp["media_content"]:
                    images, captions = resp["media_content"]
                    if images:
                        valid_images_paths = []
                        for img_path in images:
                            if isinstance(img_path, str) and img_path.strip():
                                abs_img_path = os.path.join(os.path.dirname(__file__), img_path)
                                if os.path.isfile(abs_img_path):  # Ensure it's a valid file
                                    valid_images_paths.append(abs_img_path)
                                else:
                                    st.warning(f"Image not found or invalid: {img_path}")

                        if valid_images_paths:
                            num_cols = min(len(valid_images_paths), 3)
                            cols = st.columns(num_cols)
                            if not isinstance(captions, list):
                                captions = [captions] if captions else [f"Image {i + 1}" for i in range(len(valid_images_paths))]
                            captions = captions[:len(valid_images_paths)]
                            for i, abs_img_path in enumerate(valid_images_paths):
                                with cols[i % num_cols]:
                                    st.image(abs_img_path, caption=captions[i] if i < len(captions) else f"Image {i + 1}")
            st.session_state.messages.append({"role": "assistant", "text": '\n'.join([r["text"] for r in responses])})

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Giới thiệu về trường", key="suggested_question_button"):
            hardcoded_response = "Tôi xin giới thiệu bạn video về trường."
            hardcoded_video_url = "https://www.youtube.com/watch?v=HzvZVAvBkto"
            st.session_state.messages.append({"role": "user", "text": "Giới thiệu về trường"})
            with st.chat_message("assistant"):
                st.markdown(hardcoded_response)
                st.video(hardcoded_video_url)
            st.session_state.messages.append(
                {"role": "assistant", "text": hardcoded_response, "video": hardcoded_video_url})
            st.rerun()

    with col2:
        if st.button("Xóa lịch sử trò chuyện", key="clear_history_button"):
            st.session_state.messages = []
            st.rerun()

    # Add a new button for "Hiệu trưởng"
    col3, col4 = st.columns(2)
    with col3:
        if st.button("Hiệu trưởng", key="principal_question_button"):
            hardcoded_response = "Bạn muốn biết về hiệu trưởng hiện tại hay hiệu trưởng qua từng thời kỳ?"
            st.session_state.messages.append({"role": "user", "text": "Hiệu trưởng"})
            with st.chat_message("assistant"):
                st.markdown(hardcoded_response)
            st.session_state.messages.append({"role": "assistant", "text": hardcoded_response})
            st.rerun()


if __name__ == "__main__":
    main()
