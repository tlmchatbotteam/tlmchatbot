# Copyright (c) [2025] [Nguyễn Minh Tấn Phúc]. Bảo lưu mọi quyền.
# Nguồn: https://tlmchattest.streamlit.app/
import json
import streamlit as st
import os
import re
from sentence_transformers import SentenceTransformer, util
import torch
from pyvi import ViTokenizer
import unicodedata
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import py_vncorenlp
import difflib

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
    variants = set([keyword])
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

# Tách ý nhỏ cho câu hỏi nhiều ý, không dấu

def split_subquestions(text):
    norm_text = normalize_and_unaccent(text)
    # Chỉ tách theo các liên từ rõ ràng, không tách theo trùng từ khóa
    conjunctions = [r'và', r'hay', r'hoặc', r'va', r'hoac']
    pattern = r'[;,]|\b(' + '|'.join(conjunctions) + r')\b'
    subqs = re.split(pattern, norm_text)
    subqs = [q.strip() for q in subqs if q and len(q.strip()) > 2]
    return subqs


def get_answer(question):
    norm_question = normalize_text(question)
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
    sub_questions = split_subquestions(core_question)
    if len(sub_questions) <= 1:
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

try:
    @st.cache_resource
    def load_sentence_transformer_model():
        return SentenceTransformer('bkai-foundation-models/vietnamese-bi-encoder', device=device)


    model = load_sentence_transformer_model()
except Exception as e:
    st.error(f"Lỗi khi tải mô hình SentenceTransformer: {e}")
    model = None

# Tải và cache TF-IDF vectorizer
try:
    @st.cache_resource
    def load_tfidf_vectorizer():
        return TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}')


    tfidf_vectorizer = load_tfidf_vectorizer()
except Exception as e:
    st.error(f"Lỗi khi tải TfidfVectorizer: {e}")
    tfidf_vectorizer = None

# Initialize VnCoreNLP for word segmentation
@st.cache_resource
def load_vncorenlp_model():
    return py_vncorenlp.VnCoreNLP(save_dir=os.path.join(os.path.dirname(__file__), 'vncorenlp'))


vncorenlp_model = load_vncorenlp_model()

# Cập nhật quy trình mã hóa dữ liệu
if model and tfidf_vectorizer and 'question_embeddings' not in st.session_state:
    st.session_state.question_texts = []
    unaccented_questions_for_encoding = []
    st.session_state.question_data_map = {}

    for item in admissions_data.get('questions', []):
        questions = item.get('question', [])
        if not isinstance(questions, list):
            questions = [questions]
        for q in questions:
            if not isinstance(q, str) or not q.strip():
                continue
            norm_q = normalize_text(q)
            split_q = split_sticky_words(norm_q)
            tokenized_q = ViTokenizer.tokenize(split_q)
            unaccented_q = remove_vietnamese_accents(tokenized_q)
            unaccented_questions_for_encoding.append(unaccented_q)
            st.session_state.question_texts.append(tokenized_q)
            st.session_state.question_data_map[tokenized_q] = item

    if unaccented_questions_for_encoding:
        st.session_state.question_embeddings = model.encode(unaccented_questions_for_encoding, convert_to_tensor=True)
        st.session_state.tfidf_matrix = tfidf_vectorizer.fit_transform(unaccented_questions_for_encoding)
    else:
        st.session_state.question_embeddings = None
        st.session_state.tfidf_matrix = None


# --- HÀM TÌM KIẾM CÂU TRẢ LỜI (HYBRID SEARCH) ---
def fuzzy_match_question(user_question, admissions_data, min_ratio=0.7):
    user_question_norm = normalize_and_unaccent(user_question)
    best_ratio = 0
    best_item = None
    for item in admissions_data.get('questions', []):
        questions = item.get('question', [])
        if isinstance(questions, str):
            questions = [questions]
        for q in questions:
            q_norm = normalize_and_unaccent(q)
            ratio = difflib.SequenceMatcher(None, user_question_norm, q_norm).ratio()
            if user_question_norm in q_norm or q_norm in user_question_norm:
                ratio += 0.2
            if ratio > best_ratio:
                best_ratio = ratio
                best_item = item
    if best_item and best_ratio >= min_ratio:
        return best_item.get('answer'), best_item.get('images'), best_item.get('captions')
    return None

# Tạo từ điển tra cứu cho so khớp từ khóa trực tiếp
KEYWORD_ANSWER_MAP = {}
for item in admissions_data.get('questions', []):
    questions = item.get('question', [])
    if isinstance(questions, str):
        questions = [questions]
    for q in questions:
        norm_q = normalize_text(q)
        unaccented_q = remove_vietnamese_accents(norm_q)
        # Chỉ thêm dạng có dấu và không dấu
        KEYWORD_ANSWER_MAP[norm_q] = item
        KEYWORD_ANSWER_MAP[unaccented_q] = item

# Tạo tập hợp tất cả từ khóa đã chuẩn hóa/loại dấu để so khớp từng phần/từng từ
ALL_KEYWORDS_SET = set()
KEYWORD_TO_ITEM_MAP = {}
for item in admissions_data.get('questions', []):
    questions = item.get('question', [])
    if isinstance(questions, str):
        questions = [questions]
    for q in questions:
        norm_q = normalize_text(q)
        unaccented_q = remove_vietnamese_accents(norm_q)
        ALL_KEYWORDS_SET.add(norm_q)
        ALL_KEYWORDS_SET.add(unaccented_q)
        KEYWORD_TO_ITEM_MAP[norm_q] = item
        KEYWORD_TO_ITEM_MAP[unaccented_q] = item


def find_answer_and_media(question):
    if not model or st.session_state.question_embeddings is None or st.session_state.tfidf_matrix is None:
        return "Chatbot đang gặp sự cố, vui lòng thử lại sau.", "text", None
    norm_question = normalize_and_unaccent(question)
    # Tra cứu từ khóa trực tiếp (ưu tiên khớp chính xác)
    direct_item = KEYWORD_ANSWER_MAP.get(norm_question)
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

    tokens = norm_question.split()
    num_tokens = len(tokens)

    # 1. Khớp chính xác (đã chuẩn hóa và loại dấu)
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

    # 2. Khớp cụm từ n-gram (từ dài nhất đến ngắn nhất)
    phrase_matches = []
    for length in range(num_tokens, 1, -1):
        for i in range(num_tokens - length + 1):
            phrase = ' '.join(tokens[i:i+length])
            item = KEYWORD_TO_ITEM_MAP.get(phrase)
            if item:
                phrase_matches.append((phrase, item, length))
    # Nếu có khớp cụm từ, luôn trả về cụm từ tốt nhất và bỏ qua khớp từng từ
    if phrase_matches:
        # Tìm cụm từ khớp dài nhất và ưu tiên cao nhất
        phrase_matches.sort(key=lambda x: (-x[2], -len(x[0])))
        best_phrase, best_item, _ = phrase_matches[0]
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

    # 3. Nếu truy vấn có 2 từ, thử khớp cụm từ 2 từ
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

    # 4. Nếu truy vấn >2 từ, chuyển sang khớp từng từ nếu không có khớp cụm từ
    matched_items = []
    matched_tokens = []
    # Chỉ thực hiện khớp từng từ nếu không c�� khớp cụm từ
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
        # Nếu không có best_token_item, thử khớp từng từ (chỉ trả về cái tốt nhất)
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
                    if norm_q in user_input:
                        if len(norm_q) > best_length:
                            best_length = len(norm_q)
                            best_item = item
            if not best_item:
                best_length = 0
                for idx, item in enumerate(matched_items):
                    questions = item.get('question', [])
                    if isinstance(questions, str):
                        questions = [questions]
                    for q in questions:
                        norm_q = normalize_and_unaccent(q)
                        if norm_q == matched_tokens[idx]:
                            if len(norm_q) > best_length:
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

    # 5. Khớp mờ (fuzzy matching) là phương án cuối cùng
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

    # Không tìm thấy kết quả phù hợp
    return "Xin lỗi, tôi chưa tìm thấy thông tin phù hợp.", "text", None


# --- GIAO DIỆN STREAMLIT ---
def main():
    st.title("Chatbot Tư vấn Tuyển sinh")
    st.markdown("Hỏi về thông tin tuyển sinh và xem hình ảnh hoặc video liên quan!")

    if 'messages' not in st.session_state:
        st.session_state.messages = []

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
                            if isinstance(img_path, str) and img_path.strip() != "":
                                abs_img_path = os.path.join(os.path.dirname(__file__), img_path)
                                if os.path.exists(abs_img_path):
                                    valid_images_paths.append(abs_img_path)
                                else:
                                    st.warning(f"Không tìm thấy hình ảnh: {img_path}")
                        if valid_images_paths:
                            num_cols = min(len(valid_images_paths), 3)
                            cols = st.columns(num_cols)
                            if not isinstance(captions, list):
                                captions = [captions] if captions else [f"Ảnh {i + 1}" for i in range(len(valid_images_paths))]
                            captions = captions[:len(valid_images_paths)]
                            for i, abs_img_path in enumerate(valid_images_paths):
                                with cols[i % num_cols]:
                                    st.image(abs_img_path, caption=captions[i] if i < len(captions) else f"Ảnh {i + 1}")
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


if __name__ == "__main__":
    main()