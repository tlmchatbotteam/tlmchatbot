# import json
# import os
# from flask import Flask, request, jsonify, render_template, send_from_directory
# from openai import OpenAI
#
# # Khởi tạo Flask, trỏ tới thư mục templates
# app = Flask(__name__, template_folder="templates")
#
# # Load dữ liệu JSON
# with open("admissions_data.json", "r", encoding="utf-8") as f:
#     admissions_data = json.load(f)
#
# def gpt_select_entry(user_message: str):
#     # Tạo danh sách các câu hỏi từ JSON
#     questions = []
#     for idx, item in enumerate(admissions_data):
#         q = item.get("question")
#         if isinstance(q, str):
#             questions.append(f"{idx}: {q}")
#         elif isinstance(q, list):
#             for q_item in q:
#                 questions.append(f"{idx}: {q_item}")
#
#     prompt = f"""
#     Câu hỏi của người dùng: "{user_message}"
#     Dưới đây là các lựa chọn:
#
#     {chr(10).join(questions)}
#
#     Hãy trả về chỉ số (index) phù hợp nhất trong dữ liệu.
#     Nếu không có, trả về -1.
#     """
#
#     try:
#         response = client.chat.completions.create(
#             model="gpt-4o-mini",
#             messages=[
#                 {"role": "system", "content": "Bạn là bộ não, chỉ chọn index."},
#                 {"role": "user", "content": prompt},
#             ],
#             temperature=0
#         )
#         idx_str = response.choices[0].message.content.strip()
#         return int(idx_str) if idx_str.isdigit() else -1
#     except Exception as e:
#         print("❌ GPT chọn lỗi:", e)
#         return -1
# ["questions"]
#
# # Ghép tất cả Q-A thành text để gửi GPT
# qa_pairs = "\n".join(
#     f"- Q: {item['question']} | A: {item['answer']}"
#     for item in admissions_data
# )
#
# # Lấy API key từ biến môi trường (set trên VM)
# api_key = os.environ.get("OPENAI_API_KEY")
# if not api_key:
#     raise ValueError("❌ OPENAI_API_KEY chưa được thiết lập trong biến môi trường!")
#
# client = OpenAI(api_key=api_key)
#
#
#
#     # Bước 2: viết lại tự nhiên
#     rewrite_prompt = f"""
# Dữ liệu gốc (answer): "{selected_answer}"
#
# Nhiệm vụ:
# - Viết lại câu trả lời một cách tự nhiên, thân thiện.
# - Giữ nguyên đầy đủ ý chính từ dữ liệu gốc.
# - Tuyệt đối KHÔNG được bịa thêm thông tin ngoài dữ liệu gốc.
#     """
#
#     rewrite_response = client.chat.completions.create(
#         model="gpt-4o",
#         messages=[{"role": "user", "content": rewrite_prompt}],
#         temperature=0.5
#     )
#     final_answer = rewrite_response.choices[0].message.content.strip()
#     return (final_answer, selected_answer)
#
#
# def extract_media(item):
#     media = {}
#     if item.get("images"):
#         media["images"] = item["images"]
#     if item.get("captions"):
#         media["captions"] = item["captions"]
#     if item.get("video_url"):
#         media["video_url"] = item["video_url"]
#     return media
#
# def find_media_for_question(user_message: str):
#     for item in admissions_data:
#         q = item.get("question")
#         if not q:
#             continue
#
#         if isinstance(q, str):
#             if q.lower() in user_message.lower():
#                 return extract_media(item)
#         elif isinstance(q, list):
#             for q_item in q:
#                 if q_item.lower() in user_message.lower():
#                     return extract_media(item)
#     return {}
#
#
#
#
# @app.route("/", methods=["GET"])
# def home():
#     return render_template("index.html")
#
#
#
#
# @app.route("/ask", methods=["POST"])
# def ask():
#     data = request.get_json() or {}
#     user_input = data.get("question", "")
#
#     idx = gpt_select_entry(user_input)
#
#     if idx >= 0 and idx < len(admissions_data):
#         item = admissions_data[idx]
#         entry = {"text": item.get("answer", "")}
#         if item.get("images"):
#             entry["images"] = ["/images/" + img for img in item["images"]]
#             entry["captions"] = item.get("captions", [])
#         if item.get("video_url"):
#             entry["video_url"] = item["video_url"]
#     else:
#         entry = {"text": "Xin lỗi, tôi chưa có thông tin về câu hỏi này."}
#
#     return jsonify([entry])
#
#
#
#
# @app.route("/images/<path:filename>")
# def serve_images(filename):
#     return send_from_directory("images", filename)
#
#
# if __name__ == "__main__":
#     port = int(os.environ.get("PORT", 8080))
#     debug = os.environ.get("FLASK_DEBUG") == "1"
#
#     api_key = os.environ.get("OPENAI_API_KEY")
#     if api_key:
#         print(f"✅ OPENAI_API_KEY loaded: ...{api_key[-4:]}")
#     else:
#         print("❌ OPENAI_API_KEY is NOT set!")
#
#     app.run(host="0.0.0.0", port=port, debug=debug)
