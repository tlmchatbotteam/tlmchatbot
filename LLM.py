import json
import os
from flask import Flask, request, jsonify
from openai import OpenAI

# Khởi tạo Flask
app = Flask(__name__)

# Load dữ liệu JSON
with open("admissions_data.json", "r", encoding="utf-8") as f:
    admissions_data = json.load(f)["questions"]

# Ghép tất cả Q-A thành text để gửi GPT
qa_pairs = "\n".join(
    f"- Q: {item['question']} | A: {item['answer']}"
    for item in admissions_data
)

# Lấy API key từ biến môi trường (set trên VM)
api_key = os.environ.get("OPENAI_API_KEY")
if not api_key:
    raise ValueError("❌ OPENAI_API_KEY chưa được thiết lập trong biến môi trường!")

client = OpenAI(api_key=api_key)

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    user_input = data.get("message", "")

    # Bước 1: Nhờ GPT chọn câu trả lời
    selection_prompt = f"""
Bạn là chatbot tuyển sinh.
Dưới đây là dữ liệu hỏi–đáp:

{qa_pairs}

Người dùng hỏi: "{user_input}"

Nhiệm vụ:
- Chọn câu trả lời (A) phù hợp nhất trong dữ liệu trên.
- Chỉ chọn đúng một câu trả lời.
- Nếu không tìm thấy câu trả lời liên quan, hãy trả lời: "NOT_FOUND".
    """

    selection_response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": selection_prompt}],
        temperature=0
    )
    selected_answer = selection_response.choices[0].message.content.strip()

    if selected_answer == "NOT_FOUND":
        return jsonify({"answer": "Xin lỗi, tôi không có thông tin về câu hỏi này."})

    # Bước 2: Tìm media (ảnh/video) kèm theo
    media = {}
    for item in admissions_data:
        if str(item.get("answer")).strip() == selected_answer:
            if item.get("images"):
                media["images"] = item.get("images")
            if item.get("captions"):
                media["captions"] = item.get("captions")
            if item.get("video_url"):
                media["video_url"] = item.get("video_url")
            break

    # Bước 3: Nhờ GPT viết lại câu trả lời
    rewrite_prompt = f"""
Dữ liệu gốc (answer): "{selected_answer}"

Nhiệm vụ:
- Viết lại câu trả lời một cách tự nhiên, thân thiện.
- Giữ nguyên đầy đủ ý chính từ dữ liệu gốc.
- Tuyệt đối KHÔNG được bịa thêm thông tin ngoài dữ liệu gốc.
    """

    rewrite_response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": rewrite_prompt}],
        temperature=0.5
    )
    final_answer = rewrite_response.choices[0].message.content.strip()

    return jsonify({
        "answer": final_answer,
        "media": media if media else None
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    debug = os.environ.get("FLASK_DEBUG") == "1"
    app.run(host="0.0.0.0", port=port, debug=debug)
