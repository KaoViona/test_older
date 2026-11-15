import os
from flask import Flask, request, jsonify, send_from_directory
from google import genai
from google.genai import types

app = Flask(__name__, static_folder='.', static_url_path='')

def query_rag_response(user_query: str) -> str:
    try:
        gemini_key = os.environ.get("GEMINI_API_KEY")

        if not gemini_key:
            return "系統提示：本地尚未設定 GEMINI_API_KEY 環境變數。"

        client = genai.Client(api_key=gemini_key)

        # 系統設定與陪伴原則
        system_instruction = """
        你是一位專為高齡者設計的智慧健康陪伴助手。
        請以溫和親切、具同理心的語氣與長者對話。
        回答原則：
        1. 給予白話且具體可執行的健康與生活建議。
        2. 提醒長者如有持續不適應尋求專業醫師協助。
        """

        # 直接透過 Gemini 生成關懷與衛教回覆
        response = client.models.generate_content(
            model="gemini-3.6-flash",
            contents=user_query,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=0.3
            )
        )
        return response.text
    except Exception as e:
        print(f"Error in Generation: {e}")
        return "對不起，我現在連線有點問題，請稍後再試試看喔。"

# 首頁導向 index.html
@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

# API 路由
@app.route('/api/chat', methods=['POST'])
def chat_api():
    data = request.get_json()
    user_text = data.get("message", "")
    reply_text = query_rag_response(user_text)
    return jsonify({"reply": reply_text})

if __name__ == '__main__':
    print("🚀 本地伺服器已啟動：http://127.0.0.1:5000")
    app.run(port=5000, debug=True)