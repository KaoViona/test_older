import os
import json
from http.server import BaseHTTPRequestHandler
from google import genai
from google.genai import types
from pinecone import Pinecone

def query_rag_response(user_query: str) -> str:
    try:
        gemini_key = os.environ.get("GEMINI_API_KEY")
        pinecone_key = os.environ.get("PINECONE_API_KEY")

        if not gemini_key or not pinecone_key:
            return "系統提示：目前環境變數中未設定 GEMINI_API_KEY 或 PINECONE_API_KEY，請於 Vercel 後台設定。"

        # 初始化 SDK
        client = genai.Client(api_key=gemini_key)
        pc = Pinecone(api_key=pinecone_key)
        index = pc.Index("elderly-health-rag")

        # 1. 提問轉向量
        embed_response = client.models.embed_content(
            model="text-embedding-004",
            contents=user_query
        )
        query_vector = embed_response.embedding.values

        # 2. Pinecone 檢索
        search_results = index.query(
            vector=query_vector,
            top_k=2,
            include_metadata=True
        )

        # 3. 知識提取
        retrieved_contexts = [
            match["metadata"]["text"] for match in search_results.get("matches", []) if "metadata" in match and "text" in match["metadata"]
        ]
        context_str = "\n".join(retrieved_contexts)

        # 4. Prompt 組合 (加強權威性與同理回應)
        system_instruction = """
        你是一位專為高齡者設計的智慧健康陪伴助手。
        請根據提供的【衛教參考資料】來回答長者的問題。
        回答原則：
        1. 語氣溫和親切、具同理心。
        2. 給予白話且具體可執行的建議，並標註衛教或醫學指南根據。
        3. 提醒長者如有持續不適應尋求專業醫師協助。
        """

        prompt = f"""
        【衛教參考資料】:
        {context_str}

        【長者的問題】:
        {user_query}
        """

        # 5. Gemini 生成
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=0.3
            )
        )
        return response.text
    except Exception as e:
        print(f"Error in RAG: {e}")
        return "對不起，我現在連線有點問題，請稍後再試試看喔。"

# Vercel Serverless Handler
class handler(BaseHTTPRequestHandler):
    # 處理 CORS 預檢請求
    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()

    # 處理 POST 對話請求
    def do_POST(self):
        try:
            content_length = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(content_length)
            data = json.loads(body.decode('utf-8'))
            
            user_text = data.get("message", "")
            reply_text = query_rag_response(user_text)

            self.send_response(200)
            self.send_header('Access-Control-Allow-Origin', '*')
            self.send_header('Content-Type', 'application/json; charset=utf-8')
            self.end_headers()
            
            response_data = json.dumps({"reply": reply_text}, ensure_ascii=False)
            self.wfile.write(response_data.encode('utf-8'))
        except Exception as e:
            self.send_response(500)
            self.send_header('Access-Control-Allow-Origin', '*')
            self.send_header('Content-Type', 'application/json; charset=utf-8')
            self.end_headers()
            self.wfile.write(json.dumps({"error": str(e)}).encode('utf-8'))