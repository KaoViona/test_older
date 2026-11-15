import os
from google import genai
from pinecone import Pinecone, ServerlessSpec

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")

if not GEMINI_API_KEY or not PINECONE_API_KEY:
    raise ValueError("請先設定 GEMINI_API_KEY 與 PINECONE_API_KEY 環境變數！")

client = genai.Client(api_key=GEMINI_API_KEY)
pc = Pinecone(api_key=PINECONE_API_KEY)

index_name = "elderly-health-rag"

existing_indexes = pc.list_indexes().names()
if index_name not in existing_indexes:
    print("正在建立 Pinecone Index...")
    pc.create_index(
        name=index_name,
        dimension=768,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

index = pc.Index(index_name)

health_knowledge = [
    {"id": "doc1", "text": "高血壓長者每日鹽分攝取量應控制在 6 克以下，並建議定期量測血壓。"},
    {"id": "doc2", "text": "防跌須知：浴室應加裝防滑墊與扶手，走道保持乾燥與光線充足。"},
    {"id": "doc3", "text": "糖尿病衛教：若出現頭暈、冒冷汗等低血糖症狀，應立即補充15克快速吸收之糖分（如半杯果汁）。"}
]

vectors = []
for doc in health_knowledge:
    # 修正：使用全名文字嵌入模型呼叫
    response = client.models.embed_content(
        model="text-embedding-004",
        contents=doc["text"]
    )
    embedding = response.embedding.values
    vectors.append((doc["id"], embedding, {"text": doc["text"]}))

index.upsert(vectors=vectors)
print("✅ 衛教知識庫已成功向量化並匯入 Pinecone！")