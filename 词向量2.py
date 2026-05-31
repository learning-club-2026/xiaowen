from sentence_transformers import SentenceTransformer

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

while True:
    sentence1 = input("请输入第一个句子：")
    if sentence1.strip().lower() == "exit":
        break

    sentence2 = input("请输入第二个句子：")

    # 生成句子向量
    embedding1 = model.encode(sentence1)
    embedding2 = model.encode(sentence2)

    # 计算相似度
    similarity = model.similarity(embedding1, embedding2)

    print(f"\n两个句子的语义相似度：{similarity.item():.4f}\n")
