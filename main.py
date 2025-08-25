from fastapi import FastAPI
import gensim
import numpy as np
from proto import TrainReq, GenVecReq, GenVecResp
import uvicorn

app = FastAPI()

# 全局变量存储训练好的模型
word2vec_model = None

# 尝试加载已保存的模型
try:
    word2vec_model = gensim.models.Word2Vec.load("item2vec_model")
    print("Loaded existing model from item2vec_model")
except FileNotFoundError:
    print("No existing model found, will create new model on first training")
    word2vec_model = None

@app.get("/health")
def health():
    model_status = "trained" if word2vec_model is not None else "not_trained"
    vocab_size = len(word2vec_model.wv.key_to_index) if word2vec_model is not None else 0
    return {
        "message": "Hello, World!", 
        "model_status": model_status,
        "vocab_size": vocab_size
    }

@app.post("/train")
def train(req: TrainReq):
    global word2vec_model
    
    # 将输入数据转换为gensim需要的格式（每个文档是单词列表）
    sentences = []
    for text in req.data:
        # 简单的分词，可以根据需要改进
        words = text.lower().split()
        if words:  # 确保不是空列表
            sentences.append(words)
    
    if word2vec_model is None:
        # 第一次训练，创建新模型
        word2vec_model = gensim.models.Word2Vec(
            sentences=sentences,
            vector_size=100,  # 向量维度
            window=5,         # 上下文窗口大小
            min_count=1,      # 最小词频
            workers=4,        # 并行线程数
            sg=1,            # 使用Skip-gram模型
            epochs=10        # 训练轮数
        )
        message = "New model created and trained successfully"
    else:
        # 渐进训练，基于已有模型继续训练
        # 构建词汇表
        word2vec_model.build_vocab(sentences, update=True)
        # 继续训练
        word2vec_model.train(
            sentences, 
            total_examples=len(sentences), 
            epochs=10
        )
        message = "Model updated with new data successfully"
    
    # 保存模型到项目根目录
    model_path = "item2vec_model"
    word2vec_model.save(model_path)
    
    return {"message": message, "vocab_size": len(word2vec_model.wv.key_to_index), "model_saved": model_path}

@app.post("/gen_vec")
def gen_vec(req: GenVecReq):
    global word2vec_model
    
    if word2vec_model is None:
        return {"error": "Model not trained yet. Please train the model first."}
    
    item = req.item.lower()
    
    try:
        # 获取item的向量
        vec = word2vec_model.wv[item].tolist()
        return GenVecResp(vec=vec)
    except KeyError:
        # 如果item不在词汇表中，返回零向量或随机向量
        return GenVecResp(vec=np.zeros(100).tolist())

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)