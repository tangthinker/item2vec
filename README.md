# Item2Vec

基于gensim的Word2Vec实现的物品向量化服务，提供REST API接口。

## 环境搭建

```shell
conda create -n item2vec python=3.9  
conda activate item2vec
pip install -r requirements.txt
```

## 启动服务

```shell
python main.py
```

服务将在 `http://localhost:8000` 启动。

## API接口

### 1. 健康检查
- **GET** `/health`
- 返回服务状态和模型信息
- 响应：
```json
{
    "message": "Hello, World!",
    "model_status": "trained",
    "vocab_size": 8
}
```

### 2. 训练模型
- **POST** `/train`
- 支持渐进训练：第一次调用创建新模型，后续调用基于已有模型继续训练
- 请求体：
```json
{
    "data": [
        "item1 item2 item3",
        "item2 item4 item5",
        "item1 item3 item6"
    ]
}
```
- 响应：
```json
{
    "message": "New model created and trained successfully",
    "vocab_size": 6,
    "model_saved": "item2vec_model"
}
```
- 第二次调用响应：
```json
{
    "message": "Model updated with new data successfully",
    "vocab_size": 8,
    "model_saved": "item2vec_model"
}
```

### 3. 生成向量
- **POST** `/gen_vec`
- 请求体：
```json
{
    "item": "item1"
}
```
- 响应：
```json
{
    "vec": [0.1, 0.2, 0.3, ...]
}
```

## 测试

运行基础测试脚本：
```shell
python test_api.py
```

运行模型持久化测试：
```shell
python test_persistence.py
```

## 技术特点

- 使用gensim的Word2Vec模型
- 向量维度：100
- 支持Skip-gram和CBOW两种模式
- 自动处理未知词汇（返回零向量）
- 支持多线程训练
- **渐进训练**：支持基于已有模型继续训练，无需重新开始
- **模型持久化**：自动保存模型到本地文件，服务重启后自动加载

## 配置参数

在`main.py`中可以调整以下参数：
- `vector_size`: 向量维度
- `window`: 上下文窗口大小
- `min_count`: 最小词频
- `workers`: 并行线程数
- `sg`: 模型类型（1=Skip-gram, 0=CBOW）
- `epochs`: 训练轮数

## 模型文件

- 模型自动保存为：`item2vec_model`
- 服务启动时自动加载已保存的模型
- 每次训练后自动更新保存的模型文件