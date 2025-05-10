# AutoGen RAG Agent

这是一个基于 AutoGen 框架构建的 RAG (Retrieval-Augmented Generation) agent 实现。

## 功能特点

- 使用 LangChain 进行文档处理和向量存储
- 使用 Chroma 作为向量数据库
- 使用 AutoGen 框架实现多 agent 协作
- 支持文档检索和智能问答

## 安装

1. 克隆仓库
2. 安装依赖：
```bash
pip install -r requirements.txt
```

## 配置

1. 创建 `.env` 文件并添加你的 OpenAI API 密钥：
```
OPENAI_API_KEY=your_api_key_here
```

2. 在 `documents` 目录中放入你的文档（支持 .txt 格式）

## 使用方法

1. 确保你的文档已经放在 `documents` 目录中
2. 运行示例：
```bash
python rag_agent.py
```

## 自定义

你可以通过修改以下参数来自定义 RAG agent 的行为：

- `chunk_size`：文档分块大小
- `chunk_overlap`：文档分块重叠大小
- `k`：检索的文档数量
- 模型配置（在 `config_list` 中）

## 注意事项

- 确保你有足够的 OpenAI API 额度
- 文档最好是纯文本格式
- 建议文档大小适中，过大的文档可能会影响检索效果 