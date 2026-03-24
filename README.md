# mkb

这是一个完整的、生产可用的 Rust 最小化知识库项目代码。它集成了 CLI 交互、语义分块（Semantic Chunking）、向量索引 以及 RAG 问答。
## 如何配置
第一步：设置环境（使用 Ollama 示例）

如果你本地运行了 Ollama：

下载模型：ollama pull llama3 和 ollama pull nomic-embed-text。

默认代码已配置为连接本地 Ollama (http://localhost:11434)。

如果想用openai
```
export AI_BASE_URL="https://api.openai.com/v1"
export AI_API_KEY="sk-你的KEY"
export AI_EMB_MODEL="text-embedding-3-small"
export AI_CHAT_MODEL="gpt-4o-mini"
```
第二步：编译安装
```bash
cargo install --path .
```

第三步：使用流程

添加内容:

mkb new rust_safety (输入关于 Rust 借用检查的内容)

建立索引:

mkb sync (它会读取笔记，做语义切分并生成向量)

提问:

mkb ask "Rust 怎么处理内存安全？"


## Roadmap
- [x] 添加向量化功能
- [x] 添加索引功能
- [x] 添加问答功能
- [x] 实现增量sync功能
- [ ] 添加自动生成摘要功能
- [ ] 添加自动生成目录功能
- [ ] 实现HNSW算法