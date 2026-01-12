# RAG Agent

## 代码实现说明

本项目基于 assignment1 的 LLM 客户端进行扩展开发：
- `llm_client.py` 是在 assignment1 客户端基础上修改完善
- RAG agent 是从 LLM 基类派生的子类，实现了检索增强功能
- 新增向量数据库管理模块实现上下文检索

代码注释以及RAG测试数据由Cline生成。
README中**部分**由Cline生成。

## 嵌入模型优化

在测试中，发现ChromaDB给出的默认嵌入模型对中文内容的识别较差。

例如查询“如何推导理想气体的压强公式？”时，给出的前10个相关文本都与压强无关，然而这是其中一条数据的原话。

故更改为物院api中的text-embedding-v4进行测试，问题解决

## 文本分割

除了添加已经处理好的数据之外，我使用langchain的text_splitter，递归地分割长文本来形成document。

然而……嗯对大概就是发现了langchain_chroma里自带的Chroma几乎是和我的collection.py功能一样，还实现了langchain的vectorstores的统一接口之类的。但我写都写了……就没改成langchain的。主要第一是运行效率上本身差别不大，第二是修改的话我需要改多个地方，影响范围较大。

## 主要成果

1. **增强的 LLM 客户端**：
   - 完善了所有代码注释（英文）
   - 优化了流式处理逻辑
   - 增加了 token 使用统计功能
   - 实现交互式配置向导（流式输出/深度思考/联网搜索）

2. **创新的 RAG 架构**：
   - 实现检索增强生成工作流
   - 支持动态配置搜索数量
   - 多数据源支持（JSON/文本/维基百科）
   - 文本分割功能（RecursiveCharacterTextSplitter）

3. **用户友好体验**：
   - 交互式配置向导
   - 实时流式响应
   - 彩色终端标识区分对话角色
   - 命令快捷操作（quit/clear）

4. **工程规范**：
   - 类型提示全覆盖
   - 完善的错误处理机制
   - 进度条显示文档导入

## 文件结构

```
assignment2/
├── src/
│   ├── collection.py       # 向量数据库管理（支持JSON/文本/维基百科）
│   ├── colors.py           # 终端颜色工具（提供ANSI颜色常量）
│   ├── llm_client.py       # LLM 客户端实现（交互式配置向导）
│   └── main.py             # 主入口和 RAG 代理（动态搜索配置）
├── requirements.txt        # 依赖清单
└── README.md               # 本文件
```

## 运行方法

### 1. 激活环境
```bash
conda activate pubpy
```

### 2. 配置环境变量
创建 `.env` 文件并添加 API 密钥：
```env
DASHSCOPE_API_KEY=您的API密钥
OPENAI_API_KEY=您的OpenAI_API密钥（可选）
```

### 3. 启动应用
```bash
python src/main.py
```

## 使用说明

1. 根据提示配置运行时选项（包括LLM配置和RAG搜索数量）
2. 开始与 RAG 代理对话
3. 使用命令：
   - `quit` - 退出对话
   - `clear` - 清空对话历史

## 配置选项

### Embedding Function 选择
在初始化CollectionManager时，可通过参数指定embedding function：
- 默认使用本地embedding function
- 提供OpenAI API密钥、API基础地址和模型名称可使用OpenAI embedding
- 支持扩展其他自定义embedding function

```python
# 使用OpenAI embedding示例
collection_manager = CollectionManager(
    api_key="your_openai_api_key",
    api_base="https://api.openai.com/v1",
    model_name="text-embedding-ada-002"
)
```

### LLM 配置
通过 `LLMConfig` 类支持多种配置选项：
- `enable_thinking`：启用/禁用深度推理
- `thinking_budget`：限制推理 token 数量（可选）
- `stream`：启用流式响应
- `enable_search`：启用联网搜索增强
