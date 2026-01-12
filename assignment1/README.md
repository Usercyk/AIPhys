# Assignment 1

## 学生信息
- 姓名：曹以楷
- 学号：2400011486

## 运行方法

1. 安装依赖：
	```bash
	pip install openai colorama python-dotenv
	```

2. 配置环境变量：
	在项目根目录下创建 `.env` 文件，内容如下：
	```env
	DASHSCOPE_API_KEY=你的API密钥
	```

3. 运行主程序：
	```bash
	python code/main.py
	```

4. 按照命令行提示选择流式输出、深度思考、联网搜索等功能。

## 主要结果

- 实现了一个支持多轮对话的 LLM 客户端，支持 OpenAI 兼容 API。
- 支持流式输出、深度思考（可设定思考步数）、联网搜索等高级功能。
- 交互式命令行界面，支持自定义配置。
- 自动统计每轮对话的 token 使用量。
- 支持清空历史、退出等指令。

## 致谢

README部分由AI完善。
