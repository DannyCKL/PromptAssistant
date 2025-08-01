---
# 详细文档见https://modelscope.cn/docs/%E5%88%9B%E7%A9%BA%E9%97%B4%E5%8D%A1%E7%89%87
domain: #领域：cv/nlp/audio/multi-modal/AutoML
# - cv
tags: #自定义标签
-
datasets: #关联数据集
  evaluation:
  #- iic/ICDAR13_HCTR_Dataset
  test:
  #- iic/MTWI
  train:
  #- iic/SIBR
models: #关联模型
#- iic/ofa_ocr-recognition_general_base_zh

## 启动文件(若SDK为Gradio/Streamlit，默认为app.py, 若为Static HTML, 默认为index.html)
# deployspec:
#   entry_file: app.py
license: Apache License 2.0
---
# DeepSeek Chat API Demo

这是一个基于OpenAI DeepSeek Chat模型的聊天应用，使用Gradio构建用户界面。

## 功能特性

- 支持与OpenAI DeepSeek Chat模型进行对话
- 流式和非流式响应模式
- 会话管理（创建、删除、重命名对话）
- 可自定义提示词模板
- 思考过程显示
- 智能流式输出效果
- 错误处理和重试机制

## 安装依赖

```bash
pip install -r requirements.txt
```

## 环境配置

在使用前，需要设置OpenAI API密钥：

```bash
export OPENAI_API_KEY="your_openai_api_key_here"
```

### 测试配置

运行测试脚本验证配置是否正确：

```bash
python test_deepseek.py
```

## 启动应用

```bash
python app.py
```

应用将在 http://127.0.0.1:7860 启动。

## 使用说明

1. **模型选择**: 当前支持 deepseek-chat 模型
2. **提示词模板**: 可以选择不同的系统提示词模板
3. **流式输出**: 可以选择是否使用流式响应模式
4. **会话管理**: 
   - 创建新对话
   - 删除现有对话
   - 重命名对话
   - 切换不同对话

## 故障排除

### 1. API密钥设置
如果遇到API密钥问题：
```bash
export OPENAI_API_KEY="your_openai_api_key_here"
```

### 2. 网络连接问题
如果遇到连接超时错误：
- 检查网络连接
- 确认API密钥有效
- 尝试使用VPN（如果在中国大陆）

### 3. 依赖问题
如果遇到依赖问题：
```bash
pip install --upgrade openai gradio
```

## 文件结构

- `app.py`: 主应用文件
- `prompt.json`: 提示词模板配置
- `requirements.txt`: Python依赖包
- `test_deepseek.py`: 模型测试脚本
- `conversations/`: 对话历史存储目录

## 技术特点

- **智能流式输出**: 根据字符数量和时间间隔优化输出频率
- **重试机制**: 3次重试，指数退避策略
- **错误处理**: 详细的错误信息和解决方案建议
- **会话持久化**: 自动保存对话历史

## 许可证

Apache License 2.0