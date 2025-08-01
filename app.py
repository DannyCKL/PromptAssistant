import os
import gradio as gr
from openai import OpenAI
import openai
import json
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Generator, Optional, Tuple, Union
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 加载提示词模板
def load_prompts() -> Dict[str, Dict[str, str]]:
    prompt_file = Path("prompt.json")
    if prompt_file.exists():
        try:
            with open(prompt_file, "r", encoding="utf-8") as f:
                prompts = json.load(f)
                return prompts
        except Exception as e:
            print(f"加载提示词模板失败: {e}")
    return {
        "default": {
            "system": "你是DeepSeek Chat，一个由DeepSeek开发的人工智能助手，擅长对话和思考。",
            "description": "默认系统提示词"
        }
    }

# 初始化 OpenAI 客户端
def get_client():
    return OpenAI(
        base_url="https://api.deepseek.com",
        # 使用OpenAI API
        api_key=os.getenv('OPENAI_API_KEY'),
        # 使用默认的OpenAI API地址
        timeout=120.0,  # 设置120秒超时
    )

# 流式响应处理函数
def process_stream_response(stream) -> Generator[str, None, str]:
    full_response = ""
    reasoning = ""
    buffer = ""
    last_yield_time = time.time()
    char_count = 0
    
    for chunk in stream:
        # 处理推理内容（如果存在）
        if hasattr(chunk.choices[0].delta, 'reasoning_content') and chunk.choices[0].delta.reasoning_content is not None:
            reasoning_content = chunk.choices[0].delta.reasoning_content
            reasoning += reasoning_content
            yield reasoning, full_response
        
        # 处理回复内容
        if hasattr(chunk.choices[0].delta, 'content') and chunk.choices[0].delta.content is not None:
            content = chunk.choices[0].delta.content
            full_response += content
            buffer += content
            char_count += len(content)
            
            # 智能流式输出：根据字符数量和时间间隔决定输出频率
            current_time = time.time()
            should_yield = (
                current_time - last_yield_time >= 0.03 or  # 每30ms输出一次
                char_count >= 5 or  # 每5个字符输出一次
                content in ['\n', '。', '！', '？', '.', '!', '?']  # 在标点符号处输出
            )
            
            if should_yield:
                yield reasoning, full_response
                last_yield_time = current_time
                char_count = 0
                buffer = ""
    
    # 确保最后的内容也被输出
    if buffer:
        yield reasoning, full_response
    
    return reasoning, full_response

# 非流式响应处理函数
def process_non_stream_response(response) -> tuple[str, str]:
    response_data = response.model_dump()
    reasoning = ""
    content = ""
    
    if 'choices' in response_data and response_data['choices']:
        if 'message' in response_data['choices'][0]:
            content = response_data['choices'][0]['message'].get('content', '')
            reasoning = response_data['choices'][0]['message'].get('reasoning_content', '')
    
    return reasoning, content

# 会话管理类
class ConversationManager:
    def __init__(self, storage_dir="conversations"):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(exist_ok=True)
        self.current_conversation_id = None
        self.conversations = self._load_conversations_index()
    
    def _load_conversations_index(self) -> Dict[str, Dict]:
        index_file = self.storage_dir / "index.json"
        if index_file.exists():
            try:
                with open(index_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except json.JSONDecodeError:
                return {}
        return {}
    
    def _save_conversations_index(self):
        with open(self.storage_dir / "index.json", "w", encoding="utf-8") as f:
            json.dump(self.conversations, f, ensure_ascii=False, indent=2)
    
    def create_conversation(self, title: Optional[str] = None) -> str:
        """创建新的对话，返回对话ID"""
        conversation_id = str(int(time.time()))
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        if not title:
            title = "新对话"
        
        self.conversations[conversation_id] = {
            "id": conversation_id,
            "title": title,
            "created_at": timestamp,
            "updated_at": timestamp,
            "messages": [],
            "likes": 0,
            "dislikes": 0
        }
        
        self._save_conversations_index()
        self.current_conversation_id = conversation_id
        return conversation_id
    
    def update_title_with_ai(self, conversation_id: str, model: str = "deepseek-chat"):
        """使用AI更新对话标题"""
        if conversation_id not in self.conversations:
            return False
        
        messages = self.conversations[conversation_id].get("messages", [])
        if len(messages) >= 2:  # 至少有一轮对话
            try:
                new_title = generate_conversation_title(messages, model)
                self.conversations[conversation_id]["title"] = new_title
                self._save_conversations_index()
                return True
            except Exception as e:
                print(f"AI生成标题失败: {e}")
                return False
        return False
    
    def get_conversation(self, conversation_id: str) -> Dict:
        """获取指定ID的对话"""
        return self.conversations.get(conversation_id, {})
    
    def get_conversation_history(self, conversation_id: str) -> List[Dict[str, str]]:
        """获取指定ID的对话历史，格式为Gradio messages格式"""
        conversation = self.get_conversation(conversation_id)
        if not conversation:
            return []
        
        messages = conversation.get("messages", [])
        history = []
        
        # 将消息转换为Gradio messages格式 {"role": "user/assistant", "content": "message"}
        for message in messages:
            history.append({"role": message["role"], "content": message["content"]})
        
        return history
    
    def add_message(self, conversation_id: str, role: str, content: str):
        """添加消息到指定对话"""
        if conversation_id not in self.conversations:
            return False
        
        if "messages" not in self.conversations[conversation_id]:
            self.conversations[conversation_id]["messages"] = []
        
        # 检查是否为空内容，避免保存空消息
        if not content or content.strip() == "":
            return False
        
        # 检查是否与上一条消息重复
        messages = self.conversations[conversation_id]["messages"]
        if messages and messages[-1]["role"] == role and messages[-1]["content"] == content:
            return True  # 重复消息，不保存但返回成功
        
        self.conversations[conversation_id]["messages"].append({
            "role": role,
            "content": content,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })
        
        self.conversations[conversation_id]["updated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self._save_conversations_index()
        return True
    
    def update_last_message(self, conversation_id: str, content: str):
        """更新最后一条消息的内容"""
        if conversation_id not in self.conversations:
            return False
        
        messages = self.conversations[conversation_id].get("messages", [])
        if not messages:
            return False
        
        # 更新最后一条消息
        messages[-1]["content"] = content
        messages[-1]["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        self.conversations[conversation_id]["updated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self._save_conversations_index()
        return True
    
    def remove_last_message(self, conversation_id: str):
        """移除最后一条消息"""
        if conversation_id not in self.conversations:
            return False
        
        messages = self.conversations[conversation_id].get("messages", [])
        if not messages:
            return False
        
        # 移除最后一条消息
        messages.pop()
        
        self.conversations[conversation_id]["updated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self._save_conversations_index()
        return True
    
    def like_conversation(self, conversation_id: str):
        """为对话点赞"""
        if conversation_id not in self.conversations:
            return False
        
        if "likes" not in self.conversations[conversation_id]:
            self.conversations[conversation_id]["likes"] = 0
        
        self.conversations[conversation_id]["likes"] += 1
        self.conversations[conversation_id]["updated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self._save_conversations_index()
        return True
    
    def dislike_conversation(self, conversation_id: str):
        """为对话点踩"""
        if conversation_id not in self.conversations:
            return False
        
        if "dislikes" not in self.conversations[conversation_id]:
            self.conversations[conversation_id]["dislikes"] = 0
        
        self.conversations[conversation_id]["dislikes"] += 1
        self.conversations[conversation_id]["updated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self._save_conversations_index()
        return True
    
    def get_conversation_stats(self, conversation_id: str) -> Dict[str, int]:
        """获取对话的点赞和点踩统计"""
        if conversation_id not in self.conversations:
            return {"likes": 0, "dislikes": 0}
        
        conversation = self.conversations[conversation_id]
        return {
            "likes": conversation.get("likes", 0),
            "dislikes": conversation.get("dislikes", 0)
        }
    
    def update_conversation_title(self, conversation_id: str, title: str):
        """更新对话标题"""
        if conversation_id in self.conversations:
            self.conversations[conversation_id]["title"] = title
            self._save_conversations_index()
            return True
        return False
    
    def delete_conversation(self, conversation_id: str):
        """删除指定对话"""
        if conversation_id in self.conversations:
            del self.conversations[conversation_id]
            self._save_conversations_index()
            return True
        return False
    
    def get_all_conversations(self) -> List[Dict]:
        """获取所有对话的列表，按更新时间排序"""
        conversations_list = list(self.conversations.values())
        return sorted(conversations_list, key=lambda x: x.get("updated_at", ""), reverse=True)
    
    def get_conversation_dropdown_choices(self) -> List[Tuple[str, str]]:
        """获取对话下拉菜单的选项"""
        conversations = self.get_all_conversations()
        choices = []
        for conv in conversations:
            title = conv.get("title", "新对话")
            conv_id = conv.get("id", "")
            likes = conv.get("likes", 0)
            dislikes = conv.get("dislikes", 0)
            
            if conv_id:
                # 在标题中显示点赞和点踩数量
                display_title = f"{title} 👍{likes} 👎{dislikes}"
                choices.append((display_title, conv_id))
        return choices
    
    def refresh_conversation_list(self):
        """刷新对话列表"""
        return gr.update(choices=self.get_conversation_dropdown_choices())

# 生成对话标题的函数
def generate_conversation_title(messages: List[Dict[str, str]], model: str = "deepseek-chat") -> str:
    """使用AI模型生成对话标题"""
    if not messages:
        return "新对话"
    
    # 构建对话内容字符串
    conversation_text = ""
    for msg in messages[-6:]:  # 只取最后6条消息
        role = "用户" if msg["role"] == "user" else "助手"
        conversation_text += f"{role}: {msg['content']}\n"
    
    # 构建用于生成标题的消息
    title_messages = [
        {"role": "system", "content": "你是一个专业的对话标题生成器。请根据用户和助手的对话内容，生成一个简洁、准确的标题（不超过20个字符）。标题应该概括对话的主要主题或核心问题。只返回标题，不要其他内容。"},
        {"role": "user", "content": f"请为以下对话生成标题：\n\n{conversation_text}"}
    ]
    
    try:
        client = get_client()
        response = client.chat.completions.create(
            messages=title_messages,
            model=model,
            stream=False,
            max_tokens=50,
            temperature=0.3
        )
        title = response.choices[0].message.content.strip()
        # 清理标题，移除可能的引号等
        title = title.strip('"\'')
        return title if title else "新对话"
    except Exception as e:
        print(f"生成标题失败: {e}")
        return "新对话"

# 生成回复的主函数
def generate_response(message: str, history: List[List[str]], conversation_manager: ConversationManager, stream_mode: bool = True, model: str = "deepseek-chat", prompt_template: str = "default") -> Generator[tuple, None, None]:
    client = get_client()
    
    # 确保有当前会话ID
    if not conversation_manager.current_conversation_id:
        conversation_manager.create_conversation()
    
    # 添加用户消息到会话历史
    conversation_manager.add_message(conversation_manager.current_conversation_id, "user", message)
    
    # 构建对话历史
    messages = []
    
    # 添加系统提示词
    prompts = load_prompts()
    system_prompt = prompts.get(prompt_template, prompts["default"])["system"]
    messages.append({"role": "system", "content": system_prompt})
    
    # 添加历史对话
    for human, assistant in history:
        messages.append({"role": "user", "content": human})
        messages.append({"role": "assistant", "content": assistant})
    
    # 添加当前用户消息
    messages.append({"role": "user", "content": message})
    
    # 创建聊天完成请求
    max_retries = 3
    retry_count = 0
    
    while retry_count < max_retries:
        try:
            stream = client.chat.completions.create(
                messages=messages,
                model=model,
                stream=stream_mode,
                user="123456",
            )
            break  # 成功则跳出循环
        except Exception as e:
            retry_count += 1
            error_msg = f"API调用失败 (尝试 {retry_count}/{max_retries}): {str(e)}"
            print(error_msg)
            
            if retry_count >= max_retries:
                final_error_msg = f"API调用最终失败: {str(e)}\n\n可能的解决方案:\n1. 检查网络连接\n2. 确认API密钥有效\n3. 尝试使用VPN\n4. 稍后重试"
                print(final_error_msg)
                # 保存错误消息到会话历史
                conversation_manager.add_message(conversation_manager.current_conversation_id, "assistant", final_error_msg)
                yield "", final_error_msg
                return
            else:
                print(f"等待 {retry_count * 2} 秒后重试...")
                time.sleep(retry_count * 2)  # 指数退避
    
    if stream_mode:
        gen = process_stream_response(stream)
        final_content = ""
        for reasoning, content in gen:
            if content:
                final_content = content
            yield reasoning, content
        # 保存最终的助手回复到会话历史
        if final_content:
            conversation_manager.add_message(conversation_manager.current_conversation_id, "assistant", final_content)
    else:
        reasoning, content = process_non_stream_response(stream)
        # 保存助手回复到会话历史
        conversation_manager.add_message(conversation_manager.current_conversation_id, "assistant", content)
        yield reasoning, content

# Gradio 界面
def create_interface():
    # 初始化会话管理器
    conversation_manager = ConversationManager()
    
    # 确保至少有一个对话
    if not conversation_manager.conversations:
        conversation_manager.create_conversation()
        
    # 加载提示词模板
    prompts = load_prompts()
    
    with gr.Blocks(title="提示词框架助手", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 提示词框架助手")
        
        # 当前会话ID
        current_conversation_id = gr.State(conversation_manager.current_conversation_id)
        
        # 消息历史状态（用于撤销等功能）
        message_history = gr.State([])
        current_message_index = gr.State(-1)
        
        with gr.Row():
            with gr.Column(scale=3):
                # 对话区域
                chatbot = gr.Chatbot(height=600, show_copy_button=True, type="messages")
                
                # 输入区域
                with gr.Row():
                    msg = gr.Textbox(
                        placeholder="请输入您的问题...",
                        container=False,
                        scale=8
                    )
                    submit_btn = gr.Button("发送", scale=1)
                    clear_btn = gr.Button("清空", scale=1)
                
                # 交互按钮区域
                with gr.Row():
                    undo_btn = gr.Button("撤销", size="sm", variant="secondary")
                    retry_btn = gr.Button("重试", size="sm", variant="secondary")
                    like_btn = gr.Button("👍", size="sm", variant="secondary")
                    dislike_btn = gr.Button("👎", size="sm", variant="secondary")
                    edit_btn = gr.Button("编辑", size="sm", variant="secondary")
                
                with gr.Row():
                    # 模型选择
                    model_selector = gr.Dropdown(
                        choices=["deepseek-chat", "deepseek-reasoner"],
                        value="deepseek-chat",
                        label="模型选择",
                        info="deepseek-chat: 通用对话模型\ndeepseek-reasoner: 推理专家模型",
                        interactive=True
                    )
                    
                    # 提示词模板选择
                    prompt_template_selector = gr.Dropdown(
                        choices=[(v["description"], k) for k, v in prompts.items()],
                        value="default",
                        label="提示词模板",
                        info="提示词模板\n针对各种应用场景下的优质提示词",
                        interactive=True
                    )
                    
                    # 流式输出选择
                    stream_checkbox = gr.Checkbox(
                        value=True,
                        label="流式输出",
                        info="是否使用流式响应模式",
                        interactive=True
                    )
                
                # 思考过程显示框
                thinking_box = gr.Textbox(
                    label="思考过程",
                    placeholder="这里将显示模型的思考过程（如果有）",
                    lines=5,
                    interactive=False,
                    visible=True
                )
                
                # 编辑消息输入框（默认隐藏）
                edit_msg_box = gr.Textbox(
                    label="编辑消息",
                    placeholder="编辑您的消息...",
                    lines=3,
                    visible=False
                )
            
            with gr.Column(scale=1):
                # 会话管理区
                gr.Markdown("## 对话管理")
                
                with gr.Row():
                    with gr.Column():
                        conversation_dropdown = gr.Dropdown(
                            choices=conversation_manager.get_conversation_dropdown_choices(),
                            value=conversation_manager.current_conversation_id,
                            label="选择对话",
                            interactive=True
                        )
                        
                        # 当前对话统计信息
                        stats_display = gr.Markdown("**当前对话统计：** 👍 0 👎 0")
                
                with gr.Row():
                    new_conversation_btn = gr.Button("新建对话")
                    delete_conversation_btn = gr.Button("删除对话")
                    refresh_btn = gr.Button("🔄", size="sm", variant="secondary")
                    
                    title_input = gr.Textbox(label="对话标题", placeholder="输入新的对话标题...", interactive=True)
                    rename_btn = gr.Button("重命名")
        
        # 响应函数
        def respond(message, chat_history, conversation_id, stream_mode, model, prompt_template, message_history, current_index):
            if not message:
                return "", chat_history, conversation_id, "", message_history, current_index, conversation_manager.refresh_conversation_list()
            
            # 确保有当前会话
            if not conversation_id:
                conversation_id = conversation_manager.create_conversation()
                conversation_manager.current_conversation_id = conversation_id
            
            # 添加用户消息到聊天历史和会话历史
            chat_history = chat_history + [{"role": "user", "content": message}]
            conversation_manager.add_message(conversation_id, "user", message)
            
            # 更新消息历史
            new_message_history = message_history + [chat_history.copy()]
            new_current_index = len(new_message_history) - 1
            
            # 准备历史消息给LLM
            history = []
            for i in range(len(chat_history)-1):
                if chat_history[i]["role"] == "user" and i+1 < len(chat_history) and chat_history[i+1]["role"] == "assistant":
                    history.append([chat_history[i]["content"], chat_history[i+1]["content"]])
            
            # 调用LLM生成响应
            if stream_mode:
                # 流式响应模式
                response_generator = generate_response(message, history, conversation_manager, True, model, prompt_template)
                assistant_message = {"role": "assistant", "content": ""}
                chat_history.append(assistant_message)
                
                for thinking, content in response_generator:
                    if content:
                        # 更新助手消息的内容
                        assistant_message["content"] = content
                    yield "", chat_history, conversation_id, thinking, new_message_history, new_current_index, conversation_manager.refresh_conversation_list()
                
                # 流式响应完成后，保存助手回复并更新标题
                if assistant_message["content"]:
                    conversation_manager.update_last_message(conversation_id, assistant_message["content"])
                    conversation_manager.update_title_with_ai(conversation_id, model)
            else:
                # 非流式响应模式
                thinking, full_response = generate_response(message, history, conversation_manager, False, model, prompt_template)
                chat_history.append({"role": "assistant", "content": full_response})
                conversation_manager.add_message(conversation_id, "assistant", full_response)
                # 尝试更新标题
                conversation_manager.update_title_with_ai(conversation_id, model)
                yield "", chat_history, conversation_id, thinking, new_message_history, new_current_index, conversation_manager.refresh_conversation_list()
        
        # 创建新对话
        def create_new_conversation():
            # 创建新对话
            new_id = conversation_manager.create_conversation()
            conversation_manager.current_conversation_id = new_id
            # 更新下拉列表
            return ([], "", "", new_id, conversation_manager.refresh_conversation_list(), [], -1, "**当前对话统计：** 👍 0 👎 0")
        
        # 清空当前对话
        def clear_current_chat():
            # 只清空聊天记录，但保留当前会话ID
            return [], "", "", [], -1, "**当前对话统计：** 👍 0 👎 0"
        
        # 加载选择的对话
        def load_conversation(conversation_id):
            if not conversation_id:
                return [], "", "**当前对话统计：** 👍 0 👎 0"
            
            # 处理下拉菜单的值格式 (title, id)
            if isinstance(conversation_id, tuple):
                conversation_id = conversation_id[1]
            
            # 设置当前对话ID
            conversation_manager.current_conversation_id = conversation_id
            
            # 获取对话标题
            conversation = conversation_manager.get_conversation(conversation_id)
            title = conversation.get("title", "")
            
            # 获取对话历史并转换为图形界面格式
            messages = conversation.get("messages", [])
            chat_history = []
            
            # 将消息转换为字典格式
            for message in messages:
                chat_history.append({"role": message["role"], "content": message["content"]})
            
            # 获取统计信息
            stats = conversation_manager.get_conversation_stats(conversation_id)
            stats_text = f"**当前对话统计：** 👍 {stats['likes']} 👎 {stats['dislikes']}"
            
            return chat_history, title, stats_text
        
        # 删除对话
        def delete_conversation(conversation_id):
            if not conversation_id:
                return gr.update(), "", "", [], "**当前对话统计：** 👍 0 👎 0"
            
            # 处理下拉菜单的值格式 (title, id)
            if isinstance(conversation_id, tuple):
                conversation_id = conversation_id[1]
            
            # 删除对话
            conversation_manager.delete_conversation(conversation_id)
            
            # 创建新对话
            new_id = conversation_manager.create_conversation()
            conversation_manager.current_conversation_id = new_id
            
            # 更新下拉列表
            return (conversation_manager.refresh_conversation_list(), new_id, "", [], "**当前对话统计：** 👍 0 👎 0")
        
        # 重命名对话
        def rename_conversation(conversation_id, new_title):
            if not conversation_id or not new_title:
                return gr.update(), "**当前对话统计：** 👍 0 👎 0"
            
            # 处理下拉菜单的值格式 (title, id)
            if isinstance(conversation_id, tuple):
                conversation_id = conversation_id[1]
            
            # 更新对话标题
            conversation_manager.update_conversation_title(conversation_id, new_title)
            
            # 更新下拉列表和统计显示
            return conversation_manager.refresh_conversation_list(), update_stats_display(conversation_id)
        
        # 刷新对话列表
        def refresh_conversation_dropdown():
            return conversation_manager.refresh_conversation_list(), update_stats_display(conversation_manager.current_conversation_id)
        
        # 更新统计显示
        def update_stats_display(conversation_id):
            if not conversation_id:
                return "**当前对话统计：** 👍 0 👎 0"
            
            # 处理下拉菜单的值格式 (title, id)
            if isinstance(conversation_id, tuple):
                conversation_id = conversation_id[1]
            
            stats = conversation_manager.get_conversation_stats(conversation_id)
            return f"**当前对话统计：** 👍 {stats['likes']} 👎 {stats['dislikes']}"
        
        # 撤销功能
        def undo_last_message(chat_history, message_history, current_index):
            if current_index > 0:
                # 恢复到上一个状态
                new_index = current_index - 1
                new_history = message_history[:new_index + 1] if new_index >= 0 else []
                return new_history, new_index
            return chat_history, current_index
        
        # 重试功能
        def retry_last_message(chat_history, conversation_id, stream_mode, model, prompt_template, message_history, current_index):
            if not chat_history or current_index < 0:
                return "", chat_history, conversation_id, "", message_history, current_index
            
            # 获取最后一条用户消息
            last_user_message = None
            for i in range(len(chat_history) - 1, -1, -1):
                if chat_history[i]["role"] == "user":
                    last_user_message = chat_history[i]["content"]
                    break
            
            if not last_user_message:
                return "", chat_history, conversation_id, "", message_history, current_index
            
            # 移除最后一条助手回复
            if chat_history and chat_history[-1]["role"] == "assistant":
                chat_history = chat_history[:-1]
                # 同时从会话历史中移除
                conversation_manager.remove_last_message(conversation_id)
            
            # 重新生成回复
            if stream_mode:
                # 准备历史消息
                history = []
                for i in range(len(chat_history)-1):
                    if chat_history[i]["role"] == "user" and i+1 < len(chat_history) and chat_history[i+1]["role"] == "assistant":
                        history.append([chat_history[i]["content"], chat_history[i+1]["content"]])
                
                # 流式响应
                response_generator = generate_response(last_user_message, history, conversation_manager, True, model, prompt_template)
                assistant_message = {"role": "assistant", "content": ""}
                chat_history.append(assistant_message)
                
                for thinking, content in response_generator:
                    if content:
                        assistant_message["content"] = content
                    yield "", chat_history, conversation_id, thinking, message_history, current_index
                
                # 更新标题和保存回复
                if assistant_message["content"]:
                    conversation_manager.update_last_message(conversation_id, assistant_message["content"])
                    conversation_manager.update_title_with_ai(conversation_id, model)
                    # 自动刷新对话列表
                    yield "", chat_history, conversation_id, thinking, message_history, current_index, conversation_manager.refresh_conversation_list()
            else:
                # 非流式响应
                history = []
                for i in range(len(chat_history)-1):
                    if chat_history[i]["role"] == "user" and i+1 < len(chat_history) and chat_history[i+1]["role"] == "assistant":
                        history.append([chat_history[i]["content"], chat_history[i+1]["content"]])
                
                thinking, full_response = generate_response(last_user_message, history, conversation_manager, False, model, prompt_template)
                chat_history.append({"role": "assistant", "content": full_response})
                conversation_manager.add_message(conversation_id, "assistant", full_response)
                conversation_manager.update_title_with_ai(conversation_id, model)
                yield "", chat_history, conversation_id, thinking, message_history, current_index, conversation_manager.refresh_conversation_list()
        
        # 点赞功能
        def like_message(chat_history, message_history, current_index, conversation_id):
            if conversation_id:
                # 处理下拉菜单的值格式 (title, id)
                if isinstance(conversation_id, tuple):
                    conversation_id = conversation_id[1]
                
                conversation_manager.like_conversation(conversation_id)
                print(f"用户为对话 {conversation_id} 点赞")
            
            return chat_history, message_history, current_index, conversation_manager.refresh_conversation_list(), update_stats_display(conversation_id)
        
        # 点踩功能
        def dislike_message(chat_history, message_history, current_index, conversation_id):
            if conversation_id:
                # 处理下拉菜单的值格式 (title, id)
                if isinstance(conversation_id, tuple):
                    conversation_id = conversation_id[1]
                
                conversation_manager.dislike_conversation(conversation_id)
                print(f"用户为对话 {conversation_id} 点踩")
            
            return chat_history, message_history, current_index, conversation_manager.refresh_conversation_list(), update_stats_display(conversation_id)
        
        # 编辑功能
        def edit_message(chat_history, message_history, current_index):
            if not chat_history or current_index < 0:
                return gr.update(visible=True, value="")
            
            # 获取最后一条用户消息
            last_user_message = ""
            for i in range(len(chat_history) - 1, -1, -1):
                if chat_history[i]["role"] == "user":
                    last_user_message = chat_history[i]["content"]
                    break
            
            return gr.update(visible=True, value=last_user_message)
        
        # 保存编辑的消息
        def save_edited_message(edited_content, chat_history, conversation_id, stream_mode, model, prompt_template, message_history, current_index):
            if not edited_content or not chat_history:
                return "", chat_history, conversation_id, "", message_history, current_index, gr.update(visible=False)
            
            # 更新最后一条用户消息
            for i in range(len(chat_history) - 1, -1, -1):
                if chat_history[i]["role"] == "user":
                    chat_history[i]["content"] = edited_content
                    # 同时更新会话历史中的用户消息
                    conversation_manager.update_last_message(conversation_id, edited_content)
                    break
            
            # 移除最后一条助手回复
            if chat_history and chat_history[-1]["role"] == "assistant":
                chat_history = chat_history[:-1]
                # 同时从会话历史中移除
                conversation_manager.remove_last_message(conversation_id)
            
            # 重新生成回复
            if stream_mode:
                # 准备历史消息
                history = []
                for i in range(len(chat_history)-1):
                    if chat_history[i]["role"] == "user" and i+1 < len(chat_history) and chat_history[i+1]["role"] == "assistant":
                        history.append([chat_history[i]["content"], chat_history[i+1]["content"]])
                
                # 流式响应
                response_generator = generate_response(edited_content, history, conversation_manager, True, model, prompt_template)
                assistant_message = {"role": "assistant", "content": ""}
                chat_history.append(assistant_message)
                
                for thinking, content in response_generator:
                    if content:
                        assistant_message["content"] = content
                    yield "", chat_history, conversation_id, thinking, message_history, current_index, gr.update(visible=False)
                
                # 更新标题和保存回复
                if assistant_message["content"]:
                    conversation_manager.update_last_message(conversation_id, assistant_message["content"])
                    conversation_manager.update_title_with_ai(conversation_id, model)
                    # 自动刷新对话列表
                    yield "", chat_history, conversation_id, thinking, message_history, current_index, gr.update(visible=False), conversation_manager.refresh_conversation_list()
            else:
                # 非流式响应
                history = []
                for i in range(len(chat_history)-1):
                    if chat_history[i]["role"] == "user" and i+1 < len(chat_history) and chat_history[i+1]["role"] == "assistant":
                        history.append([chat_history[i]["content"], chat_history[i+1]["content"]])
                
                thinking, full_response = generate_response(edited_content, history, conversation_manager, False, model, prompt_template)
                chat_history.append({"role": "assistant", "content": full_response})
                conversation_manager.add_message(conversation_id, "assistant", full_response)
                conversation_manager.update_title_with_ai(conversation_id, model)
                yield "", chat_history, conversation_id, thinking, message_history, current_index, gr.update(visible=False), conversation_manager.refresh_conversation_list()
        
        # 事件绑定
        msg.submit(fn=respond, 
                  inputs=[msg, chatbot, current_conversation_id, stream_checkbox, model_selector, prompt_template_selector, message_history, current_message_index], 
                  outputs=[msg, chatbot, current_conversation_id, thinking_box, message_history, current_message_index, conversation_dropdown])
        submit_btn.click(fn=respond, 
                       inputs=[msg, chatbot, current_conversation_id, stream_checkbox, model_selector, prompt_template_selector, message_history, current_message_index], 
                       outputs=[msg, chatbot, current_conversation_id, thinking_box, message_history, current_message_index, conversation_dropdown])
        
        # 会话管理事件
        conversation_dropdown.change(fn=load_conversation, 
                                    inputs=conversation_dropdown, 
                                    outputs=[chatbot, title_input, stats_display])
        new_conversation_btn.click(fn=create_new_conversation, 
                                 inputs=None, 
                                 outputs=[chatbot, thinking_box, msg, current_conversation_id, conversation_dropdown, message_history, current_message_index, stats_display])
        delete_conversation_btn.click(fn=delete_conversation, 
                                    inputs=conversation_dropdown, 
                                    outputs=[conversation_dropdown, current_conversation_id, title_input, chatbot, stats_display])
        clear_btn.click(fn=clear_current_chat, 
                       inputs=None, 
                       outputs=[chatbot, msg, thinking_box, message_history, current_message_index, stats_display])
        rename_btn.click(fn=rename_conversation, 
                        inputs=[conversation_dropdown, title_input], 
                        outputs=[conversation_dropdown, stats_display])
        refresh_btn.click(fn=refresh_conversation_dropdown, 
                         inputs=None, 
                         outputs=[conversation_dropdown, stats_display])
        
        # 交互按钮事件绑定
        undo_btn.click(fn=undo_last_message,
                      inputs=[chatbot, message_history, current_message_index],
                      outputs=[chatbot, message_history, current_message_index])
        
        retry_btn.click(fn=retry_last_message,
                       inputs=[chatbot, current_conversation_id, stream_checkbox, model_selector, prompt_template_selector, message_history, current_message_index],
                       outputs=[msg, chatbot, current_conversation_id, thinking_box, message_history, current_message_index, conversation_dropdown])
        
        like_btn.click(fn=like_message,
                      inputs=[chatbot, message_history, current_message_index, current_conversation_id],
                      outputs=[chatbot, message_history, current_message_index, conversation_dropdown, stats_display])
        
        dislike_btn.click(fn=dislike_message,
                         inputs=[chatbot, message_history, current_message_index, current_conversation_id],
                         outputs=[chatbot, message_history, current_message_index, conversation_dropdown, stats_display])
        
        edit_btn.click(fn=edit_message,
                      inputs=[chatbot, message_history, current_message_index],
                      outputs=[edit_msg_box])
        
        # 编辑消息保存事件
        edit_msg_box.submit(fn=save_edited_message,
                           inputs=[edit_msg_box, chatbot, current_conversation_id, stream_checkbox, model_selector, prompt_template_selector, message_history, current_message_index],
                           outputs=[msg, chatbot, current_conversation_id, thinking_box, message_history, current_message_index, edit_msg_box, conversation_dropdown])
        
    return demo

# 主函数
if __name__ == "__main__":
    # 确保会话目录存在
    Path("conversations").mkdir(exist_ok=True)
    
    # 创建并启动界面
    demo = create_interface()
    demo.launch(share=False, server_name="127.0.0.1")