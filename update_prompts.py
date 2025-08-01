#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
更新提示词脚本
从prompts文件夹中的Markdown文件读取内容，并更新prompt.json
"""

import os
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple

def extract_system_prompt_from_markdown(markdown_content: str) -> str:
    """
    从Markdown文件中提取系统提示词
    整个Markdown文件的内容都作为system prompt
    """
    # 直接返回整个文件内容作为system prompt
    return markdown_content.strip()

def extract_description_from_filename(filename: str) -> str:
    """
    从文件名提取描述
    移除.md扩展名作为描述
    """
    return filename.replace('.md', '')

def read_markdown_files(prompts_dir: str) -> Dict[str, Dict[str, str]]:
    """
    读取prompts文件夹中的所有Markdown文件
    返回格式化的提示词字典
    """
    prompts = {}
    prompts_path = Path(prompts_dir)
    
    if not prompts_path.exists():
        print(f"错误：目录 {prompts_dir} 不存在")
        return prompts
    
    # 读取所有.md文件
    markdown_files = list(prompts_path.glob('*.md'))
    
    for md_file in markdown_files:
        try:
            with open(md_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 提取系统提示词
            system_prompt = extract_system_prompt_from_markdown(content)
            
            # 从文件名提取描述
            description = extract_description_from_filename(md_file.name)
            
            # 添加到提示词字典
            prompts[description] = {
                "system": system_prompt,
                "description": description
            }
            
            print(f"✓ 已处理: {md_file.name}")
            
        except Exception as e:
            print(f"✗ 处理文件 {md_file.name} 时出错: {e}")
    
    return prompts

def update_prompt_json(prompt_json_path: str, new_prompts: Dict[str, Dict[str, str]]):
    """
    更新prompt.json文件
    保留现有的default和reasoner，添加新的提示词
    """
    # 读取现有的prompt.json
    existing_prompts = {}
    if os.path.exists(prompt_json_path):
        try:
            with open(prompt_json_path, 'r', encoding='utf-8') as f:
                existing_prompts = json.load(f)
            print(f"✓ 已读取现有文件: {prompt_json_path}")
        except Exception as e:
            print(f"✗ 读取现有文件时出错: {e}")
    
    # 保留default和reasoner
    updated_prompts = {
        "default": existing_prompts.get("default", {
            "system": "你是DeepSeek Chat，一个由DeepSeek开发的人工智能助手，擅长对话和思考。",
            "description": "默认系统提示词"
        }),
        "reasoner": existing_prompts.get("reasoner", {
            "system": "你是DeepSeek Reasoner，一个专门用于复杂推理和问题解决的人工智能助手。你擅长逻辑推理、数学计算、代码分析、多步骤问题解决等任务。在回答问题时，你会先进行深入的思考和分析，然后提供清晰、准确的答案。请确保你的推理过程逻辑清晰，步骤明确。",
            "description": "DeepSeek推理专家"
        })
    }
    
    # 添加新的提示词
    for key, value in new_prompts.items():
        updated_prompts[key] = value
    
    # 写入更新后的文件
    try:
        with open(prompt_json_path, 'w', encoding='utf-8') as f:
            json.dump(updated_prompts, f, ensure_ascii=False, indent=4)
        print(f"✓ 已更新文件: {prompt_json_path}")
        print(f"✓ 总共包含 {len(updated_prompts)} 个提示词模板")
    except Exception as e:
        print(f"✗ 写入文件时出错: {e}")

def main():
    """
    主函数
    """
    print("🔄 开始更新提示词...")
    
    # 设置路径
    current_dir = Path(__file__).parent
    prompts_dir = current_dir / "prompts"
    prompt_json_path = current_dir / "prompt.json"
    
    print(f"📁 提示词目录: {prompts_dir}")
    print(f"📄 输出文件: {prompt_json_path}")
    
    # 读取Markdown文件
    print("\n📖 读取Markdown文件...")
    new_prompts = read_markdown_files(prompts_dir)
    
    if not new_prompts:
        print("❌ 没有找到任何Markdown文件")
        return
    
    # 更新prompt.json
    print("\n💾 更新prompt.json...")
    update_prompt_json(prompt_json_path, new_prompts)
    
    print("\n✅ 更新完成！")
    print("\n📋 可用的提示词模板:")
    for key, value in new_prompts.items():
        print(f"  - {key}: {value['description']}")

if __name__ == "__main__":
    main() 