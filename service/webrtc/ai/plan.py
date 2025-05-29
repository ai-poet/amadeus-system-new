"""
AI主动对话行动计划模块
使AI能够根据对话上下文主动规划下一步行动和预生成回复内容
让AI表现出自然的主动性，增强拟人对话体验
"""

import os
import json
import logging
import asyncio
import aiohttp
import random
from typing import List, Dict, Any, Optional, Tuple
from .llm import ai_stream, trim_messages

class ActionPlanner:
    """
    AI主动对话行动计划器
    根据对话历史动态决定AI的下一步主动行动类型和预生成回复内容
    使AI在前端触发时能够自然地发起对话，而不总是被动等待用户输入
    """
    
    def __init__(self, conversation_history: List[Dict[str, str]] = None):
        """
        初始化行动计划器
        
        参数:
            conversation_history: 对话历史记录列表
        """
        self.conversation_history = conversation_history or []
        self.openai_api_key = os.environ.get("LLM_API_KEY", "")
        self.openai_api_base_url = os.environ.get("LLM_BASE_URL", "https://api.openai.com")
        self.next_action = None  # 存储下一步行动
        self.next_response = ""  # 存储预生成的回复内容
    
    async def plan_next_action(self, client=None) -> str:
        """
        规划下一步行动
        
        参数:
            client (OpenAI, optional): OpenAI 客户端，如不指定则创建新的客户端
            
        返回:
            str: 行动类型，可能的值为：
                - continue_topic: 继续当前话题
                - change_topic: 更换话题
                - ask_question: 提问
                - share_memory: 分享记忆
                - express_emotion: 表达情感
        """
        # 如果对话历史为空，默认分享记忆
        if not self.conversation_history:
            self.next_action = "share_memory"
            return "share_memory"
        
        # 准备请求数据
        data = {
            "model": "gpt-4.1-nano",
            "messages": [
                {
                    "role": "system",
                    "content": "你现在是一个智能体的主动行动模块部分，根据当前智能体和用户的上下文，为我的智能体随机选择下一步行动"
                },
                *self.conversation_history
            ],
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "action_response",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {
                            "result": {
                                "type": "string",
                                "enum": [
                                    "continue_topic",
                                    "change_topic",
                                    "ask_question", 
                                    "share_memory",
                                    "express_emotion"
                                ]
                            }
                        },
                        "required": ["result"],
                        "additionalProperties": False
                    }
                }
            }
        }
        
        try:
            # 如果提供了客户端，直接使用客户端
            if client:
                try:
                    response = client.chat.completions.create(**data)
                    content = response.choices[0].message.content
                    try:
                        parsed_content = json.loads(content)
                        action = parsed_content.get('result', 'share_memory')
                        logging.info(f"行动计划生成结果: {action}")
                        self.next_action = action
                        return action
                    except json.JSONDecodeError:
                        logging.error(f"无法解析JSON响应: {content}")
                        self.next_action = "share_memory"
                        return 'share_memory'
                except Exception as e:
                    logging.error(f"客户端调用失败: {e}")
                    # 如果客户端调用失败，回退到HTTP请求
            
            # 配置请求头
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.openai_api_key}"
            }
            
            # 使用aiohttp进行异步HTTP请求
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.openai_api_base_url}/chat/completions",
                    headers=headers,
                    json=data
                ) as response:
                    # 检查响应状态
                    if response.status == 200:
                        response_data = await response.json()
                        content = response_data.get('choices', [{}])[0].get('message', {}).get('content', '{}')
                        
                        try:
                            parsed_content = json.loads(content)
                            action = parsed_content.get('result', 'share_memory')
                            logging.info(f"行动计划生成结果: {action}")
                            self.next_action = action
                            return action
                        except json.JSONDecodeError:
                            logging.error(f"无法解析JSON响应: {content}")
                            self.next_action = "share_memory"
                            return "share_memory"
                    else:
                        response_text = await response.text()
                        logging.error(f"API请求失败: {response.status} {response_text}")
                        self.next_action = "share_memory"
                        return "share_memory"
        except Exception as e:
            logging.error(f"行动计划生成失败: {str(e)}")
            self.next_action = "share_memory"
            return "share_memory"  # 出错时默认返回分享记忆