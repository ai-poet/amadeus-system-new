"""
情感分析模块
根据文本内容预测情感状态
"""

import os
import json
import logging
import aiohttp
from dotenv import load_dotenv
from openai import OpenAI

# 加载环境变量
load_dotenv()

# 从环境变量获取默认 API 密钥和基础 URL
DEFAULT_OPENAI_API_KEY = os.getenv("LLM_API_KEY", "")
DEFAULT_OPENAI_API_BASE_URL = os.getenv("LLM_BASE_URL", "")

async def predict_emotion(message, client=None):
    """
    根据给定的消息文本预测情感
    
    参数:
        message (str): 用于情感分析的消息文本
        client (OpenAI, optional): OpenAI 客户端，如不指定则创建新的客户端
        
    返回:
        str: 预测的情感类型，如'neutral'、'anger'、'joy'等
    """
    try:
        api_key = DEFAULT_OPENAI_API_KEY
        base_url = DEFAULT_OPENAI_API_BASE_URL
        
        # 准备请求数据
        data = {
            "model": "gpt-4.1-nano",
            "messages": [
                {
                    "role": "system",
                    "content": "你现在是一个虚拟形象的动作驱动器，你需要根据输入的虚拟形象的语言，驱动虚拟形象的动作和表情，请尽量输出得随机并丰富一些"
                },
                {
                    "role": "user",
                    "content": message
                }
            ],
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "motion_response",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {
                            "result": {
                                "type": "string",
                                "enum": ["neutral", "anger", "joy", "sadness", "shy", "shy2", "smile1", "smile2", "unhappy"]
                            }
                        },
                        "required": ["result"],
                        "additionalProperties": False
                    }
                }
            }
        }
        
        # 如果提供了客户端，直接使用客户端
        if client:
            try:
                response = client.chat.completions.create(**data)
                content = response.choices[0].message.content
                try:
                    parsed_content = json.loads(content)
                    emotion = parsed_content.get('result', 'neutral')
                    logging.info(f"情感分析结果: {emotion}")
                    return emotion
                except json.JSONDecodeError:
                    logging.error(f"无法解析JSON响应: {content}")
                    return 'neutral'
            except Exception as e:
                logging.error(f"客户端调用失败: {e}")
                # 如果客户端调用失败，回退到HTTP请求
        
        # 使用aiohttp进行异步HTTP请求
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {api_key}'
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{base_url}/chat/completions",
                headers=headers,
                json=data
            ) as response:
                # 检查响应状态
                if response.status == 200:
                    response_data = await response.json()
                    content = response_data.get('choices', [{}])[0].get('message', {}).get('content', '{}')
                    
                    try:
                        parsed_content = json.loads(content)
                        emotion = parsed_content.get('result', 'neutral')
                        logging.info(f"情感分析结果: {emotion}")
                        return emotion
                    except json.JSONDecodeError:
                        logging.error(f"无法解析JSON响应: {content}")
                        return 'neutral'
                else:
                    response_text = await response.text()
                    logging.error(f"API请求失败: {response.status} {response_text}")
                    return 'neutral'
            
    except Exception as e:
        logging.error(f"预测情感时出错: {e}")
        return 'neutral' 