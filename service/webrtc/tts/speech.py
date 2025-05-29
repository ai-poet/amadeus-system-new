"""
文本转语音模块
提供将文本转换为语音的功能
"""

import logging
import os
import requests
import numpy as np
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor
import json
from openai import OpenAI

# 加载环境变量
load_dotenv()

# 从环境变量获取默认的 API 密钥和语音模型
DEFAULT_SILICONFLOW_API_KEY = os.getenv("SILICONFLOW_API_KEY", "")
DEFAULT_SILICONFLOW_VOICE = os.getenv("SILICONFLOW_VOICE", "speech:siliconflow-kurisu:clzv7bjjm041fufyct2z0setm:mphrsbbmvrjfophbsted")

# 从环境变量获取OpenAI API密钥
LLM_BASE_URL = os.getenv("LLM_BASE_URL")
LLM_API_KEY = os.getenv("LLM_API_KEY")

# 创建OpenAI客户端
client = OpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL)

# 创建一个模块级别的线程池用于翻译任务
_translate_pool = ThreadPoolExecutor(max_workers=2)

def translate_text(text, target_language, source_language='zh'):
    """
    使用OpenAI的GPT-4.1-nano模型将文本从源语言翻译到目标语言
    
    参数:
        text (str): 要翻译的文本
        target_language (str): 目标语言代码
        source_language (str): 源语言代码，默认为'zh'
        
    返回:
        str: 翻译后的文本，如果翻译失败则返回原文本
    """
    if not text or not text.strip():
        return text
    
    # 检查API密钥是否有效
    if not LLM_API_KEY:
        logging.error("缺少OpenAI API密钥，无法进行文本翻译")
        return text
    
    try:
        # 构建语言名称映射
        language_map = {
            'zh': '中文',
            'en': '英语',
            'ja': '日语'
        }
        
        source_lang_name = language_map.get(source_language, source_language)
        target_lang_name = language_map.get(target_language, target_language)
        
        # 构建翻译提示
        system_prompt = f'你是一个专业的翻译助手，负责将{source_lang_name}翻译成{target_lang_name}。请直接提供翻译结果，不要添加任何解释或额外内容。'
        user_prompt = f"请将以下{source_lang_name}文本翻译成{target_lang_name}，只返回翻译结果，不要添加任何解释或额外内容：\n\n{text}"
        
        # 使用OpenAI SDK发送请求
        response = client.chat.completions.create(
            model="gpt-4.1-nano",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.3  # 使用较低的温度以获得更确定性的翻译
        )
        
        # 获取翻译结果
        translated_text = response.choices[0].message.content.strip()
        logging.info(f"文本翻译成功: {text[:30]}... -> {translated_text[:30]}...")
        return translated_text
    except Exception as e:
        logging.error(f"使用OpenAI SDK进行翻译时出错: {e}")
        return text

def text_to_speech_stream(text, voice=None, sample_rate=32000, api_key=None):
    """
    将文本转换为语音流
    
    参数:
        text (str): 要转换为语音的文本
        voice (str): 使用的声音模型，如不指定则使用环境变量中的设置
        sample_rate (int): 采样率，默认为32000Hz
        api_key (str): SiliconFlow API 密钥，如不指定则使用环境变量中的设置
        
    返回:
        generator: 生成(sample_rate, audio_array)元组的生成器
    """
    if not text or not text.strip():
        logging.warning("文本为空，不进行转换")
        return
    
    # 如果未指定voice参数，则使用环境变量中的设置
    if voice is None:
        voice = DEFAULT_SILICONFLOW_VOICE
    
    # 如果未指定api_key参数，则使用环境变量中的设置
    if api_key is None:
        api_key = DEFAULT_SILICONFLOW_API_KEY
    
    # 检查API密钥是否有效
    if not api_key:
        logging.error("缺少SiliconFlow API密钥，无法进行文本转语音")
        return
        
    # 设置请求头
    headers = {
        'Authorization': f'Bearer {api_key}',
        'Content-Type': 'application/json',
    }
    
    # 设置请求数据
    data = {
        'model': 'FunAudioLLM/CosyVoice2-0.5B',
        'input': text,
        'voice': voice,
        'sample_rate': sample_rate,
        'response_format': 'pcm',
    }
    
    try:
        # 发送请求，获取流式响应
        response = requests.post(
            'https://api.siliconflow.cn/v1/audio/speech',
            json=data,
            headers=headers,
            stream=True
        )
        
        # 处理流式响应
        if response.status_code == 200:
            # 创建一个缓冲区来存储接收到的数据
            buffer = bytearray() # 创建一个缓冲区来存储接收到的数据
            
            # 处理流式响应的每个块
            for chunk in response.iter_content(chunk_size=None): # chunk_size=None以便获取任意大小的块
                if chunk:
                    # 将新接收的块数据追加到缓冲区
                    buffer.extend(chunk)
                    
                    # 计算缓冲区中完整的样本数 (每个样本2字节)
                    num_samples = len(buffer) // 2
                    
                    if num_samples > 0:
                        # 提取所有完整的样本数据
                        process_len = num_samples * 2
                        data_to_process = buffer[:process_len]
                        
                        # 从缓冲区移除已提取的数据
                        buffer = buffer[process_len:]
                        
                        try:
                            # 将字节数据转换为16位整数，然后转换为-1到1之间的浮点数
                            audio_array = np.frombuffer(data_to_process, dtype=np.int16).astype(np.float32) / 32768.0
                            yield (sample_rate, audio_array)
                        except Exception as e:
                            logging.error(f"处理音频数据时出错: {e}")
            
            # 处理循环结束后缓冲区中可能剩余的完整样本
            # (通常情况下，如果API正确结束流，这里不应该有太多数据，但为了健壮性处理)
            if len(buffer) >= 2 : # 确保至少有一个完整样本
                num_samples = len(buffer) // 2
                if num_samples > 0:
                    process_len = num_samples * 2
                    data_to_process = buffer[:process_len]
                    # 此时缓冲区中不足一个样本的部分将被丢弃，这是合理的，因为流已结束
                    buffer = buffer[process_len:] # 清空或保留不足一个样本的部分
                    try:
                        audio_array = np.frombuffer(data_to_process, dtype=np.int16).astype(np.float32) / 32768.0
                        yield (sample_rate, audio_array)
                    except Exception as e:
                        logging.error(f"处理剩余音频数据时出错: {e}")
        else:
            logging.error(f"SILICONFLOW API返回错误: {response.status_code} {response.text}")
    except Exception as e:
        logging.error(f"调用文本转语音API时出错: {e}")