"""
文本转语音模块
提供将文本转换为语音的功能
"""

from .speech import text_to_speech_stream, translate_text

__all__ = ['text_to_speech_stream', 'translate_text'] 