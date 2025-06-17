"""
流处理工具函数
包含处理 LLM 流式响应的工具函数
"""

import logging
import json
import re
import asyncio
from fastrtc import AdditionalOutputs
from typing import Any, Generator, Tuple, Union, Dict, Optional, List, Callable, Deque
import time
import threading
from concurrent.futures import ThreadPoolExecutor, Future
import queue
from collections import deque

from .async_utils import run_async
from tts.speech import translate_text

def split_text_by_punctuation(text, min_segment_length=15):
    """
    根据标点符号分割文本，并确保每个分段具有最小长度
    
    参数:
        text (str): 要分割的文本
        min_segment_length (int): 分段的最小长度，短于此长度的片段将尝试与相邻片段合并
        
    返回:
        list: 分割后的文本片段列表
    """
    # 标点符号模式：中英文常见标点符号
    punctuation_pattern = r'([,.?!，。？！;；])'
    
    # 第一步：按标点符号分割
    segments = re.split(punctuation_pattern, text)
    
    # 第二步：合并标点和前面的文本
    intermediate_result = []
    i = 0
    while i < len(segments):
        if i + 1 < len(segments) and re.match(punctuation_pattern, segments[i + 1]):
            intermediate_result.append(segments[i] + segments[i + 1])
            i += 2
        else:
            if segments[i].strip():  # 只添加非空片段
                intermediate_result.append(segments[i])
            i += 1
    
    # 第三步：合并长度过短的片段
    result = []
    current_segment = ""
    
    for segment in intermediate_result:
        # 处理第一个片段或当前累积片段为空的情况
        if not current_segment:
            current_segment = segment
            continue
            
        # 如果当前片段结尾有标点符号，表示是一个自然的分割点
        has_ending_punctuation = bool(re.search(r'[,.?!，。？！;；]$', current_segment))
        
        # 判断当前累积片段是否足够长
        if len(current_segment) >= min_segment_length and has_ending_punctuation:
            # 如果当前累积片段足够长且有标点符号，将其添加到结果中并重置
            result.append(current_segment)
            current_segment = segment
        else:
            # 如果当前累积片段不够长或没有标点符号，继续累积
            current_segment += segment
    
    # 处理最后剩余的片段
    if current_segment:
        result.append(current_segment)
    
    # 过滤掉任何仍然太短且不包含有意义内容的片段
    result = [seg for seg in result if len(seg.strip()) >= 2]  # 至少保留两个字符的内容
                
    return result

# 创建线程池执行器
_thread_pool = ThreadPoolExecutor(max_workers=4)
_tts_pool = ThreadPoolExecutor(max_workers=4)  # 专门用于TTS转换的线程池

# 使用共享队列存储TTS音频块，实现真正的流式传输
_audio_chunk_queue = queue.Queue()

def run_emotion_analysis_in_thread(run_predict_emotion, text, client):
    """
    在线程池中运行情感分析以避免阻塞主流程
    
    参数:
        run_predict_emotion: 情感分析函数
        text: 要分析的文本
        client: 客户端对象
        
    返回:
        情感分析结果
    """
    try:
        return run_async(run_predict_emotion, text, client)
    except Exception as e:
        logging.error(f"情感分析出错: {e}")
        return None

def run_tts_in_thread(text_to_speech_stream, segment, voice, segment_id):
    """
    在线程池中运行TTS转换，并将音频块实时添加到共享队列
    
    参数:
        text_to_speech_stream: TTS函数
        segment: 要转换的文本段落
        voice: 语音配置
        segment_id: 段落ID，用于标识音频块所属段落
        
    返回:
        None (结果通过队列传递)
    """
    try:
        logging.info(f"开始TTS转换 - 段落ID: {segment_id}, 文本长度: {len(segment)}, 文本预览: {segment[:50]}...")
        
        chunk_count = 0
        start_time = time.time()
        
        for audio_chunk in text_to_speech_stream(segment, voice=voice):
            chunk_count += 1
            # 直接将音频块放入共享队列，实现实时流式传输
            # 添加时间戳和块序号以便调试
            chunk_with_meta = (audio_chunk[0], audio_chunk[1])  # (sample_rate, audio_array)
            _audio_chunk_queue.put((segment_id, chunk_with_meta))
            
        end_time = time.time()
        logging.info(f"TTS转换完成 - 段落ID: {segment_id}, 生成音频块数量: {chunk_count}, 耗时: {end_time - start_time:.2f}秒")
        
    except Exception as e:
        logging.error(f"TTS转换出错 - 段落ID: {segment_id}, 错误: {e}, 文本: {segment[:30]}...")
    finally:
        # 添加一个None标记表示这个段落的TTS已经完成
        _audio_chunk_queue.put((segment_id, None))
        logging.info(f"TTS任务结束标记已发送 - 段落ID: {segment_id}")

def check_and_process_tts_tasks(
    pending_tts_tasks: Dict[str, int],  # 改为int，value表示已产生的chunk数量
    audio_queue: Deque[Tuple[str, Any]], 
    segment_order: List[str],
    current_output_segment_id: List[Optional[str]]
):
    """检查并处理已完成的TTS任务，从共享队列获取音频块"""
    # 从共享队列中获取所有可用的音频块
    temp_chunks = []  # 临时存储新获取的音频块
    completed_segments = []  # 记录正常完成的段落 (segment_id, chunk_count)
    failed_segments = []  # 记录失败（无音频块）的段落
    
    while not _audio_chunk_queue.empty():
        try:
            segment_id, audio_chunk = _audio_chunk_queue.get_nowait()
            
            # 如果是None标记，表示该段落的TTS已完成
            if audio_chunk is None:
                # TTS流标记完成，pop出来拿到它的chunk计数
                count = pending_tts_tasks.pop(segment_id, 0)
                if count == 0:
                    # 真正失败，没有产生任何音频块
                    failed_segments.append(segment_id)
                else:
                    # 正常完成，记录产生的块数
                    completed_segments.append((segment_id, count))
            else:
                # 真正的音频块，增加计数
                pending_tts_tasks[segment_id] = pending_tts_tasks.get(segment_id, 0) + 1
                temp_chunks.append((segment_id, audio_chunk))
        except queue.Empty:
            break
    
    # 处理失败的段落：从segment_order中彻底移除
    for failed_segment_id in failed_segments:
        if failed_segment_id in segment_order:
            segment_order.remove(failed_segment_id)
            logging.warning(f"段落 {failed_segment_id} 的TTS完成但真没有音频块，已从处理序列中移除")
            
            # 如果当前输出段落指向失败的段落，需要切换到下一个
            if current_output_segment_id[0] == failed_segment_id:
                _skip_to_next_segment(segment_order, current_output_segment_id)
    
    # 记录正常完成的段落
    if completed_segments:
        completed_ids = [seg_id for seg_id, count in completed_segments]
        logging.info(f"TTS正常完成的段落: {completed_ids}, 剩余待处理段落: {list(pending_tts_tasks.keys())}")
    
    # 按照segment_order对新获取的音频块进行排序，然后添加到队列
    if temp_chunks:
        # 创建一个映射，将segment_id映射到它在segment_order中的位置
        segment_position = {seg_id: idx for idx, seg_id in enumerate(segment_order)}
        
        # 按照在segment_order中的位置对音频块进行排序
        temp_chunks.sort(key=lambda x: segment_position.get(x[0], float('inf')))
        
        # 记录音频块信息
        chunk_info = [(seg_id, f"pos_{segment_position.get(seg_id, 'unknown')}") for seg_id, _ in temp_chunks]
        logging.info(f"处理音频块: {chunk_info}, 当前队列长度: {len(audio_queue)}")
        
        # 将排序后的音频块添加到输出队列
        for segment_id, audio_chunk in temp_chunks:
            audio_queue.append((segment_id, audio_chunk))
    
    return None

def yield_ready_audio_chunks(audio_queue, segment_order, current_output_segment_id, force_all=False, pending_tts_tasks=None):
    """按照段落顺序输出准备好的音频块"""
    # 如果当前没有正在输出的段落，取第一个
    if current_output_segment_id[0] is None and segment_order:
        current_output_segment_id[0] = segment_order[0]
        logging.info(f"开始输出第一个段落: {current_output_segment_id[0]}")
    
    yielded_count = 0
    
    # 处理队列中的音频块 - 使用简化逻辑
    while audio_queue:
        head_seg, chunk = audio_queue[0]
        
        # 如果是当前输出段落的块或强制输出所有内容，则yield出去
        if head_seg == current_output_segment_id[0] or force_all:
            audio_queue.popleft()
            yielded_count += 1
            yield (head_seg, chunk)
        else:
            # 检查队首段落是否在 segment_order 中且排在当前段落之后
            if (head_seg in segment_order and 
                current_output_segment_id[0] in segment_order and
                segment_order.index(head_seg) > segment_order.index(current_output_segment_id[0])):
                # 队首段落在当前段落之后，跳到下一个段落
                logging.info(f"队首段落 {head_seg} 在当前段落 {current_output_segment_id[0]} 之后，跳过当前段落")
                _skip_to_next_segment(segment_order, current_output_segment_id)
                continue  # 重新开始循环，用新的当前段落ID检查
            elif head_seg not in segment_order:
                # 队首段落不在 segment_order 中，这是过期的音频块，直接丢弃
                logging.warning(f"发现过期音频块，段落ID: {head_seg}，直接丢弃")
                audio_queue.popleft()
                continue
            else:
                # 正常等待当前段落完成
                logging.info(f"等待当前段落 {current_output_segment_id[0]} 完成，队首段落: {head_seg}")
                break
        
        # 如果当前段落的所有块都已输出，检查是否需要切换到下一个段落
        if not audio_queue or audio_queue[0][0] != current_output_segment_id[0]:
            _skip_to_next_segment(segment_order, current_output_segment_id)
    
    if yielded_count > 0:
        logging.info(f"本次输出了 {yielded_count} 个音频块，剩余队列长度: {len(audio_queue)}")
    
    return None

def _skip_to_next_segment(segment_order, current_output_segment_id):
    """跳转到下一个段落的辅助函数"""
    if segment_order and current_output_segment_id[0] in segment_order:
        idx = segment_order.index(current_output_segment_id[0])
        if idx + 1 < len(segment_order):
            old_segment_id = current_output_segment_id[0]
            current_output_segment_id[0] = segment_order[idx + 1]
            logging.info(f"切换到下一个段落: {old_segment_id} -> {current_output_segment_id[0]}")
        else:
            logging.info(f"当前段落 {current_output_segment_id[0]} 是最后一个段落")

def process_llm_stream(
    client, 
    messages, 
    model, 
    siliconflow_config, 
    voice_output_language=None, 
    text_output_language='zh',
    is_same_language=True, 
    run_predict_emotion=None, 
    ai_stream=None, 
    text_to_speech_stream=None,
    max_tokens=None,
    max_context_length=None,
    min_segment_length=15,  # 添加最小片段长度参数
):
    """
    处理 LLM 的流式响应，使用统一的处理逻辑并支持基于标点符号的分段
    
    参数:
        client: OpenAI 客户端
        messages: 消息历史
        model: 使用的模型名称
        siliconflow_config: 语音合成配置
        voice_output_language: 语音输出语言
        is_same_language: 文本和语音是否为同一语言
        run_predict_emotion: 情感分析函数
        ai_stream: AI 流式生成函数
        text_to_speech_stream: 文本转语音流函数
        max_tokens: 最大生成令牌数
        max_context_length: 上下文最大消息数
        min_segment_length: 分段的最小长度，短于此长度的片段将尝试与相邻片段合并
        
    返回:
        生成器，产生音频块和额外输出
    """
    full_response = ""
    full_response_for_client_segments = [] # New initialization
    current_buffer = ""
    processed_length = 0
    
    # 清空音频块队列，避免之前的残留
    while not _audio_chunk_queue.empty():
        try:
            _audio_chunk_queue.get_nowait()
        except queue.Empty:
            break
    
    # 存储正在进行的TTS任务（现在只用于跟踪哪些段落正在处理）
    pending_tts_tasks: Dict[str, int] = {}
    
    # 存储待输出的音频块队列
    audio_queue: Deque[Tuple[str, Any]] = deque()  # (segment_id, audio_chunk)
    
    # 为了确保流顺序，维护一个处理顺序列表
    segment_order: List[str] = []
    
    # 当前正在输出的段落ID - 使用列表以便在函数间传递状态
    current_output_segment_id = [None]
    
    # 最后一个段落的ID
    last_segment_id = None
    
    # 上次检查TTS任务完成情况的时间
    last_check_time = time.time()
    
    # 标记LLM是否完成
    llm_completed = False
    
    # 添加延迟机制，避免音频块挤在一起
    last_audio_yield_time = 0
    min_audio_interval = 0.01  # 最小音频块间隔 10ms
    
    for text_chunk, current_full_response in ai_stream(client, messages, model=model, max_tokens=max_tokens, max_context_length=max_context_length):        
        full_response = current_full_response
        current_buffer += text_chunk
        
        # 使用优化后的分段函数，传入最小片段长度
        segments = split_text_by_punctuation(current_buffer, min_segment_length)
        
        # 定期检查TTS任务完成情况，不仅在有新分段时
        current_time = time.time()
        if current_time - last_check_time > 0.05:  # 每50毫秒检查一次，更频繁地检查
            last_check_time = current_time
            
            # 检查已完成的TTS任务并获取新的音频块
            check_and_process_tts_tasks(
                pending_tts_tasks, 
                audio_queue, 
                segment_order,
                current_output_segment_id
            )
            
            # 输出准备好的音频块，添加间隔控制
            for output in yield_ready_audio_chunks(audio_queue, segment_order, current_output_segment_id, pending_tts_tasks=pending_tts_tasks):
                if isinstance(output, AdditionalOutputs):
                    yield output
                else:
                    # 控制音频块输出间隔，避免挤在一起
                    current_time = time.time()
                    if current_time - last_audio_yield_time < min_audio_interval:
                        time.sleep(min_audio_interval - (current_time - last_audio_yield_time))
                    
                    yield output[1]  # 实际音频块
                    last_audio_yield_time = time.time()
        
        if len(segments) > 1:  # 如果有多个分段
            segments_to_process = segments[:-1]
            current_buffer = segments[-1]
            
            for segment in segments_to_process:
                if segment.strip():
                    # 为段落生成唯一ID
                    segment_id = f"segment_{time.time()}_{len(segment)}"
                    segment_order.append(segment_id)
                    
                    # 先初始化计数为0，必须在submit之前
                    pending_tts_tasks[segment_id] = 0

                    # Submit TTS to _tts_pool early
                    _tts_pool.submit(
                        run_tts_in_thread,
                        text_to_speech_stream,
                        segment, # Original segment for TTS
                        siliconflow_config.get("voice"),
                        segment_id
                    )

                    translated_segment = None
                    if voice_output_language and text_output_language and voice_output_language != text_output_language:
                        if segment.strip(): # Ensure there's text to translate
                            translation_future = _thread_pool.submit(translate_text, segment, target_language=text_output_language, source_language=voice_output_language)
                            try:
                                translated_segment = translation_future.result(timeout=5) # Wait for this specific translation with a timeout
                            except Exception as e:
                                logging.error(f"Translation for segment failed or timed out: {e}")
                                translated_segment = None # Ensure it's None if error or timeout
                    
                    # 决定要在流式事件中发送的文本：如果有翻译就用翻译，否则用原文
                    stream_text = translated_segment if (translated_segment and translated_segment.strip()) else segment
                    
                    event_data = {"type": "llm_stream", "data": stream_text}
                    # 如果进行了翻译，也保留原文作为参考
                    if translated_segment and translated_segment.strip():
                        event_data["original"] = segment
                    # Yield llm_stream event
                    logging.info(f"Yielding llm_stream event_data: {event_data}")
                    yield AdditionalOutputs(json.dumps(event_data))

                    # Append to full_response_for_client_segments
                    if translated_segment and translated_segment.strip():
                        full_response_for_client_segments.append(translated_segment)
                    else:
                        full_response_for_client_segments.append(segment)
                    
                    # 立即检查是否有新的音频块可用
                    check_and_process_tts_tasks(
                        pending_tts_tasks, 
                        audio_queue, 
                        segment_order,
                        current_output_segment_id
                    )
                    
                    # 立即输出就绪的音频块，添加间隔控制
                    for output in yield_ready_audio_chunks(audio_queue, segment_order, current_output_segment_id, pending_tts_tasks=pending_tts_tasks):
                        if isinstance(output, AdditionalOutputs):
                            yield output
                        else:
                            # 控制音频块输出间隔，避免挤在一起
                            current_time = time.time()
                            if current_time - last_audio_yield_time < min_audio_interval:
                                time.sleep(min_audio_interval - (current_time - last_audio_yield_time))
                            
                            yield output[1]  # 实际音频块
                            last_audio_yield_time = time.time()
    
    # LLM已完成生成，标记完成状态
    llm_completed = True
    
    # 处理最后可能剩余的内容
    if current_buffer.strip():
        last_segment_text = current_buffer.strip() # Use a new variable for clarity
        last_segment_id = f"last_segment_{time.time()}"
        segment_order.append(last_segment_id)
        
        # 先初始化计数为0，必须在submit之前
        pending_tts_tasks[last_segment_id] = 0

        # Submit TTS to _tts_pool early for the last segment
        _tts_pool.submit(
            run_tts_in_thread,
            text_to_speech_stream,
            last_segment_text, # Use the stripped text
            siliconflow_config.get("voice"),
            last_segment_id
        )

        translated_last_segment = None
        if voice_output_language and text_output_language and voice_output_language != text_output_language:
            if last_segment_text: # Ensure there's text to translate
                translation_future = _thread_pool.submit(translate_text, last_segment_text, target_language=text_output_language, source_language=voice_output_language)
                try:
                    translated_last_segment = translation_future.result(timeout=5) # Wait for this specific translation with a timeout
                except Exception as e:
                    logging.error(f"Translation for last segment failed or timed out: {e}")
                    translated_last_segment = None # Ensure it's None if error or timeout

        # 决定要在流式事件中发送的文本：如果有翻译就用翻译，否则用原文
        stream_text = translated_last_segment if (translated_last_segment and translated_last_segment.strip()) else last_segment_text
        
        event_data = {"type": "llm_stream", "data": stream_text}
        # 如果进行了翻译，也保留原文作为参考
        if translated_last_segment and translated_last_segment.strip():
            event_data["original"] = last_segment_text
        # Yield final llm_stream event
        logging.info(f"Yielding final llm_stream event_data: {event_data}")
        yield AdditionalOutputs(json.dumps(event_data))

        # Append to full_response_for_client_segments for the final segment
        if translated_last_segment and translated_last_segment.strip():
            full_response_for_client_segments.append(translated_last_segment)
        else:
            full_response_for_client_segments.append(last_segment_text)
    
    # 在LLM完成后立即进行情感分析，不等待TTS
    if run_predict_emotion and llm_completed:
        try:
            # 对完整响应进行情感分析
            emotion_result = run_emotion_analysis_in_thread(run_predict_emotion, full_response, client)
            emotion_json = json.dumps({
                "type": "emotion_response", 
                "data": f"{emotion_result}", 
                "segment_id": "full_response"
            })
            yield AdditionalOutputs(emotion_json)
        except Exception as e:
            logging.error(f"情感分析出错: {e}")
    
    # 将文本包装成JSON对象，表示这是LLM返回的完整响应
    final_full_response_for_client = "".join(full_response_for_client_segments)
    
    # 对完整响应进行统一翻译（如果需要翻译）
    unified_translation = None
    if voice_output_language and text_output_language and voice_output_language != text_output_language:
        if full_response.strip():  # 确保有内容需要翻译
            try:
                logging.info(f"开始对完整响应进行统一翻译，原文长度: {len(full_response)}")
                translation_future = _thread_pool.submit(translate_text, full_response, target_language=text_output_language, source_language=voice_output_language)
                unified_translation = translation_future.result(timeout=10)  # 给统一翻译更长的超时时间
                logging.info(f"统一翻译完成，译文长度: {len(unified_translation) if unified_translation else 0}")
            except Exception as e:
                logging.error(f"统一翻译失败: {e}")
                unified_translation = None
    
    # 构建llm_response事件数据
    llm_response_data = {"type": "llm_response", "data": final_full_response_for_client}
    if unified_translation and unified_translation.strip():
        llm_response_data["data"] = unified_translation
        llm_response_data["original"] = full_response
    
    llm_response_json = json.dumps(llm_response_data)
    logging.info(f"Yielding llm_response event_data: {llm_response_json}")
    yield AdditionalOutputs(llm_response_json)
    
    # 继续处理TTS
    # 等待所有TTS任务完成，添加重试机制防止无限循环
    consecutive_no_progress_count = 0
    max_no_progress_iterations = 50  # 最多50次无进展迭代
    
    while pending_tts_tasks:
        # 记录循环前的状态
        initial_pending_count = len(pending_tts_tasks)
        initial_queue_size = len(audio_queue)
        
        # 检查已完成的TTS任务并获取新的音频块
        check_and_process_tts_tasks(
            pending_tts_tasks, 
            audio_queue, 
            segment_order,
            current_output_segment_id
        )
        
        # 输出准备好的音频块，添加间隔控制
        yielded_audio_count = 0
        for output in yield_ready_audio_chunks(audio_queue, segment_order, current_output_segment_id, pending_tts_tasks=pending_tts_tasks):
            if isinstance(output, AdditionalOutputs):
                yield output
            else:
                yielded_audio_count += 1
                # 控制音频块输出间隔，避免挤在一起
                current_time = time.time()
                if current_time - last_audio_yield_time < min_audio_interval:
                    time.sleep(min_audio_interval - (current_time - last_audio_yield_time))
                
                yield output[1]  # 实际音频块
                last_audio_yield_time = time.time()
        
        # 检查是否有进展
        final_pending_count = len(pending_tts_tasks)
        final_queue_size = len(audio_queue)
        
        has_progress = (
            initial_pending_count > final_pending_count or  # 有任务完成
            initial_queue_size < final_queue_size or       # 队列有新音频块
            yielded_audio_count > 0                        # 输出了音频块
        )
        
        if has_progress:
            consecutive_no_progress_count = 0
        else:
            consecutive_no_progress_count += 1
            
        # 如果连续无进展次数过多，强制结束防止无限循环
        if consecutive_no_progress_count >= max_no_progress_iterations:
            logging.warning(f"检测到可能的无限循环，连续{max_no_progress_iterations}次无进展。强制结束TTS处理。")
            logging.warning(f"剩余未完成段落: {list(pending_tts_tasks.keys())}")
            break
        
        # 如果仍有未完成的任务，等待一小段时间
        if pending_tts_tasks:
            time.sleep(0.05)  # 等待50毫秒再检查
    
    # 确保所有音频块都已经输出，添加间隔控制
    for output in yield_ready_audio_chunks(audio_queue, segment_order, current_output_segment_id, force_all=True):
        if isinstance(output, AdditionalOutputs):
            yield output
        else:
            # 控制音频块输出间隔，避免挤在一起
            current_time = time.time()
            if current_time - last_audio_yield_time < min_audio_interval:
                time.sleep(min_audio_interval - (current_time - last_audio_yield_time))
            
            yield output[1]  # 实际音频块
            last_audio_yield_time = time.time()
    
    # 在yield完所有内容后，再yield一次full_response字符串
    # 这样调用者就可以获取完整的响应文本
    yield full_response          