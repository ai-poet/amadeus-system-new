import styles from './index.module.less'
import { useEffect, useRef, useState, useCallback } from 'react'
import { useTranslation } from 'react-i18next'
import ParticleBackground from './components/ParticleBackground'
import Live2dModel from '@/components/Live2dModel'
import { observer } from 'mobx-react'
import VideoChat from '@/components/VideoChat'
import ChatHistory from '@/components/ChatHistory'
import Toolbar from '@/components/Toolbar'
import LoginOverlay from '@/components/LoginOverlay'
import ConfigPanel from '@/components/ConfigPanel'
import StartDialog from '@/components/StartDialog'
import { useWebRTC } from '@/hooks/useWebRTC'
import { useStore } from '@/store/storeProvider'
import { MessageTypes } from '@/constants'

/**
 * 聊天信息接口定义
 * @interface ChatMessage
 * @property {string} role - 消息角色：用户或助手
 * @property {string} content - 消息内容
 * @property {number} timestamp - 消息时间戳
 */
interface ChatMessage {
  role: 'user' | 'assistant'; // 消息角色：用户或助手
  content: string;           // 消息内容
  timestamp: number;         // 消息时间戳（毫秒）
}

// 使用API前缀，将请求通过Node服务转发
export const API_BASE_URL = "/api";

/**
 * 语音助手组件
 * 实现了一个基于WebRTC的AI语音对话系统，集成了Live2D模型呈现
 */
const VoiceAssistant = observer(() => {
  const { t } = useTranslation();
  // 音频元素引用，用于播放AI回复的声音
  const audioRef = useRef<HTMLAudioElement>(null);
  // 是否正在加载(等待AI响应)
  const [isLoading, setIsLoading] = useState(false)
  // 是否正在说话
  const [isSpeaking, setIsSpeaking] = useState(false)
  // 是否显示登录蒙层
  const [showOverlay, setShowOverlay] = useState(true)
  // 是否开启视频
  const [isVideoOn, setIsVideoOn] = useState(false)
  // 聊天历史记录
  const [chatHistory, setChatHistory] = useState<ChatMessage[]>([]);
  // 是否显示历史记录面板
  const [showHistory, setShowHistory] = useState(false);
  // 最新的AI消息内容
  const [latestAiMessage, setLatestAiMessage] = useState<string>('');
  // Live2D模型是否已加载完成
  const [isModelReady, setIsModelReady] = useState(false);
  // 是否显示配置面板
  const [showConfig, setShowConfig] = useState(false);
  // 是否为首次登录
  const [isFirstLogin, setIsFirstLogin] = useState(false);
  // 是否正在流式传输
  const [isStreaming, setIsStreaming] = useState(false);
  // 是否显示开始对话框
  const [showStartDialog, setShowStartDialog] = useState(false);
  // 从全局状态中获取Live2D模型状态
  const { live2dStore } = useStore();

  /**
   * 使用WebRTC Hook，建立与服务器的实时音频通信
   */
  const {
    connect,         // 连接WebRTC的函数
    disconnect,      // 断开WebRTC连接的函数
    webrtcId,        // WebRTC连接ID
    isConnected,     // 是否已连接
    toggleMicrophone, // 切换麦克风状态的函数
    isMicrophoneMuted // 麦克风是否静音
  } = useWebRTC({
    apiBaseUrl: API_BASE_URL,
    // 当收到音频流时的处理函数
    onAudioStream: (stream) => {
      if (audioRef.current) {
        audioRef.current.srcObject = stream;
      }
    },
    // 当收到完整的LLM响应时的处理函数
    onLLMResponse: (text) => {
      console.log('收到LLM完整响应:', text);
      if (text) {
        // 更新聊天历史并保存到本地存储
        setChatHistory(prevHistory => {
          const newMessage: ChatMessage = {
            role: 'assistant',
            content: text,
            timestamp: Date.now()
          };
          const updatedHistory = [...prevHistory, newMessage];
          localStorage.setItem('chatHistory', JSON.stringify(updatedHistory));
          return updatedHistory;
        });
        // 更新最新AI消息并停止加载状态
        setLatestAiMessage(text);
        setIsLoading(false);
        setIsStreaming(false);
      }
    },
    // 当语音转文字完成时的处理函数
    onTranscript: (text) => {
      console.log('收到用户语音转文字:', text);
      // 清空当前AI消息，准备显示新回复
      setLatestAiMessage("");
      // 更新聊天历史添加用户消息
      setChatHistory(prevHistory => {
        const newMessage: ChatMessage = {
          role: 'user',
          content: text,
          timestamp: Date.now()
        };
        const updatedHistory = [...prevHistory, newMessage];
        localStorage.setItem('chatHistory', JSON.stringify(updatedHistory));
        return updatedHistory;
      });
    },
    // 处理LLM流式响应的函数
    onLLMStream: (text: string) => {
      console.log('收到LLM流式响应片段:', text);
      // 停止加载状态并累加流式响应文本
      setIsLoading(false)
      setIsSpeaking(false)
      setIsStreaming(true)
      setLatestAiMessage(prev => prev + text)
    },
    // 处理情感分析结果的函数
    onEmotionResponse: (emotion: string) => {
      console.log('收到情感分析结果:', emotion);
      // 更新live2d模型的情感和动作
      live2dStore.setEmotion(emotion || '');
      live2dStore.setMotion(emotion || '');
    },
    // 处理其他WebRTC消息的函数
    onMessage: (message: Record<string, unknown>) => {
      console.log('收到消息:', message);
      // 当用户开始说话时
      if (message.data === 'started_talking') {
        live2dStore.setEmotion('neutral')
        live2dStore.setMotion('thinking')
        setIsSpeaking(true)
      }
    },
    onEvent: (event: string, data: unknown) => {
      console.log('收到事件:', event, data);
      if (event === 'transcript') {
        setIsLoading(true)
        setIsSpeaking(false)
      }
    },
    // 当检测到超过2秒的音频静音时的处理
    onAudioSilence: () => {
      console.log('检测到音频静音超过2秒');
      
      // 如果有WebRTC ID且已连接，触发AI主动对话
      if (webrtcId && isConnected && !isSpeaking && !isLoading && !isStreaming) {
        // 获取localStorage中保存的下一步行动计划
        const nextAction = localStorage.getItem('next_action') || 'share_memory';
        
        console.log(`调用AI触发接口，行动计划: ${nextAction}`);
        
        // 调用ai-trigger接口，传递WebRTC ID和下一步行动计划
        fetch(`${API_BASE_URL}/ai-trigger`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({ 
            webrtc_id: webrtcId,
            next_action: nextAction // 额外添加的属性，用于传递行动计划
          }),
        })
        .then(response => response.json())
        .then(data => {
          console.log('AI触发响应:', data);
          fetch(`${API_BASE_URL}/ai-trigger-reset`, {
            method: 'POST',
            headers: {
              'Content-Type': 'application/json',
            },
            body: JSON.stringify({ webrtc_id: webrtcId }),
          })
        })
        .catch(error => {
          console.error('AI触发请求失败:', error);
        });
      }
    },
    // 当接收到服务器返回的下一步行动计划
    onNextAction: (action: string) => {
      console.log('收到下一步行动计划:', action);
      // 保存当前的行动计划到本地存储以便后续使用
      localStorage.setItem('next_action', action);
    }
  });

  /**
   * 发送配置到后端的函数
   * 根据用户配置选择使用内置服务或自定义服务
   */
  const sendConfigToServer = () => {
    // 如果没有WebRTC ID，则不继续执行
    if (!webrtcId) return;
    
    // 读取存储的配置
    const savedAiConfig = localStorage.getItem('ai_config');
    if (savedAiConfig) {
      const aiConfig = JSON.parse(savedAiConfig);
      
      // 根据配置选择使用内置服务或自定义服务
      if (aiConfig.useBuiltinService) {
        // 使用内置服务的API调用
        fetch(`${API_BASE_URL}/use_builtin_service`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({ 
            webrtc_id: webrtcId,
            ai_model: aiConfig?.ai_model,
            voice_output_language: aiConfig?.voice_output_language || 'ja',
            text_output_language: aiConfig?.text_output_language || 'zh',
            system_prompt: aiConfig?.system_prompt || '',
            user_name: aiConfig?.user_name || '',
            max_context_length: 20  // 内置服务默认为20
          }),
        }).catch(error => {
          console.error('发送内置服务配置失败:', error);
        });
      } else {
        // 使用自定义服务的API调用
        fetch(`${API_BASE_URL}/input_hook`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            webrtc_id: webrtcId,
            llm_api_key: aiConfig.llm_api_key,
            whisper_api_key: aiConfig.whisper_api_key,
            siliconflow_api_key: aiConfig.siliconflow_api_key,
            llm_base_url: aiConfig.llm_base_url,
            whisper_base_url: aiConfig.whisper_base_url,
            siliconflow_voice: aiConfig.siliconflow_voice,
            ai_model: aiConfig.ai_model,
            voice_output_language: aiConfig.voice_output_language,
            text_output_language: aiConfig.text_output_language,
            system_prompt: aiConfig.system_prompt,
            user_name: aiConfig.user_name,
            max_context_length: aiConfig.max_context_length || 20,
          }),
        }).catch(error => {
          console.error('发送自定义服务配置失败:', error);
        });
      }
    } else {
      // 如果没有保存的配置，默认使用内置服务
      fetch(`${API_BASE_URL}/use_builtin_service`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ 
          webrtc_id: webrtcId,
          ai_model: '',
          voice_output_language: 'ja',
          text_output_language: 'zh',
          system_prompt: '',
          user_name: '',
          max_context_length: 20  // 内置服务默认为20
        }),
      }).catch(error => {
        console.error('发送内置服务配置失败:', error);
      });
    }
  };

  /**
   * 初始化时加载本地存储的聊天历史
   */
  useEffect(() => {
    // 从本地存储中加载聊天历史
    const savedHistory = localStorage.getItem('chatHistory');
    if (savedHistory) {
      const history: ChatMessage[] = JSON.parse(savedHistory);
      setChatHistory(history);
      // 查找最近一条AI消息并设置为当前显示内容
      const latestAi = history.reverse().find(msg => msg.role === 'assistant');
      if (latestAi) {
        setLatestAiMessage(latestAi.content);
      }
    }
  }, []);

  /**
   * 删除聊天历史的处理函数
   */
  const handleDeleteHistory = () => {
    // 清空聊天历史状态
    setChatHistory([]);
    // 从本地存储中移除聊天历史
    localStorage.removeItem('chatHistory');
    // 清空当前显示的AI消息
    setLatestAiMessage('');
  };

  /**
   * 登出处理函数
   */
  const handleLogout = () => {
    // 移除用户名
    localStorage.removeItem('amadeus_username');
    // 显示登录蒙层
    setShowOverlay(true);
    // 重置状态
    setIsSpeaking(false);
    setIsLoading(false);
    setLatestAiMessage('');
    // 断开WebRTC连接
    disconnect(); 
    // 关闭开始对话框
    setShowStartDialog(false); 
  };

  /**
   * 登录处理函数
   * @param {string} username - 用户名
   */
  const handleLogin = (username: string) => {
    // 检查是否有已保存的Live2D配置
    const savedConfig = localStorage.getItem('live2d_config');
    if (!savedConfig) {
      // 首次登录，需要先进行配置
      setIsFirstLogin(true);
      setShowOverlay(false);
      setShowConfig(true);
      localStorage.setItem('amadeus_username', username);
    } else {
      // 非首次登录，直接进入应用
      setShowOverlay(false);
      localStorage.setItem('amadeus_username', username);
      
      // 立即显示开始对话框，但状态为连接中
      setShowStartDialog(true);
      
      // 如果模型已加载完成，建立WebRTC连接
      if (isModelReady) {
        connect();
      }
    }
  };

  /**
   * 保存配置后的处理函数
   */
  const handleConfigSave = () => {
    // 关闭配置面板
    setShowConfig(false);
    if (isFirstLogin) {
      // 如果是首次登录，更新状态并连接WebRTC
      setIsFirstLogin(false);
      
      // 立即显示开始对话框，但状态为连接中
      setShowStartDialog(true);
      
      // 如果模型已加载完成，建立WebRTC连接
      if (isModelReady) {
        connect();
      }
    }
  };

  /**
   * 开始对话的处理函数
   */
  const handleStartDialog = () => {
    // 关闭开始对话框
    setShowStartDialog(false);
    
    // 点击开始对话时发送配置到后端
    sendConfigToServer();
  };

  /**
   * 页面加载时检查登录状态并自动连接
   * 依赖于模型加载状态，当模型准备好时执行
   */
  useEffect(() => {
    // 获取保存的用户名和配置
    const savedUsername = localStorage.getItem('amadeus_username');
    const savedConfig = localStorage.getItem('live2d_config');
    
    // 如果已登录且有配置，自动连接
    if (savedUsername && savedConfig) {
      setShowOverlay(false);
      
      // 立即显示开始对话框，但状态为连接中
      setShowStartDialog(true);
      
      // 如果模型已加载完成，自动连接WebRTC
      if (isModelReady) {
        connect();
      }
    }
  }, [isModelReady]); // 依赖于模型加载状态

  // 处理摄像头切换的函数
  const handleToggleVideo = useCallback(() => {
    // 先切换本地状态
    const newState = !isVideoOn;
    setIsVideoOn(newState);
    
    // 如果没有webrtcId，则不发送请求
    if (!webrtcId) return;
    
    // 发送摄像头状态到后端
    fetch(`${API_BASE_URL}/camera-state`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ 
        webrtc_id: webrtcId,
        is_camera_on: newState
      }),
    })
    .then(response => response.json())
    .then(data => {
      console.log('摄像头状态更新响应:', data);
    })
    .catch(error => {
      console.error('摄像头状态更新失败:', error);
    });
  }, [isVideoOn, webrtcId]);

  // 处理视频帧数据发送的函数
  const handleSendVideoFrame = useCallback((message: { type: MessageTypes; data?: string | string[] }) => {
    // 只处理视频帧类型的消息
    if (message.type === MessageTypes.VIDEO_FRAME && typeof message.data === 'string') {
      // 确保有webrtcId并且摄像头状态为开启
      if (!webrtcId || !isVideoOn) return;
      
      // 发送视频帧到后端
      fetch(`${API_BASE_URL}/video-frame`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ 
          webrtc_id: webrtcId,
          frame_data: message.data,
          timestamp: Date.now()
        }),
      })
      .catch(error => {
        console.error('视频帧发送失败:', error);
      });
    } 
  }, [webrtcId, isVideoOn]);

  // 处理麦克风切换
  const handleToggleMicrophone = useCallback(() => {
    // 调用WebRTC的切换麦克风函数
    toggleMicrophone();
  }, [toggleMicrophone]);

  return (
    <>
      {/* 粒子背景效果 */}
      <ParticleBackground />
      
      {/* Live2D模型，角色为牧濑红莉栖 */}
      <Live2dModel role="牧濑红莉栖" onModelReady={() => setIsModelReady(true)} />
      
      {/* 登录蒙层，未登录时显示 */}
      {showOverlay && (
        <LoginOverlay 
          onLogin={handleLogin}
          isModelReady={isModelReady}
        />
      )}
      
      {/* 开始对话框，登录后显示 */}
      {showStartDialog && (
        <StartDialog 
          onStart={handleStartDialog} 
          isFirstConfig={isFirstLogin}
          isConnecting={!isConnected} // 当未连接时，设置为连接中状态
        />
      )}
      
      {/* 视频聊天组件，当视频开启时显示 */}
      {isVideoOn && <VideoChat sendMessage={handleSendVideoFrame} />}
      
      {/* 对话框，显示AI响应内容 */}
      <div className={styles.dialogBox}>
        <div className={styles.dialogHeader} />
        <div className={styles.content}>
          {isLoading ? (
            <span className={styles.loadingDot} />
          ) : isSpeaking ? (
            <span>{t('dialog.listening')}</span>
          ) : (
            <span>{latestAiMessage || t('dialog.waiting')}</span>
          )}
        </div>
      </div>
      
      {/* 工具栏，提供各种控制按钮 */}
      <Toolbar
        isListening={!isMicrophoneMuted}
        isVideoOn={isVideoOn}
        onToggleListening={handleToggleMicrophone}
        onToggleVideo={handleToggleVideo}
        onShowHistory={() => setShowHistory(true)}
        onLogout={handleLogout}
        onShowConfig={() => setShowConfig(true)}
      />
      
      {/* 配置面板，设置AI和Live2D参数 */}
      <ConfigPanel 
        open={showConfig}
        onOpenChange={setShowConfig}
        onSave={handleConfigSave}
      />
      
      {/* 聊天历史记录组件 */}
      <ChatHistory 
        open={showHistory}
        onOpenChange={setShowHistory}
        chatHistory={chatHistory}
        onDeleteHistory={handleDeleteHistory}
      />
      
      {/* 音频元素，用于播放AI响应 */}
      <audio ref={audioRef} autoPlay />
    </>
  )
})

export default VoiceAssistant
