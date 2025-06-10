import { useState, useEffect } from 'react';
import { roleToLive2dMapper } from '@/constants/live2d';
import toast from 'react-hot-toast';


interface Live2DConfig {
  scale1: number;
  x1: number;
  y1: number;
}

interface AIConfig {
  useBuiltinService: boolean;
  llm_api_key: string;
  whisper_api_key: string;
  siliconflow_api_key: string;
  llm_base_url: string;
  llm_base_url_type: string;
  whisper_base_url: string;
  whisper_base_url_type: string;
  whisper_model: string;
  custom_whisper_model: string;
  siliconflow_voice: string;
  ai_model: string;
  custom_model_name: string;
  voice_output_language: string;
  text_output_language: string;
  system_prompt: string;
  user_name: string;
  max_context_length: number;
  mem0_api_key: string;
}

// 默认提示词
const DEFAULT_SYSTEM_PROMPT = "命运石之门(steins gate)的牧濑红莉栖(kurisu),一个天才少女,性格傲娇,不喜欢被叫克里斯蒂娜";

export const useConfigPanel = (open: boolean, onOpenChange: (open: boolean) => void, onSave?: () => void) => {
  const [activeTab, setActiveTab] = useState('avatar');
  const [live2dConfig, setLive2dConfig] = useState<Live2DConfig>({
    scale1: 0.42,
    x1: 550,
    y1: 50
  });
  
  // 添加错误状态
  const [errors, setErrors] = useState<{[key: string]: string}>({});
  
  // 添加语音克隆弹窗状态
  const [voiceCloneModalOpen, setVoiceCloneModalOpen] = useState(false);
  
  const [aiConfig, setAiConfig] = useState<AIConfig>({
    useBuiltinService: true,
    llm_api_key: '',
    whisper_api_key: '',
    siliconflow_api_key: '',
    llm_base_url: '',
    llm_base_url_type: 'amadeus-web',
    whisper_base_url: '',
    whisper_base_url_type: 'aihubmix',
    whisper_model: 'whisper-large-v3',
    custom_whisper_model: '',
    siliconflow_voice: '',
    ai_model: 'gpt-4.1-mini',
    custom_model_name: '',
    voice_output_language: 'ja',
    text_output_language: 'zh',
    system_prompt: DEFAULT_SYSTEM_PROMPT,
    user_name: localStorage.getItem('amadeus_username') || '',
    max_context_length: 20,
    mem0_api_key: ''
  });
  
  useEffect(() => {
    const savedLive2dConfig = localStorage.getItem('live2d_config');
    if (savedLive2dConfig) {
      setLive2dConfig(JSON.parse(savedLive2dConfig));
    } else {
      const defaultConfig = roleToLive2dMapper['牧濑红莉栖'];
      setLive2dConfig({
        scale1: defaultConfig.scale1,
        x1: defaultConfig.x1,
        y1: defaultConfig.y1
      });
    }
    
    const savedAiConfig = localStorage.getItem('ai_config');
    const userName = localStorage.getItem('amadeus_username') || '';
    
    if (savedAiConfig) {
      const parsedConfig = JSON.parse(savedAiConfig);
      // 确保用户名称被正确设置，如果保存的配置中没有用户名或为空，则使用localStorage中的用户名
      setAiConfig({
        ...parsedConfig,
        user_name: parsedConfig.user_name || userName
      });
    } else {
      // 如果没有保存的配置，则设置默认系统提示词和用户名
      setAiConfig(prev => ({
        ...prev,
        system_prompt: DEFAULT_SYSTEM_PROMPT,
        user_name: userName
      }));
    }
  }, [open]);

  // 监听 localStorage 变化
  useEffect(() => {
    // 处理 storage 事件，当其他页面修改 localStorage 时触发
    const handleStorageChange = (event: StorageEvent) => {
      if (event.key === 'amadeus_username' && event.newValue) {
        setAiConfig(prev => ({
          ...prev,
          user_name: event.newValue || ''
        }));
      } else if (event.key === 'ai_config' && event.newValue) {
        const newConfig = JSON.parse(event.newValue);
        setAiConfig(newConfig);
      }
    };

    // 添加监听器
    window.addEventListener('storage', handleStorageChange);

    // 处理自定义事件 - 用于同一页面内的更新
    const handleSamePageStorageChange = () => {
      const userName = localStorage.getItem('amadeus_username') || '';
      const savedAiConfig = localStorage.getItem('ai_config');
      
      if (savedAiConfig) {
        try {
          const parsedConfig = JSON.parse(savedAiConfig);
          setAiConfig({
            ...parsedConfig,
            user_name: parsedConfig.user_name || userName
          });
        } catch (e) {
          console.error('解析 AI 配置失败:', e);
        }
      } else {
        setAiConfig(prev => ({
          ...prev,
          user_name: userName
        }));
      }
    };

    window.addEventListener('amadeus_storage_updated', handleSamePageStorageChange);

    // 清理监听器
    return () => {
      window.removeEventListener('storage', handleStorageChange);
      window.removeEventListener('amadeus_storage_updated', handleSamePageStorageChange);
    };
  }, []);

  const handleSave = () => {
    // 重置错误状态
    setErrors({});
    
    const newErrors: {[key: string]: string} = {};
    
    // 先验证通用设置 - 无论是否使用内置服务都必须填写
    if (!aiConfig.voice_output_language) {
      newErrors.voice_output_language = '请选择语音输出语言';
    }
    
    if (!aiConfig.text_output_language) {
      newErrors.text_output_language = '请选择文本输出语言';
    }
    
    if (!aiConfig.user_name) {
      newErrors.user_name = '请填写用户名称';
    }
    
    if (!aiConfig.system_prompt) {
      newErrors.system_prompt = '请填写系统提示词';
    }
    
    // 非内置服务模式下进行更多字段校验
    if (!aiConfig.useBuiltinService) {
      // 校验 LLM 配置
      if (!aiConfig.llm_api_key) {
        newErrors.llm_api_key = '请填写 LLM API 密钥';
      }
      
      if (aiConfig.llm_base_url_type === 'custom' && !aiConfig.llm_base_url) {
        newErrors.llm_base_url = '请填写自定义 LLM 基础 URL';
      }
      
      // 校验 ASR 配置 
      if (!aiConfig.whisper_api_key) {
        newErrors.whisper_api_key = '请填写 ASR API 密钥';
      }
      
      if (aiConfig.whisper_base_url_type === 'custom' && !aiConfig.whisper_base_url) {
        newErrors.whisper_base_url = '请填写自定义 ASR 基础 URL';
      }
      
      if (aiConfig.whisper_model === 'custom' && !aiConfig.custom_whisper_model) {
        newErrors.custom_whisper_model = '请填写自定义 ASR 模型名称';
      }
      
      // 校验 TTS 配置
      if (!aiConfig.siliconflow_api_key) {
        newErrors.siliconflow_api_key = '请填写语音合成 API 密钥';
      }
      
      // 校验语音ID
      if (!aiConfig.siliconflow_voice) {
        newErrors.siliconflow_voice = '请完成语音克隆';
      }
      
      // 校验 AI 模型
      if (!aiConfig.ai_model) {
        newErrors.ai_model = '请选择 AI 模型';
      }
      
      if (aiConfig.ai_model === 'custom' && !aiConfig.custom_model_name) {
        newErrors.custom_model_name = '请填写自定义模型名称';
      }
      
      // 校验上下文消息数量
      if (!aiConfig.max_context_length || aiConfig.max_context_length < 5) {
        newErrors.max_context_length = '请填写有效的上下文消息数量(至少5)';
      }
      
      // 校验MEM0 API密钥
      if (!aiConfig.mem0_api_key) {
        newErrors.mem0_api_key = '请填写 MEM0 API 密钥';
      }
    }
    
    // 如果有错误，阻止保存并显示错误
    if (Object.keys(newErrors).length > 0) {
      setErrors(newErrors);
      // 判断显示哪个tab
      const aiSettingsErrors = ['llm_api_key', 'llm_base_url', 'whisper_api_key', 
                              'whisper_base_url', 'custom_whisper_model', 'siliconflow_api_key',
                              'siliconflow_voice', 'ai_model', 'custom_model_name', 'max_context_length',
                              'voice_output_language', 'text_output_language', 'user_name', 'system_prompt',
                              'mem0_api_key'];
                              
      if (Object.keys(newErrors).some(key => aiSettingsErrors.includes(key))) {
        setActiveTab('ai');
      }
      return;
    }
    
    // 处理自定义模型名称和基础URL
    let finalConfig = {...aiConfig};
    if (aiConfig.ai_model === 'custom' && aiConfig.custom_model_name) {
      finalConfig = {
        ...finalConfig,
        ai_model: aiConfig.custom_model_name
      };
    }
    
    // 处理ASR模型名称
    if (aiConfig.whisper_model === 'custom' && aiConfig.custom_whisper_model) {
      finalConfig = {
        ...finalConfig,
        whisper_model: aiConfig.custom_whisper_model
      };
    }
    
    // 确保LLM基础URL与选择的类型相匹配
    if (aiConfig.llm_base_url_type === 'aihubmix') {
      finalConfig = {
        ...finalConfig,
        llm_base_url: 'https://aihubmix.com/v1'
      };
    } else if (aiConfig.llm_base_url_type === 'amadeus-web') {
      finalConfig = {
        ...finalConfig,
        llm_base_url: 'https://api.amadeus-web.top/v1'
      };
    }
    // 自定义LLM类型的URL保持不变
    
    // 确保ASR基础URL与选择的类型相匹配
    if (aiConfig.whisper_base_url_type === 'aihubmix') {
      finalConfig = {
        ...finalConfig,
        whisper_base_url: 'https://aihubmix.com/v1'
      };
    } else if (aiConfig.whisper_base_url_type === 'groq') {
      finalConfig = {
        ...finalConfig,
        whisper_base_url: 'https://api.groq.com/openai/v1'
      };
    }
    // 自定义ASR类型的URL保持不变
    
    // 只在本地保存配置，不发送到后端
    // 配置会在用户点击"开始对话"时发送到后端
    localStorage.setItem('live2d_config', JSON.stringify(live2dConfig));
    localStorage.setItem('ai_config', JSON.stringify(finalConfig));
    // 单独保存用户名称，使其在其他地方可以轻松获取
    localStorage.setItem('amadeus_username', finalConfig.user_name);
    
    window.dispatchEvent(new StorageEvent('storage', {
      key: 'live2d_config',
      newValue: JSON.stringify(live2dConfig)
    }));
    
    // 触发自定义事件，通知同一页面内的其他组件
    window.dispatchEvent(new Event('amadeus_storage_updated'));
    
    onOpenChange(false);
    onSave?.();
  };

  const handleLive2dReset = () => {
    const defaultConfig = roleToLive2dMapper['牧濑红莉栖'];
    const newConfig = {
      scale1: defaultConfig.scale1,
      x1: defaultConfig.x1,
      y1: defaultConfig.y1
    };
    setLive2dConfig(newConfig);
    localStorage.removeItem('live2d_config');
    window.dispatchEvent(new StorageEvent('storage', {
      key: 'live2d_config',
      newValue: JSON.stringify(newConfig)
    }));
  };

  const handleLive2dConfigChange = (newConfig: Partial<Live2DConfig>) => {
    const updatedConfig = { ...live2dConfig, ...newConfig };
    setLive2dConfig(updatedConfig);
    window.dispatchEvent(new StorageEvent('storage', {
      key: 'live2d_config',
      newValue: JSON.stringify(updatedConfig)
    }));
  };
  
  const handleAiConfigChange = (key: keyof AIConfig, value: string | boolean | number) => {
    if (key === 'useBuiltinService' && value === true) {
      setAiConfig(prev => ({ 
        ...prev, 
        [key]: value,
        ai_model: '',
        custom_model_name: ''
      }));
      // 切换到内置服务模式时清除错误状态
      setErrors({});
    } else if (key === 'llm_base_url_type') {
      const baseUrlValue = value as string;
      let actualBaseUrl = '';
      
      // 根据选择的类型设置实际的基础URL
      if (baseUrlValue === 'aihubmix') {
        actualBaseUrl = 'https://aihubmix.com/v1';
      } else if (baseUrlValue === 'amadeus-web') {
        actualBaseUrl = 'https://api.amadeus-web.top/v1';
      } else if (baseUrlValue === 'custom') {
        // 保持当前的自定义URL
        actualBaseUrl = aiConfig.llm_base_url;
      }
      
      setAiConfig(prev => ({ 
        ...prev, 
        llm_base_url_type: baseUrlValue,
        llm_base_url: actualBaseUrl
      }));
      
      // 清除相关字段的错误
      if (baseUrlValue !== 'custom') {
        setErrors(prev => {
          const newErrors = {...prev};
          delete newErrors.llm_base_url;
          return newErrors;
        });
      }
    } else if (key === 'whisper_base_url_type') {
      const baseUrlValue = value as string;
      let actualBaseUrl = '';
      
      // 根据选择的类型设置实际的基础URL
      if (baseUrlValue === 'aihubmix') {
        actualBaseUrl = 'https://aihubmix.com/v1';
      } else if (baseUrlValue === 'groq') {
        actualBaseUrl = 'https://api.groq.com/openai/v1';
      } else if (baseUrlValue === 'custom') {
        // 保持当前的自定义URL
        actualBaseUrl = aiConfig.whisper_base_url;
      }
      
      setAiConfig(prev => ({ 
        ...prev, 
        whisper_base_url_type: baseUrlValue,
        whisper_base_url: actualBaseUrl
      }));
      
      // 清除相关字段的错误
      if (baseUrlValue !== 'custom') {
        setErrors(prev => {
          const newErrors = {...prev};
          delete newErrors.whisper_base_url;
          return newErrors;
        });
      }
    } else if (key === 'whisper_model') {
      const modelValue = value as string;
      
      // 如果不是自定义选项，清空自定义模型名称
      if (modelValue !== 'custom') {
        setAiConfig(prev => ({
          ...prev,
          whisper_model: modelValue,
          custom_whisper_model: ''
        }));
        
        // 清除相关字段的错误
        setErrors(prev => {
          const newErrors = {...prev};
          delete newErrors.custom_whisper_model;
          return newErrors;
        });
      } else {
        setAiConfig(prev => ({
          ...prev,
          whisper_model: modelValue
        }));
      }
    } else {
      setAiConfig(prev => ({ ...prev, [key]: value }));
      
      // 清除当前字段的错误（如果有）
      if (errors[key]) {
        setErrors(prev => {
          const newErrors = {...prev};
          delete newErrors[key];
          return newErrors;
        });
      }
    }
  };

  // 处理生成语音ID按钮点击 - 修改为打开语音克隆弹窗
  const handleGenerateVoiceId = () => {
    // 检查是否已填写语音合成API密钥
    if (!aiConfig.siliconflow_api_key) {
      // 设置错误状态
      setErrors(prev => ({
        ...prev,
        siliconflow_api_key: '请先填写语音合成 API 密钥'
      }));
      return;
    }
    
    // 打开语音克隆弹窗
    setVoiceCloneModalOpen(true);
  };
  
  // 处理语音选择
  const handleVoiceSelected = (voiceId: string) => {
    handleAiConfigChange('siliconflow_voice', voiceId);
    toast.success("语音克隆操作成功");
    
    // 清除相关字段的错误
    if (errors.siliconflow_voice) {
      setErrors(prev => {
        const newErrors = {...prev};
        delete newErrors.siliconflow_voice;
        return newErrors;
      });
    }
  };
  
  return {
    activeTab,
    setActiveTab,
    live2dConfig,
    errors,
    aiConfig,
    voiceCloneModalOpen,
    setVoiceCloneModalOpen,
    handleSave,
    handleLive2dReset,
    handleLive2dConfigChange,
    handleAiConfigChange,
    handleGenerateVoiceId,
    handleVoiceSelected
  };
};

// 导出类型定义，方便在组件中使用
export type { Live2DConfig, AIConfig };
export { DEFAULT_SYSTEM_PROMPT }; 