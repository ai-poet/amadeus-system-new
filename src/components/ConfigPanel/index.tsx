import React, { useEffect } from 'react';
import { useTranslation } from 'react-i18next';
import { Sheet, SheetContent, SheetHeader, SheetTitle } from '@/components/ui/sheet';
import { Label } from '@/components/ui/label';
import { Input } from '@/components/ui/input';
import { Button } from '@/components/ui/button';
import { roleToLive2dMapper } from '@/constants/live2d';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { AlertCircle, HelpCircle, Brain, Mic, Speaker, Database, Settings, RotateCcw, Wand2 } from 'lucide-react';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Textarea } from '@/components/ui/textarea';
import { Switch } from '@/components/ui/switch';
import { Toaster } from 'react-hot-toast';
import { useConfigPanel } from './useConfigPanel';
import VoiceCloneModal from './VoiceCloneModal';
import LanguageSelector from '@/components/LanguageSelector';

interface ConfigPanelProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  onSave?: () => void;
}

// 默认提示词
const DEFAULT_SYSTEM_PROMPT = "命运石之门(steins gate)的牧濑红莉栖(kurisu),一个天才少女,性格傲娇,不喜欢被叫克里斯蒂娜";

const ConfigPanel = ({ open, onOpenChange, onSave }: ConfigPanelProps): JSX.Element => {
  const { t } = useTranslation();
  const {
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
  } = useConfigPanel(open, onOpenChange, onSave);

  useEffect(() => {
    const savedLive2dConfig = localStorage.getItem('live2d_config');
    if (savedLive2dConfig) {
      handleLive2dConfigChange(JSON.parse(savedLive2dConfig));
    } else {
      const defaultConfig = roleToLive2dMapper['牧濑红莉栖'];
      handleLive2dConfigChange({
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
      handleAiConfigChange('user_name', parsedConfig.user_name || userName);
    } else {
      // 如果没有保存的配置，则设置默认系统提示词和用户名
      handleAiConfigChange('system_prompt', DEFAULT_SYSTEM_PROMPT);
      handleAiConfigChange('user_name', userName);
    }
  }, [open]);

  // 监听 localStorage 变化
  useEffect(() => {
    // 处理 storage 事件，当其他页面修改 localStorage 时触发
    const handleStorageChange = (event: StorageEvent) => {
      if (event.key === 'amadeus_username' && event.newValue) {
        handleAiConfigChange('user_name', event.newValue || '');
      } else if (event.key === 'ai_config' && event.newValue) {
        const newConfig = JSON.parse(event.newValue);
        handleAiConfigChange('user_name', newConfig.user_name || '');
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
          handleAiConfigChange('user_name', parsedConfig.user_name || userName);
        } catch (e) {
          console.error('解析 AI 配置失败:', e);
        }
      } else {
        handleAiConfigChange('user_name', userName);
      }
    };

    window.addEventListener('amadeus_storage_updated', handleSamePageStorageChange);

    // 清理监听器
    return () => {
      window.removeEventListener('storage', handleStorageChange);
      window.removeEventListener('amadeus_storage_updated', handleSamePageStorageChange);
    };
  }, []);

  return (
    <Sheet open={open} onOpenChange={onOpenChange}>
      <SheetContent side="left" className="w-[95%] sm:w-[450px] overflow-y-auto">
        <SheetHeader className="flex flex-row items-center justify-between">
          <SheetTitle>{t('config.title')}</SheetTitle>
          <LanguageSelector />
        </SheetHeader>
        <Toaster />
        <Tabs value={activeTab} onValueChange={setActiveTab} className="mt-6">
          <TabsList className="grid w-full grid-cols-2">
            <TabsTrigger value="avatar">{t('config.live2d')}</TabsTrigger>
            <TabsTrigger value="ai">{t('config.ai')}</TabsTrigger>
          </TabsList>
          
          <TabsContent value="avatar" className="mt-4">
            <h3 className="text-lg font-medium mb-4">{t('live2d.title')}</h3>
            <Alert variant="info" className="mb-4">
              <AlertCircle className="h-4 w-4" />
              <AlertDescription className="text-xs sm:text-sm">
                {t('alerts.live2dPositionNote')}
              </AlertDescription>
            </Alert>
            <div className="grid gap-4">
              <div className="grid grid-cols-1 sm:grid-cols-4 items-center gap-2 sm:gap-4">
                <Label className="sm:text-right">{t('live2d.scale')}</Label>
                <Input
                  type="number"
                  step="0.01"
                  value={live2dConfig.scale1}
                  onChange={(e) => handleLive2dConfigChange({ scale1: parseFloat(e.target.value) })}
                  className="sm:col-span-3"
                />
              </div>
              <div className="grid grid-cols-1 sm:grid-cols-4 items-center gap-2 sm:gap-4">
                <Label className="sm:text-right">{t('live2d.positionX')}</Label>
                <Input
                  type="number"
                  value={live2dConfig.x1}
                  onChange={(e) => handleLive2dConfigChange({ x1: parseInt(e.target.value) })}
                  className="sm:col-span-3"
                />
              </div>
              <div className="grid grid-cols-1 sm:grid-cols-4 items-center gap-2 sm:gap-4">
                <Label className="sm:text-right">{t('live2d.positionY')}</Label>
                <Input
                  type="number"
                  value={live2dConfig.y1}
                  onChange={(e) => handleLive2dConfigChange({ y1: parseInt(e.target.value) })}
                  className="sm:col-span-3"
                />
              </div>
              <div className="flex justify-end mt-2">
                <Button variant="outline" onClick={handleLive2dReset} size="sm">
                  <RotateCcw className="h-4 w-4 mr-1" />
                  {t('live2d.reset')}
                </Button>
              </div>
            </div>
          </TabsContent>
          
          <TabsContent value="ai" className="mt-4">
            <h3 className="text-lg font-medium mb-4">{t('ai.title')}</h3>
            
            <div className="flex items-center space-x-2 mb-6">
              <Switch 
                checked={aiConfig.useBuiltinService}
                onCheckedChange={(checked: boolean) => handleAiConfigChange('useBuiltinService', checked)}
              />
              <Label>{t('ai.builtinService')}</Label>
            </div>
            
            {aiConfig.useBuiltinService ? (
              <Alert className="mb-4">
                <AlertCircle className="h-4 w-4" />
                <AlertDescription className="text-xs sm:text-sm">
                  {t('ai.builtinServiceDesc')}
                </AlertDescription>
              </Alert>
            ) : (
              <div className="grid gap-6">
                {/* LLM 配置 */}
                <div className="rounded-lg border p-4">
                  <div className="flex items-center gap-2 mb-4">
                    <Brain className="h-5 w-5 text-primary" />
                    <h4 className="text-md font-medium">{t('sections.llmConfig')}</h4>
                  </div>
                  <Alert variant="info" className="mb-4">
                    <AlertCircle className="h-4 w-4" />
                    <AlertDescription className="text-xs sm:text-sm">
                      {t('alerts.apiStandardFormat')}
                    </AlertDescription>
                  </Alert>
                  <div className="grid gap-4">
                    <div className="grid grid-cols-1 sm:grid-cols-4 items-center gap-2 sm:gap-4">
                      <Label className="sm:text-right">{t('ai.llmApiKey')}</Label>
                      <Input
                        value={aiConfig.llm_api_key}
                        onChange={(e) => handleAiConfigChange('llm_api_key', e.target.value)}
                        className={`sm:col-span-3 ${errors.llm_api_key ? 'border-red-500' : ''}`}
                        placeholder={t('ai.llmApiKeyPlaceholder')}
                      />
                    </div>
                    {errors.llm_api_key && (
                      <div className="grid grid-cols-1 sm:grid-cols-4 items-center gap-2 sm:gap-4">
                        <div className="sm:col-span-1"></div>
                        <div className="text-red-500 text-xs sm:col-span-3">{errors.llm_api_key}</div>
                      </div>
                    )}
                    <div className="grid grid-cols-1 sm:grid-cols-4 items-center gap-2 sm:gap-4">
                      <Label className="sm:text-right">{t('ai.llmBaseUrl')}</Label>
                      <Select 
                        value={aiConfig.llm_base_url_type} 
                        onValueChange={(value: string) => handleAiConfigChange('llm_base_url_type', value)}
                      >
                        <SelectTrigger className="sm:col-span-3">
                          <SelectValue placeholder={t('placeholders.selectLlmBaseUrl')} />
                        </SelectTrigger>
                        <SelectContent>
                          <SelectItem value="amadeus-web">https://api.amadeus-web.top/v1</SelectItem>
                          <SelectItem value="aihubmix">https://aihubmix.com/v1</SelectItem>
                          <SelectItem value="custom">{t('ai.customUrl')}</SelectItem>
                        </SelectContent>
                      </Select>
                    </div>
                    
                    {aiConfig.llm_base_url_type === 'custom' && (
                      <div className="grid grid-cols-1 sm:grid-cols-4 items-center gap-2 sm:gap-4 mt-2">
                        <Label className="sm:text-right">{t('ai.customUrl')}</Label>
                        <Input
                          value={aiConfig.llm_base_url}
                          onChange={(e) => handleAiConfigChange('llm_base_url', e.target.value)}
                          className={`sm:col-span-3 ${errors.llm_base_url ? 'border-red-500' : ''}`}
                          placeholder={t('ai.customUrlPlaceholder')}
                        />
                      </div>
                    )}
                    {errors.llm_base_url && aiConfig.llm_base_url_type === 'custom' && (
                      <div className="grid grid-cols-1 sm:grid-cols-4 items-center gap-2 sm:gap-4">
                        <div className="sm:col-span-1"></div>
                        <div className="text-red-500 text-xs sm:col-span-3">{errors.llm_base_url}</div>
                      </div>
                    )}
                    
                    {!aiConfig.useBuiltinService && (
                      <div className="grid grid-cols-1 sm:grid-cols-4 items-center gap-2 sm:gap-4">
                        <Label className="sm:text-right">{t('ai.aiModel')}</Label>
                        <Select 
                          value={aiConfig.ai_model} 
                          onValueChange={(value: string) => {
                            handleAiConfigChange('ai_model', value);
                            // 如果不是自定义选项，清空自定义模型名称
                            if (value !== 'custom') {
                              handleAiConfigChange('custom_model_name', '');
                            }
                          }}
                        >
                          <SelectTrigger className={`sm:col-span-3 ${errors.ai_model ? 'border-red-500' : ''}`}>
                            <SelectValue placeholder={t('placeholders.selectAiModel')} />
                          </SelectTrigger>
                          <SelectContent>
                            <SelectItem value="gpt-4.1-mini">gpt-4.1-mini</SelectItem>
                            <SelectItem value="gemini-2.0-flash">gemini-2.0-flash</SelectItem>
                            <SelectItem value="claude-3-5-sonnet-20241022">claude-3-5-sonnet-20241022</SelectItem>
                            <SelectItem value="claude-3-7-sonnet-20250219">claude-3-7-sonnet-20250219</SelectItem>
                            <SelectItem value="custom">{t('placeholders.enterCustomModelId')}</SelectItem>
                          </SelectContent>
                        </Select>
                      </div>
                    )}
                    {errors.ai_model && !aiConfig.useBuiltinService && (
                      <div className="grid grid-cols-1 sm:grid-cols-4 items-center gap-2 sm:gap-4">
                        <div className="sm:col-span-1"></div>
                        <div className="text-red-500 text-xs sm:col-span-3">{errors.ai_model}</div>
                      </div>
                    )}
                    
                    {aiConfig.ai_model === 'custom' && !aiConfig.useBuiltinService && (
                      <div className="grid grid-cols-1 sm:grid-cols-4 items-center gap-2 sm:gap-4 mt-2">
                        <Label className="sm:text-right">{t('placeholders.enterCustomModelId')}:</Label>
                        <Input
                          value={aiConfig.custom_model_name}
                          onChange={(e) => {
                            const customName = e.target.value;
                            handleAiConfigChange('custom_model_name', customName);
                          }}
                          className={`sm:col-span-3 ${errors.custom_model_name ? 'border-red-500' : ''}`}
                          placeholder={t('placeholders.enterCustomModelId')}
                        />
                      </div>
                    )}
                    {errors.custom_model_name && aiConfig.ai_model === 'custom' && !aiConfig.useBuiltinService && (
                      <div className="grid grid-cols-1 sm:grid-cols-4 items-center gap-2 sm:gap-4">
                        <div className="sm:col-span-1"></div>
                        <div className="text-red-500 text-xs sm:col-span-3">{errors.custom_model_name}</div>
                      </div>
                    )}
                    
                    {!aiConfig.useBuiltinService && (
                      <div className="grid grid-cols-1 sm:grid-cols-4 items-center gap-2 sm:gap-4">
                        <div className="sm:text-right flex items-center sm:justify-end">
                          <Label>{t('ai.maxContextLength')}</Label>
                          <div className="relative inline-block ml-1 group">
                            <HelpCircle className="h-4 w-4 text-muted-foreground cursor-help" />
                            <div className="absolute invisible group-hover:visible w-64 bg-popover text-popover-foreground text-xs rounded p-2 -left-8 -top-2 opacity-0 group-hover:opacity-100 transition-opacity duration-300 pointer-events-none z-50">
                              {t('alerts.contextLengthNote')}
                            </div>
                          </div>
                        </div>
                        <Input
                          type="number"
                          value={aiConfig.max_context_length}
                          onChange={(e) => handleAiConfigChange('max_context_length', parseInt(e.target.value) || 20)}
                          className={`sm:col-span-3 ${errors.max_context_length ? 'border-red-500' : ''}`}
                          placeholder="20"
                          min="5"
                          max="100"
                        />
                      </div>
                    )}
                    {errors.max_context_length && !aiConfig.useBuiltinService && (
                      <div className="grid grid-cols-1 sm:grid-cols-4 items-center gap-2 sm:gap-4">
                        <div className="sm:col-span-1"></div>
                        <div className="text-red-500 text-xs sm:col-span-3">{errors.max_context_length}</div>
                      </div>
                    )}
                  </div>
                </div>
                
                {/* ASR 设置 */}
                <div className="rounded-lg border p-4">
                  <div className="flex items-center gap-2 mb-4">
                    <Mic className="h-5 w-5 text-primary" />
                    <h4 className="text-md font-medium">{t('sections.asrConfig')}</h4>
                  </div>
                  <Alert variant="info" className="mb-4">
                    <AlertCircle className="h-4 w-4" />
                    <AlertDescription className="text-xs sm:text-sm">
                      {t('alerts.whisperApiStandardFormat')}
                    </AlertDescription>
                  </Alert>
                  <div className="grid gap-4">
                    <div className="grid grid-cols-1 sm:grid-cols-4 items-center gap-2 sm:gap-4">
                      <Label className="sm:text-right">{t('asr.whisperApiKey')}</Label>
                      <Input
                        value={aiConfig.whisper_api_key}
                        onChange={(e) => handleAiConfigChange('whisper_api_key', e.target.value)}
                        className={`sm:col-span-3 ${errors.whisper_api_key ? 'border-red-500' : ''}`}
                        placeholder={t('asr.whisperApiKeyPlaceholder')}
                      />
                    </div>
                    {errors.whisper_api_key && (
                      <div className="grid grid-cols-1 sm:grid-cols-4 items-center gap-2 sm:gap-4">
                        <div className="sm:col-span-1"></div>
                        <div className="text-red-500 text-xs sm:col-span-3">{errors.whisper_api_key}</div>
                      </div>
                    )}
                    <div className="grid grid-cols-1 sm:grid-cols-4 items-center gap-2 sm:gap-4">
                      <Label className="sm:text-right">{t('asr.whisperBaseUrl')}</Label>
                      <Select 
                        value={aiConfig.whisper_base_url_type} 
                        onValueChange={(value: string) => handleAiConfigChange('whisper_base_url_type', value)}
                      >
                        <SelectTrigger className="sm:col-span-3">
                          <SelectValue placeholder={t('placeholders.selectAsrBaseUrl')} />
                        </SelectTrigger>
                        <SelectContent>
                          <SelectItem value="aihubmix">https://aihubmix.com/v1</SelectItem>
                          <SelectItem value="groq">https://api.groq.com/openai/v1</SelectItem>
                          <SelectItem value="custom">{t('ai.customUrl')}</SelectItem>
                        </SelectContent>
                      </Select>
                    </div>
                    
                    {aiConfig.whisper_base_url_type === 'custom' && (
                      <div className="grid grid-cols-1 sm:grid-cols-4 items-center gap-2 sm:gap-4 mt-2">
                        <Label className="sm:text-right">{t('ai.customUrl')}:</Label>
                        <Input
                          value={aiConfig.whisper_base_url}
                          onChange={(e) => handleAiConfigChange('whisper_base_url', e.target.value)}
                          className={`sm:col-span-3 ${errors.whisper_base_url ? 'border-red-500' : ''}`}
                          placeholder={t('asr.whisperBaseUrlPlaceholder')}
                        />
                      </div>
                    )}
                    {errors.whisper_base_url && aiConfig.whisper_base_url_type === 'custom' && (
                      <div className="grid grid-cols-1 sm:grid-cols-4 items-center gap-2 sm:gap-4">
                        <div className="sm:col-span-1"></div>
                        <div className="text-red-500 text-xs sm:col-span-3">{errors.whisper_base_url}</div>
                      </div>
                    )}

                    <div className="grid grid-cols-1 sm:grid-cols-4 items-center gap-2 sm:gap-4">
                      <Label className="sm:text-right">{t('asr.customWhisperModel')}:</Label>
                      <Select 
                        value={aiConfig.whisper_model} 
                        onValueChange={(value: string) => handleAiConfigChange('whisper_model', value)}
                      >
                        <SelectTrigger className="sm:col-span-3">
                          <SelectValue placeholder={t('placeholders.selectAsrModel')} />
                        </SelectTrigger>
                        <SelectContent>
                          <SelectItem value="whisper-large-v3">whisper-large-v3</SelectItem>
                          <SelectItem value="whisper-large-v3-turbo">whisper-large-v3-turbo</SelectItem>
                          <SelectItem value="whisper-1">whisper-1</SelectItem>
                          <SelectItem value="custom">{t('ai.customUrl')}</SelectItem>
                        </SelectContent>
                      </Select>
                    </div>
                    
                    {aiConfig.whisper_model === 'custom' && (
                      <div className="grid grid-cols-1 sm:grid-cols-4 items-center gap-2 sm:gap-4 mt-2">
                        <Label className="sm:text-right">{t('asr.customWhisperModel')}:</Label>
                        <Input
                          value={aiConfig.custom_whisper_model}
                          onChange={(e) => handleAiConfigChange('custom_whisper_model', e.target.value)}
                          className={`sm:col-span-3 ${errors.custom_whisper_model ? 'border-red-500' : ''}`}
                          placeholder={t('asr.customWhisperModelPlaceholder')}
                        />
                      </div>
                    )}
                    {errors.custom_whisper_model && aiConfig.whisper_model === 'custom' && (
                      <div className="grid grid-cols-1 sm:grid-cols-4 items-center gap-2 sm:gap-4">
                        <div className="sm:col-span-1"></div>
                        <div className="text-red-500 text-xs sm:col-span-3">{errors.custom_whisper_model}</div>
                      </div>
                    )}
                  </div>
                </div>
                
                {/* TTS 设置 */}
                <div className="rounded-lg border p-4">
                  <div className="flex items-center gap-2 mb-4">
                    <Speaker className="h-5 w-5 text-primary" />
                    <h4 className="text-md font-medium">{t('sections.ttsConfig')}</h4>
                  </div>
                  <div className="grid gap-4">
                    <div className="grid grid-cols-1 sm:grid-cols-4 items-center gap-2 sm:gap-4">
                      <Label className="sm:text-right">{t('tts.provider')}</Label>
                      <Select 
                        value="siliconflow" 
                        onValueChange={(value) => {
                          // 目前只支持硅基流动，未来可以扩展其他提供商
                          console.log("TTS provider selected:", value);
                        }}
                      >
                        <SelectTrigger className="sm:col-span-3">
                          <SelectValue placeholder={t('placeholders.selectTtsProvider')} />
                        </SelectTrigger>
                        <SelectContent>
                          <SelectItem value="siliconflow">SiliconFlow</SelectItem>
                        </SelectContent>
                      </Select>
                    </div>
                    <div className="grid grid-cols-1 sm:grid-cols-4 items-center gap-2 sm:gap-4">
                      <Label className="sm:text-right">{t('tts.siliconflowApiKey')}</Label>
                      <Input
                        value={aiConfig.siliconflow_api_key}
                        onChange={(e) => handleAiConfigChange('siliconflow_api_key', e.target.value)}
                        className={`sm:col-span-3 ${errors.siliconflow_api_key ? 'border-red-500' : ''}`}
                        placeholder={t('tts.siliconflowApiKeyPlaceholder')}
                      />
                    </div>
                    {errors.siliconflow_api_key && (
                      <div className="grid grid-cols-1 sm:grid-cols-4 items-center gap-2 sm:gap-4">
                        <div className="sm:col-span-1"></div>
                        <div className="text-red-500 text-xs sm:col-span-3">{errors.siliconflow_api_key}</div>
                      </div>
                    )}
                    <div className="grid grid-cols-1 sm:grid-cols-4 items-center gap-2 sm:gap-4">
                      <Label className="sm:text-right">{t('tts.voiceClone')}:</Label>
                      <div className="sm:col-span-3 flex gap-2">
                        <Button 
                          type="button" 
                          variant={aiConfig.siliconflow_voice ? "default" : "outline"}
                          onClick={handleGenerateVoiceId}
                          className={`w-full ${aiConfig.siliconflow_voice ? "bg-green-600 hover:bg-green-700" : ""}`}
                        >
                          <Wand2 className="h-4 w-4 mr-1" />
                          {aiConfig.siliconflow_voice ? t('buttons.voiceCloneSuccess') : t('buttons.startVoiceClone')}
                        </Button>
                      </div>
                    </div>
                    {errors.siliconflow_voice && (
                      <div className="grid grid-cols-1 sm:grid-cols-4 items-center gap-2 sm:gap-4">
                        <div className="sm:col-span-1"></div>
                        <div className="text-red-500 text-xs sm:col-span-3">{errors.siliconflow_voice}</div>
                      </div>
                    )}
                  </div>
                </div>
                
                {/* MEM0 记忆配置 */}
                <div className="rounded-lg border p-4">
                  <div className="flex items-center gap-2 mb-4">
                    <Database className="h-5 w-5 text-primary" />
                    <h4 className="text-md font-medium">{t('sections.memoryConfig')}</h4>
                  </div>
                  <div className="grid gap-4">
                    <div className="grid grid-cols-1 sm:grid-cols-4 items-center gap-2 sm:gap-4">
                      <div className="sm:text-right flex items-center sm:justify-end">
                        <Label>{t('mem0.apiKey')}</Label>
                        <div className="relative inline-block ml-1 group">
                          <HelpCircle className="h-4 w-4 text-muted-foreground cursor-help" />
                          <div className="absolute invisible group-hover:visible w-64 bg-popover text-popover-foreground text-xs rounded p-2 -left-8 -top-2 opacity-0 group-hover:opacity-100 transition-opacity duration-300 pointer-events-none z-50">
                            {t('alerts.memoryServiceNote')}
                          </div>
                        </div>
                      </div>
                      <Input
                        value={aiConfig.mem0_api_key}
                        onChange={(e) => handleAiConfigChange('mem0_api_key', e.target.value)}
                        className={`sm:col-span-3 ${errors.mem0_api_key ? 'border-red-500' : ''}`}
                        placeholder={t('mem0.apiKeyPlaceholder')}
                      />
                    </div>
                    {errors.mem0_api_key && (
                      <div className="grid grid-cols-1 sm:grid-cols-4 items-center gap-2 sm:gap-4">
                        <div className="sm:col-span-1"></div>
                        <div className="text-red-500 text-xs sm:col-span-3">{errors.mem0_api_key}</div>
                      </div>
                    )}
                  </div>
                </div>
              </div>
            )}
            
            <div className="rounded-lg border p-4 mt-6">
              <div className="flex items-center gap-2 mb-4">
                <Settings className="h-5 w-5 text-primary" />
                <h4 className="text-md font-medium">{t('sections.generalSettings')}</h4>
              </div>
              <div className="grid gap-4">
                <div className="grid grid-cols-1 sm:grid-cols-4 items-center gap-2 sm:gap-4">
                  <Label className="sm:text-right">{t('ai.voiceOutputLanguage')}</Label>
                  <Select 
                    value={aiConfig.voice_output_language} 
                    onValueChange={(value: string) => handleAiConfigChange('voice_output_language', value)}
                  >
                    <SelectTrigger className={`sm:col-span-3 ${errors.voice_output_language ? 'border-red-500' : ''}`}>
                      <SelectValue placeholder={t('ai.voiceOutputLanguagePlaceholder')} />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="zh">{t('common.chinese')}</SelectItem>
                      <SelectItem value="en">{t('common.english')}</SelectItem>
                      <SelectItem value="ja">{t('common.japanese')}</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
                {errors.voice_output_language && (
                  <div className="grid grid-cols-1 sm:grid-cols-4 items-center gap-2 sm:gap-4">
                    <div className="sm:col-span-1"></div>
                    <div className="text-red-500 text-xs sm:col-span-3">{errors.voice_output_language}</div>
                  </div>
                )}
                
                <div className="grid grid-cols-1 sm:grid-cols-4 items-center gap-2 sm:gap-4">
                  <Label className="sm:text-right">{t('ai.textOutputLanguage')}</Label>
                  <Select 
                    value={aiConfig.text_output_language} 
                    onValueChange={(value: string) => handleAiConfigChange('text_output_language', value)}
                  >
                    <SelectTrigger className={`sm:col-span-3 ${errors.text_output_language ? 'border-red-500' : ''}`}>
                      <SelectValue placeholder={t('ai.textOutputLanguagePlaceholder')} />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="zh">{t('common.chinese')}</SelectItem>
                      <SelectItem value="en">{t('common.english')}</SelectItem>
                      <SelectItem value="ja">{t('common.japanese')}</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
                {errors.text_output_language && (
                  <div className="grid grid-cols-1 sm:grid-cols-4 items-center gap-2 sm:gap-4">
                    <div className="sm:col-span-1"></div>
                    <div className="text-red-500 text-xs sm:col-span-3">{errors.text_output_language}</div>
                  </div>
                )}
                
                <div className="grid grid-cols-1 sm:grid-cols-4 items-center gap-2 sm:gap-4">
                  <Label className="sm:text-right">{t('ai.userName')}</Label>
                  <Input
                    value={aiConfig.user_name}
                    onChange={(e) => handleAiConfigChange('user_name', e.target.value)}
                    className={`sm:col-span-3 ${errors.user_name ? 'border-red-500' : ''}`}
                    placeholder={t('placeholders.yourName')}
                  />
                </div>
                {errors.user_name && (
                  <div className="grid grid-cols-1 sm:grid-cols-4 items-center gap-2 sm:gap-4">
                    <div className="sm:col-span-1"></div>
                    <div className="text-red-500 text-xs sm:col-span-3">{errors.user_name}</div>
                  </div>
                )}
                
                <div className="grid grid-cols-1 sm:grid-cols-4 items-start gap-2 sm:gap-4 mt-2">
                  <Label className="sm:text-right mt-2">{t('ai.systemPrompt')}</Label>
                  <Textarea
                    value={aiConfig.system_prompt}
                    onChange={(e: React.ChangeEvent<HTMLTextAreaElement>) => handleAiConfigChange('system_prompt', e.target.value)}
                    className={`sm:col-span-3 min-h-[100px] ${errors.system_prompt ? 'border-red-500' : ''}`}
                    placeholder={t('placeholders.systemPromptDesc')}
                  />
                </div>
                {errors.system_prompt && (
                  <div className="grid grid-cols-1 sm:grid-cols-4 items-center gap-2 sm:gap-4">
                    <div className="sm:col-span-1"></div>
                    <div className="text-red-500 text-xs sm:col-span-3">{errors.system_prompt}</div>
                  </div>
                )}
              </div>
            </div>
          </TabsContent>
        </Tabs>
        
        {/* 错误摘要提示 */}
        {Object.keys(errors).length > 0 && (
          <Alert variant="destructive" className="mt-4 mb-2">
            <AlertCircle className="h-4 w-4" />
            <AlertDescription className="text-xs sm:text-sm">
              {t('placeholders.pleaseCompleteFields')}
            </AlertDescription>
          </Alert>
        )}
        
        <div className="flex justify-end gap-4 mt-8 mb-12">
          <Button onClick={handleSave}>
            <Settings className="h-4 w-4 mr-1" />
            {t('common.save')}
          </Button>
        </div>
        
        {/* 语音克隆弹窗 */}
        <VoiceCloneModal 
          open={voiceCloneModalOpen} 
          onOpenChange={setVoiceCloneModalOpen}
          onVoiceSelected={handleVoiceSelected}
          apiKey={aiConfig.siliconflow_api_key}
        />
      </SheetContent>
    </Sheet>
  );
};

export default ConfigPanel; 