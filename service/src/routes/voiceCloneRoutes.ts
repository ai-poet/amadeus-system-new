import { Hono } from 'hono'
import { VoiceCloneService } from '../services/voiceCloneService'

const router = new Hono()
const voiceCloneService = new VoiceCloneService()

// 预定义语音克隆路由 - 从URL获取音频
router.post('/clone-from-url', async (c) => {
  try {
    const { audioUrl, text, customName, model } = await c.req.json()
    const authorization = c.req.header('authorization')
    const apiKey = authorization?.split(' ')[1]

    if (!apiKey) {
      return c.json({ error: { message: '未提供API密钥' } }, 401)
    }

    if (!audioUrl || !text || !customName || !model) {
      return c.json({ error: { message: '缺少必要参数' } }, 400)
    }

    const result = await voiceCloneService.cloneVoiceFromUrl(audioUrl, text, customName, model, apiKey)
    return c.json(result, 200)
  } catch (error: any) {
    console.error('语音克隆失败:', error)
    return c.json({ error: { message: error.message || '语音克隆失败' } }, 500)
  }
})

// 自定义语音克隆路由 - 从base64获取音频
router.post('/clone-from-base64', async (c) => {
  try {
    const { audio, text, customName, model } = await c.req.json()
    const authorization = c.req.header('authorization')
    const apiKey = authorization?.split(' ')[1]

    if (!apiKey) {
      return c.json({ error: { message: '未提供API密钥' } }, 401)
    }

    if (!audio || !text || !customName || !model) {
      return c.json({ error: { message: '缺少必要参数' } }, 400)
    }
    
    const result = await voiceCloneService.cloneVoiceFromBase64(audio, text, customName, model, apiKey)
    return c.json(result, 200)
  } catch (error: any) {
    console.error('语音克隆失败:', error)
    return c.json({ error: { message: error.message || '语音克隆失败' } }, 500)
  }
})

export default router 