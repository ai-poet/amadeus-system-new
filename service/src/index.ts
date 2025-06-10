import { Hono } from 'hono'
import { cors } from 'hono/cors'
import * as dotenv from 'dotenv'
import { createProxyMiddleware } from 'http-proxy-middleware'
import { createServer } from 'http'
import { Socket } from 'net'
import voiceCloneRoutes from './routes/voiceCloneRoutes'

dotenv.config()

const app = new Hono()
app.use('*', cors())

const WEBRTC_API_URL = process.env.WEBRTC_API_URL
console.log(`WebRTC API代理目标: ${WEBRTC_API_URL}`)

app.route('/node/api/voice-clone', voiceCloneRoutes)

// 创建代理中间件
const apiProxy = createProxyMiddleware({
  target: WEBRTC_API_URL,
  changeOrigin: true,
  pathRewrite: { '^/api': '' },
  ws: true,
  // @ts-ignore
  onProxyReq: (proxyReq, req, res) => {
    if (req.body) {
      const bodyData = JSON.stringify(req.body)
      proxyReq.setHeader('Content-Type', 'application/json')
      proxyReq.setHeader('Content-Length', Buffer.byteLength(bodyData))
      proxyReq.write(bodyData)
    }
  },
  // @ts-ignore
  onError: (err) => console.error('API代理错误:', err),
})

const frontendProxy = process.env.NODE_ENV === 'production' 
  ? createProxyMiddleware({
      target: 'http://127.0.0.1:4173',
      changeOrigin: true,
      ws: true,
      // @ts-ignore
      onError: (err) => console.error('前端代理错误:', err),
      filter: (pathname) => !pathname.startsWith('/api') && !pathname.startsWith('/node/api')
    })
  : null

// 使用 Node.js HTTP 服务器
const server = createServer(async (req, res) => {
  const url = new URL(req.url!, `http://${req.headers.host}`)
  
  if (url.pathname.startsWith('/api')) {
    apiProxy(req, res)
  } else if (url.pathname.startsWith('/node/api')) {
    // Hono 路由
    const response = await app.fetch(new Request(url.toString(), {
      method: req.method,
      headers: req.headers as any,
      body: req.method !== 'GET' && req.method !== 'HEAD' ? req : undefined,
      duplex: 'half'
    }))
    res.statusCode = response.status
    response.headers.forEach((value, key) => res.setHeader(key, value))
    if (response.body) {
      const reader = response.body.getReader()
      while (true) {
        const { done, value } = await reader.read()
        if (done) break
        res.write(value)
      }
    }
    res.end()
  } else if (frontendProxy) {
    frontendProxy(req, res)
  } else {
    res.statusCode = 404
    res.end('Not Found')
  }
})

server.on('upgrade', (req, socket, head) => {
  const url = new URL(req.url!, `http://${req.headers.host}`)
  
  if (url.pathname.startsWith('/api')) {
    apiProxy.upgrade(req, socket as Socket, head)
  } else if (frontendProxy) {
    frontendProxy.upgrade?.(req, socket as Socket, head)
  } else {
    socket.destroy()
  }
})

const port = Number(process.env.PORT) || 3002
server.listen(port, () => console.log(`服务器运行在端口 ${port}`))
