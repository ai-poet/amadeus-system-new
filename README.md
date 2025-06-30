# Amadeus System New Alpha

[English](README_EN.md) | 中文

一个全新的Amadeus实验版本, EL PSY CONGROO~

注意，此版本已经经过重构，和初版已经不同，文档已经更新，请查看文档。

## 🤝 参与贡献

欢迎加入 Amadeus System 的开发！我们期待你的贡献：

- 🌟 提交 Issue 报告 Bug 或提出新功能建议
- 📝 改进文档内容
- 🔧 修复已知问题
- ✨ 开发新功能
- 🎨 改进用户界面

任何形式的贡献都非常欢迎。让我们一起把 Amadeus System 变得更好！

[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat-square)](http://makeapullrequest.com)

## 演示视频

[![BiliBili](https://img.shields.io/badge/BiliBili-视频演示1-ff69b4)](https://www.bilibili.com/video/BV1JnifYcEeM/?spm_id_from=333.1387.homepage.video_card.click)
[![BiliBili](https://img.shields.io/badge/BiliBili-视频演示2-ff69b4)](https://www.bilibili.com/video/BV1ZnrcYkEKz/?spm_id_from=333.1007.top_right_bar_window_history.content.click)

## 文档

详细的说明文档请访问：[Amadeus System 文档中心](https://docs.amadeus-web.top)

## 🚀 开发步骤

### 前端开发环境搭建

1. **克隆项目**
   ```bash
   git clone https://github.com/ai-poet/amadeus-system-new-alpha.git
   cd amadeus-system-new-alpha
   ```

2. **安装依赖**
   ```bash
   npm install
   ```

3. **配置环境变量**
   ```bash
   # 复制环境变量文件
   cp .env.development.example .env.development
   # 编辑配置文件，填入必要的环境变量
   ```

4. **启动开发服务器**
   ```bash
   npm run dev
   ```
   
   应用将在 `http://localhost:1002` 启动

### 后端服务开发

1. **进入服务目录**
   ```bash
   cd service
   ```

2. **安装依赖**
   ```bash
   pnpm install
   ```

3. **配置环境变量**
   ```bash
   # 复制环境变量文件
   cp .env.example .env
   # 编辑配置文件，填入必要的API密钥
   ```

4. **启动开发服务**
   ```bash
   pnpm dev
   ```

### WebRTC服务开发（Python）

1. **进入WebRTC服务目录**
   ```bash
   cd service/webrtc
   ```

2. **创建虚拟环境**
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   # 或 venv\Scripts\activate  # Windows
   ```

3. **安装依赖**
   ```bash
   pip install -r requirements.txt
   ```

4. **配置环境变量**
   ```bash
   # 复制环境变量文件
   cp .env.example .env
   # 编辑配置文件，填入必要的API密钥
   ```

5. **启动WebRTC服务**
   ```bash
   python server.py
   ```
   
   服务将在 `http://localhost:8001` 启动

### 完整开发流程

1. **同时启动所有服务**
   ```bash
   # 终端1: 启动前端
   npm run dev
   
   # 终端2: 启动Node.js服务
   cd service && pnpm dev
   
   # 终端3: 启动WebRTC服务
   cd service/webrtc && python server.py
   ```

2. **代码检查**
   ```bash
   # 前端代码检查
   npm run lint
   
   # 后端服务代码检查
   cd service && pnpm lint
   ```

3. **构建测试**
   ```bash
   # 构建前端
   npm run build
   
   # 构建后端服务
   cd service && pnpm build
   
   # 构建Electron应用
   npm run build:electron
   ```

### 开发环境要求

- **Node.js**: ≥ 18.0.0
- **Python**: ≥ 3.8
- **pnpm**: 推荐使用pnpm作为包管理器
- **Docker**: 可选，用于容器化部署

## 部署方法

### 下载本地客户端

项目提供了预编译的桌面客户端，支持 Windows系统：

1. 访问 [GitHub Releases](https://github.com/ai-poet/amadeus-system-new/releases) 页面
2. 安装并运行客户端
3. 直接试用或者配置必要的参数

### 客户端默认安装路径
C:\Users\你的用户名\AppData\Local\Programs\Amadeus

### 客户端内置服务的配置文件路径
C:\Users\你的用户名\AppData\Local\Programs\Amadeus\resources\service\\.env 
可以修改客户端使用的WEBRTC服务器地址

本地客户端提供与Zeabur在线版本相同的功能，但无需服务器部署，适合个人使用。

### 使用 Zeabur 一键部署(推荐)

[![Deploy to Zeabur](https://zeabur.com/button.svg)](https://zeabur.com/templates/LMSUDW?referralCode=aipoet)

#### 部署步骤

1. 点击上方的 "Deploy to Zeabur" 按钮
2. 如果你还没有 Zeabur 账号，需要先[注册](https://zeabur.com?referralCode=aipoet)。需要花费$5开通Developer计划,可使用WildCard虚拟信用卡开通,也可直接使用支付宝充值余额支付。
3. 点击上方按钮一键部署到AWS香港区域，等待部署完成，然后如下图，填写环境变量，最后再点击Networking，生成域名，你就可以通过 Zeabur 提供的域名访问你的应用了


#### 环境变量配置说明

| 环境变量 | 说明 |
|---------|------|
| `VITE_APP_DEFAULT_USERNAME` | 用于前端登录系统的用户名，从而让Amadeus识别你的身份 |
| `WEBRTC_API_URL` | WEBRTC的服务器API地址，Zeabur模板里已经内置了公共的WEBRTC服务器，你也可以自行参考文档自行搭建 |

注意事项：
- 确保你的项目符合 Zeabur 的部署要求
- 如果你需要自定义域名，可以在 Zeabur 的控制面板中进行设置
- 建议查看 [Zeabur 的官方文档](https://zeabur.com/docs) 获取更多部署相关信息

### 使用 Docker Compose 部署

如果你想在自己的服务器上部署，可以使用 Docker Compose 进行部署。

#### 准备工作

1. 确保你的服务器已安装 [Docker](https://docs.docker.com/get-docker/) 和 [Docker Compose](https://docs.docker.com/compose/install/)
2. 准备好所有必需的环境变量（参考上方环境变量配置说明）

#### Docker Compose 配置

创建 `docker-compose.yml` 文件，内容如下：

```yaml
version: '3'
services:
  container:
    image: ghcr.io/ai-poet/amadeus-system-new-alpha
    ports:
      - "3002:3002"  # 服务端口
    environment:
      - VITE_APP_DEFAULT_USERNAME=${VITE_APP_DEFAULT_USERNAME}
      - WEBRTC_API_URL=${WEBRTC_API_URL}
    restart: unless-stopped
    networks:
      - amadeus-network
    volumes:
      - ./logs:/app/service/logs  # 日志持久化存储
networks:
  amadeus-network:
    driver: bridge
```

#### 部署步骤

1. 创建 `.env` 文件，填入所需的环境变量
2. 在 `docker-compose.yml` 所在目录运行：
```bash
docker-compose up -d
```
3. 服务将在后台启动，可以通过以下命令查看日志：
```bash
docker-compose logs -f
```

### 自行部署WebRTC服务

在Zeabur模板中提供了公共WebRTC服务，但公共服务可能会不稳定，建议单独自行私有化部署WebRTC服务。

#### Docker方式部署WebRTC

克隆仓库后，进入代码仓库的service/webrtc文件夹，使用Dockerfile构建WebRTC服务镜像：

```bash
cd service/webrtc
docker build -t amadeus-webrtc-service .
```

运行WebRTC服务容器：

```bash
docker run -d --name amadeus-webrtc \
  -p 8001:8001 \
  -e LLM_API_KEY=你的OpenAI_API密钥 \
  -e WHISPER_API_KEY=你的Whisper_API密钥 \
  -e SILICONFLOW_API_KEY=你的硅基流动API密钥 \
  -e SILICONFLOW_VOICE=你的硅基流动语音ID \
  -e LLM_BASE_URL=你的大语言模型API的基础URL \
  -e WHISPER_BASE_URL=你的Whisper API的基础URL \
  -e WHISPER_MODEL=你的Whisper模型版本 \
  -e AI_MODEL=大语言模型型号 \
  -e MEM0_API_KEY=你的MEM0记忆服务API密钥 \
  -e TIME_LIMIT=你的WebRTC流的最大时间限制(秒) \
  -e CONCURRENCY_LIMIT=你的最大并发连接数 \
  amadeus-webrtc-service
```

这样部署后，你就可以通过 http://你的服务器IP:8001 访问你自己的WebRTC服务了。

#### WebRTC服务环境变量说明

以下是WebRTC服务的内置AI服务的环境变量说明，可以用于搭建公共服务：

| 环境变量 | 说明 | 默认值 |
|---------|------|-------|
| `LLM_API_KEY` | OpenAI或兼容API的密钥，用于大语言模型服务 | 无 |
| `WHISPER_API_KEY` | Whisper API密钥，用于语音识别服务 | 无 |
| `SILICONFLOW_API_KEY` | 硅基流动API密钥，用于语音合成服务 | 无 |
| `SILICONFLOW_VOICE` | 硅基流动你自定义的语音ID | 无 |
| `LLM_BASE_URL` | 大语言模型API的基础URL | 无 |
| `WHISPER_BASE_URL` | Whisper API的基础URL | 无 |
| `WHISPER_MODEL` | 使用的Whisper模型版本 | 无 |
| `AI_MODEL` | 使用的大语言模型型号 | 无 |
| `MEM0_API_KEY` | MEM0记忆服务的API密钥 | 无 |
| `TIME_LIMIT` | WebRTC流的最大时间限制(秒) | 600 |
| `CONCURRENCY_LIMIT` | 最大并发连接数 | 10 |

#### 端口配置要求

部署WebRTC服务时，需要确保服务器以下端口已开放：

- 80: HTTP通信
- 443: HTTPS通信
- 3478: STUN/TURN服务（TCP）
- 5349: STUN/TURN服务（TLS）
- 49152-65535: 媒体流端口范围（UDP）

> **注意**
> 
> 如果使用云服务提供商（如AWS、阿里云等），请确保在安全组/防火墙设置中开放这些端口。

#### TURN服务器部署

在生产环境中，为了处理复杂网络环境下的音视频穿透问题，通常需要部署TURN服务器。你可以：

- 自行部署Coturn
- 参考FastRTC部署文档进行AWS自动化部署

##### 使用AWS自动部署TURN服务器

FastRTC提供了一个自动化脚本，可在AWS上部署TURN服务器：

1. 克隆FastRTC部署仓库
2. 配置AWS CLI并创建EC2密钥对
3. 修改参数文件，填入TURN用户名和密码
4. 运行CloudFormation脚本自动部署

详细步骤请参考FastRTC的自托管部署指南。

部署完成后，可在WebRTC服务的代码中填入TURN服务器信息：

```json
{
  "iceServers": [
    {
      "urls": "turn:你的TURN服务器IP:3478",
      "username": "你设置的用户名",
      "credential": "你设置的密码"
    }
  ]
}
```

> **提示**
> 
> 正确配置TURN服务器后，即使在复杂的网络环境（如对称NAT、企业防火墙后）也能保证音视频通信的稳定性。

## 🙏 致谢

感谢 **命运石之门 0** 提供的创意灵感

感谢所有开源项目和贡献者

感谢社区用户的反馈和建议

*"The universe has a beginning, but no end. — Infinite.  
Stars, too, have their own beginnings, but their own power results in their destruction. — Finite.  
It is those who possess wisdom who are the greatest fools. History has shown us this.  
You could say that this is the final warning from God to those who resist."*

— **Steins;Gate 0**
