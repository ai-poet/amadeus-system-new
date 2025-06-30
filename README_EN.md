# Amadeus System New Alpha

English | [中文](README.md)

A brand new experimental version of Amadeus, EL PSY CONGROO~

Note: This version has been refactored and is different from the initial version. The documentation has been updated, please check the documentation.

## 🤝 Contributing

Welcome to join the development of Amadeus System! We look forward to your contributions:

- 🌟 Submit Issues to report bugs or suggest new features
- 📝 Improve documentation content
- 🔧 Fix known issues
- ✨ Develop new features
- 🎨 Improve user interface

Any form of contribution is very welcome. Let's make Amadeus System better together!

[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat-square)](http://makeapullrequest.com)

## Demo Videos

[![BiliBili](https://img.shields.io/badge/BiliBili-Demo%201-ff69b4)](https://www.bilibili.com/video/BV1JnifYcEeM/?spm_id_from=333.1387.homepage.video_card.click)
[![BiliBili](https://img.shields.io/badge/BiliBili-Demo%202-ff69b4)](https://www.bilibili.com/video/BV1ZnrcYkEKz/?spm_id_from=333.1007.top_right_bar_window_history.content.click)

## Documentation

For detailed documentation, please visit: [Amadeus System Documentation Center](https://docs.amadeus-web.top)

## 🚀 Development Guide

### Frontend Development Setup

1. **Clone the project**
   ```bash
   git clone https://github.com/ai-poet/amadeus-system-new-alpha.git
   cd amadeus-system-new-alpha
   ```

2. **Install dependencies**
   ```bash
   npm install
   ```

3. **Configure environment variables**
   ```bash
   # Copy environment file
   cp .env.development.example .env.development
   # Edit configuration file and fill in necessary environment variables
   ```

4. **Start development server**
   ```bash
   npm run dev
   ```
   
   The application will start at `http://localhost:1002`

### Backend Service Development

1. **Navigate to service directory**
   ```bash
   cd service
   ```

2. **Install dependencies**
   ```bash
   pnpm install
   ```

3. **Configure environment variables**
   ```bash
   # Copy environment file
   cp .env.example .env
   # Edit configuration file and fill in necessary API keys
   ```

4. **Start development service**
   ```bash
   pnpm dev
   ```

### WebRTC Service Development (Python)

1. **Navigate to WebRTC service directory**
   ```bash
   cd service/webrtc
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   # or venv\Scripts\activate  # Windows
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment variables**
   ```bash
   # Copy environment file
   cp .env.example .env
   # Edit configuration file and fill in necessary API keys
   ```

5. **Start WebRTC service**
   ```bash
   python server.py
   ```
   
   The service will start at `http://localhost:8001`

### Complete Development Workflow

1. **Start all services simultaneously**
   ```bash
   # Terminal 1: Start frontend
   npm run dev
   
   # Terminal 2: Start Node.js service
   cd service && pnpm dev
   
   # Terminal 3: Start WebRTC service
   cd service/webrtc && python server.py
   ```

2. **Code linting**
   ```bash
   # Frontend code linting
   npm run lint
   
   # Backend service code linting
   cd service && pnpm lint
   ```

3. **Build testing**
   ```bash
   # Build frontend
   npm run build
   
   # Build backend service
   cd service && pnpm build
   
   # Build Electron application
   npm run build:electron
   ```

### Development Environment Requirements

- **Node.js**: ≥ 18.0.0
- **Python**: ≥ 3.8
- **pnpm**: Recommended package manager
- **Docker**: Optional, for containerized deployment

## Deployment Methods

### Download Local Client

The project provides precompiled desktop clients that support Windows systems:

1. Visit the [GitHub Releases](https://github.com/ai-poet/amadeus-system-new/releases) page
2. Install and run the client
3. Try it directly or configure necessary parameters

### Default Client Installation Path
C:\Users\YourUsername\AppData\Local\Programs\Amadeus

### Built-in Service Configuration File Path for Client
C:\Users\YourUsername\AppData\Local\Programs\Amadeus\resources\service\\.env 
You can modify the WebRTC server address used by the client

The local client provides the same functionality as the Zeabur online version, but without server deployment, suitable for personal use.

### One-Click Deployment with Zeabur (Recommended)

[![Deploy to Zeabur](https://zeabur.com/button.svg)](https://zeabur.com/templates/LMSUDW?referralCode=aipoet)

#### Deployment Steps

1. Click the "Deploy to Zeabur" button above
2. If you don't have a Zeabur account yet, you need to [register](https://zeabur.com?referralCode=aipoet) first. You need to spend $5 to activate the Developer plan, you can use WildCard virtual credit card to activate, or directly use Alipay to top up balance for payment.
3. Click the button above for one-click deployment to AWS Hong Kong region, wait for deployment to complete, then fill in environment variables as shown below, and finally click Networking to generate a domain name, you can access your application through the domain provided by Zeabur


#### Environment Variable Configuration

| Environment Variable | Description |
|---------------------|-------------|
| `VITE_APP_DEFAULT_USERNAME` | Username for frontend login system, allowing Amadeus to recognize your identity |
| `WEBRTC_API_URL` | WebRTC server API address, the Zeabur template has built-in public WebRTC server, you can also build your own according to the documentation |

Notes:
- Ensure your project meets Zeabur's deployment requirements
- If you need a custom domain, you can set it up in Zeabur's control panel
- It's recommended to check [Zeabur's official documentation](https://zeabur.com/docs) for more deployment-related information

### Deploy with Docker Compose

If you want to deploy on your own server, you can use Docker Compose for deployment.

#### Prerequisites

1. Ensure your server has [Docker](https://docs.docker.com/get-docker/) and [Docker Compose](https://docs.docker.com/compose/install/) installed
2. Prepare all required environment variables (refer to the environment variable configuration above)

#### Docker Compose Configuration

Create a `docker-compose.yml` file with the following content:

```yaml
version: '3'
services:
  container:
    image: ghcr.io/ai-poet/amadeus-system-new-alpha
    ports:
      - "3002:3002"  # Service port
    environment:
      - VITE_APP_DEFAULT_USERNAME=${VITE_APP_DEFAULT_USERNAME}
      - WEBRTC_API_URL=${WEBRTC_API_URL}
    restart: unless-stopped
    networks:
      - amadeus-network
    volumes:
      - ./logs:/app/service/logs  # Log persistence storage
networks:
  amadeus-network:
    driver: bridge
```

#### Deployment Steps

1. Create a `.env` file and fill in the required environment variables
2. Run in the directory where `docker-compose.yml` is located:
```bash
docker-compose up -d
```
3. The service will start in the background, you can view logs with the following command:
```bash
docker-compose logs -f
```

### Self-hosted WebRTC Service Deployment

The Zeabur template provides a public WebRTC service, but public services may be unstable, so it's recommended to deploy WebRTC service privately.

#### Docker-based WebRTC Deployment

After cloning the repository, go to the service/webrtc folder in the code repository and use Dockerfile to build the WebRTC service image:

```bash
cd service/webrtc
docker build -t amadeus-webrtc-service .
```

Run the WebRTC service container:

```bash
docker run -d --name amadeus-webrtc \
  -p 8001:8001 \
  -e LLM_API_KEY=YourOpenAI_API_Key \
  -e WHISPER_API_KEY=YourWhisper_API_Key \
  -e SILICONFLOW_API_KEY=YourSiliconFlow_API_Key \
  -e SILICONFLOW_VOICE=YourSiliconFlow_Voice_ID \
  -e LLM_BASE_URL=YourLLM_API_Base_URL \
  -e WHISPER_BASE_URL=YourWhisper_API_Base_URL \
  -e WHISPER_MODEL=YourWhisper_Model_Version \
  -e AI_MODEL=YourLLM_Model_Version \
  -e MEM0_API_KEY=YourMEM0_Memory_Service_API_Key \
  -e TIME_LIMIT=YourWebRTC_Stream_Max_Time_Limit_Seconds \
  -e CONCURRENCY_LIMIT=YourMax_Concurrent_Connections \
  amadeus-webrtc-service
```

After deployment, you can access your own WebRTC service through http://YourServerIP:8001.

#### WebRTC Service Environment Variables

The following are the environment variables for the built-in AI services of the WebRTC service, which can be used to build public services:

| Environment Variable | Description | Default Value |
|---------------------|-------------|---------------|
| `LLM_API_KEY` | OpenAI or compatible API key for large language model service | None |
| `WHISPER_API_KEY` | Whisper API key for speech recognition service | None |
| `SILICONFLOW_API_KEY` | SiliconFlow API key for text-to-speech service | None |
| `SILICONFLOW_VOICE` | Your custom voice ID in SiliconFlow | None |
| `LLM_BASE_URL` | Base URL for large language model API | None |
| `WHISPER_BASE_URL` | Base URL for Whisper API | None |
| `WHISPER_MODEL` | Whisper model version to use | None |
| `AI_MODEL` | Large language model version to use | None |
| `MEM0_API_KEY` | MEM0 memory service API key | None |
| `TIME_LIMIT` | Maximum time limit for WebRTC stream (seconds) | 600 |
| `CONCURRENCY_LIMIT` | Maximum concurrent connections | 10 |

#### Port Configuration Requirements

When deploying WebRTC service, ensure the following ports are open on your server:

- 80: HTTP communication
- 443: HTTPS communication
- 3478: STUN/TURN service (TCP)
- 5349: STUN/TURN service (TLS)
- 49152-65535: Media stream port range (UDP)

> **Note**
> 
> If using cloud service providers (such as AWS, Alibaba Cloud, etc.), please ensure these ports are opened in security group/firewall settings.

#### TURN Server Deployment

In production environments, to handle audio/video penetration issues in complex network environments, TURN servers usually need to be deployed. You can:

- Deploy Coturn yourself
- Refer to FastRTC deployment documentation for AWS automated deployment

##### Automated TURN Server Deployment on AWS

FastRTC provides an automation script that can deploy TURN servers on AWS:

1. Clone the FastRTC deployment repository
2. Configure AWS CLI and create EC2 key pairs
3. Modify parameter files, fill in TURN username and password
4. Run CloudFormation script for automated deployment

For detailed steps, please refer to FastRTC's self-hosted deployment guide.

After deployment is complete, you can fill in the TURN server information in the WebRTC service code:

```json
{
  "iceServers": [
    {
      "urls": "turn:YourTURNServerIP:3478",
      "username": "YourSetUsername",
      "credential": "YourSetPassword"
    }
  ]
}
```

> **Tip**
> 
> After correctly configuring the TURN server, even in complex network environments (such as symmetric NAT, behind corporate firewalls), the stability of audio and video communication can be guaranteed.

## 🙏 Acknowledgments

Thanks to **Steins;Gate 0** for the creative inspiration

Thanks to all open source projects and contributors

Thanks to community users for their feedback and suggestions

*"The universe has a beginning, but no end. — Infinite.  
Stars, too, have their own beginnings, but their own power results in their destruction. — Finite.  
It is those who possess wisdom who are the greatest fools. History has shown us this.  
You could say that this is the final warning from God to those who resist."*

— **Steins;Gate 0**