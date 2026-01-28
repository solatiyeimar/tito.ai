# Tito.AI Backend - Advanced Pipecat Demo

Sistema de asistente de voz con IA altamente extensible para calificación de clientes potenciales y consultoría técnica, basado en [Pipecat](https://github.com/pipecat-ai/pipecat).

---

## Características

-   **Multi-Proveedor End-to-End**: Intercambia LLMs, STT y TTS de forma transparente.
-   **Bot Multimodal (Voz a Voz)**: Soporte nativo para latencia ultra-baja con Gemini Live, OpenAI Realtime, AWS Nova Sonic y Ultravox.
-   **Bots de Flujo**: Gestión de estados compleja con `pipecat-flows`.
-   **Conectividad Flexible**: Integración con Daily.co (WebRTC) y soporte para RTVI.

---

## Proveedores Soportados

| Tipo | Proveedores |
| :--- | :--- |
| **LLM** | Google (Gemini 2.0+), OpenAI (GPT-4o), Anthropic (Claude 3.5), Groq, Together AI, Mistral |
| **STT** | Deepgram (Nova 3), Gladia, AssemblyAI, Groq (Whisper) |
| **TTS** | Deepgram, Cartesia, ElevenLabs, PlayHT, Rime, OpenAI, Azure |
| **Multimodal** | Gemini Multimodal Live, OpenAI Realtime, AWS Nova Sonic, Ultravox |

---

## Inicio Rápido

### 1. Clonar e instalar dependencias

```bash
cd agents/examples/demo

# Opción 1: Con uv (recomendado)
uv sync

# Opción 2: Con pip
pip install -e .
```

### 2. Configurar variables de entorno

```bash
cp .env.example .env
# Edita .env con tus claves de API
```

### 3. Ejecutar el servidor

```bash
# Con uv
uv run main.py

# Con Python directamente
python main.py
```

El servidor estará disponible en `http://localhost:7860`.

---

## Modos de Ejecución

### Modo Servidor (Recomendado)

Ejecuta el servidor FastAPI que crea salas de Daily.co automáticamente:

```bash
python main.py
```

Luego:
- **Acceso directo**: Abre `http://localhost:7860/` en tu navegador
- **API REST**: Haz `POST` a `http://localhost:7860/connect`

### Modo Runner (CLI Directo)

Si ya tienes una sala de Daily.co, puedes lanzar el bot directamente:

```bash
python runner.py -u "https://tu-dominio.daily.co/sala" -t "TOKEN"
```

---

## Combinaciones de Proveedores

Configura las variables en tu archivo `.env` y luego inicia el servidor:

```bash
uv run main.py
# Abre http://localhost:7860 en tu navegador
```

### Ejemplo 1: Bot de Flujo con Google Gemini + Deepgram

```bash
# .env
BOT_TYPE=flow
LLM_PROVIDER=google
STT_PROVIDER=deepgram
TTS_PROVIDER=deepgram
```

### Ejemplo 2: Bot de Flujo con Anthropic + ElevenLabs

```bash
# .env
BOT_TYPE=flow
LLM_PROVIDER=anthropic
TTS_PROVIDER=elevenlabs
ELEVENLABS_VOICE_ID=21m00Tcm4TlvDq8ikWAM
```

### Ejemplo 3: Bot con Groq (LLM + STT) + Cartesia

```bash
# .env
BOT_TYPE=flow
LLM_PROVIDER=groq
STT_PROVIDER=groq
TTS_PROVIDER=cartesia
CARTESIA_VOICE=79a125e8-cd45-4c13-8a67-188112f4dd22
```

### Ejemplo 4: Bot Multimodal con Gemini Live

```bash
# .env
BOT_TYPE=multimodal
LLM_PROVIDER=google
```

> **Nota**: El bot multimodal NO usa STT/TTS separados. El modelo maneja audio directamente.

### Ejemplo 5: Bot Multimodal con OpenAI Realtime

```bash
# .env
BOT_TYPE=multimodal
LLM_PROVIDER=openai
```

### Ejemplo 6: Bot de Alta Velocidad (Groq + Rime)

```bash
# .env
BOT_TYPE=flow
LLM_PROVIDER=groq
LLM_MODEL=llama-3.1-70b-versatile
STT_PROVIDER=deepgram
TTS_PROVIDER=rime
RIME_VOICE_ID=mist
```

---

## Bots Multimodales (Voz a Voz Nativo)

Los bots multimodales procesan audio directamente sin pasar por STT/LLM/TTS separados, lo que reduce la latencia significativamente.

### Uso Básico

```bash
# 1. Configura el .env
BOT_TYPE=multimodal
LLM_PROVIDER=google  # o openai, aws, ultravox

# 2. Inicia el servidor
uv run main.py

# 3. Abre en navegador o usa la API
curl -X POST http://localhost:7860/connect
```

### Gemini Multimodal Live (Google)

El modelo más avanzado de Google para conversación en tiempo real.

```bash
# .env
BOT_TYPE=multimodal
LLM_PROVIDER=google
GOOGLE_API_KEY=tu_clave_de_google
```

### OpenAI Realtime

API de tiempo real de OpenAI para conversación fluida.

```bash
# .env
BOT_TYPE=multimodal
LLM_PROVIDER=openai
OPENAI_API_KEY=tu_clave_de_openai
```

### AWS Nova Sonic

El modelo multimodal de Amazon Bedrock.

```bash
# .env
BOT_TYPE=multimodal
LLM_PROVIDER=aws
AWS_ACCESS_KEY_ID=tu_access_key
AWS_SECRET_ACCESS_KEY=tu_secret_key
AWS_REGION=us-east-1
```

### Ultravox

Modelo open-source optimizado para conversación por voz.

```bash
# .env
BOT_TYPE=multimodal
LLM_PROVIDER=ultravox
ULTRAVOX_API_KEY=tu_clave_de_ultravox
```

### Comparación de Proveedores Multimodales

| Proveedor | Latencia | Calidad de Voz | Idiomas | Costo |
| :--- | :--- | :--- | :--- | :--- |
| **Gemini Live** | ~200ms | Excelente | Multilingüe | Bajo |
| **OpenAI Realtime** | ~150ms | Excelente | Multilingüe | Alto |
| **AWS Nova Sonic** | ~250ms | Muy buena | EN/ES | Medio |
| **Ultravox** | ~300ms | Buena | EN | Bajo |

---

## Integración con Asterisk (Telefonía)

Este proyecto incluye soporte para conectar Asterisk PBX mediante WebSockets (chan_websocket).

### 1. Iniciar el Servidor de Bots

```bash
# Iniciar en puerto 8765
uv run python asterisk_runner.py --port 8765

# O con proveedores específicos
uv run python asterisk_runner.py -l google -s deepgram -p cartesia
```

### 2. Configurar Asterisk

Añade a `/etc/asterisk/websocket_client.conf`:

```ini
[pipecat]
type = websocket_client
uri = ws://127.0.0.1:8765
protocols = media
connection_type = per_call_config
tls_enabled = no
```

Añade a `/etc/asterisk/extensions.conf`:

```ini
[ai-agents]
exten => 100,1,Answer()
same => n,Set(JITTERBUFFER(adaptive)=60,300,40)
same => n,Dial(WebSocket/pipecat/c(slin16)f(json))
same => n,Hangup()
```

Ver ejemplos completos en la carpeta `asterisk_config/`.

---

## API REST

### POST /connect

Crea una sala y lanza un bot. Acepta configuración dinámica en el body:

```bash
curl -X POST http://localhost:7860/connect \
  -H "Content-Type: application/json" \
  -d '{
    "bot_type": "flow",
    "llm_provider": "anthropic",
    "tts_provider": "cartesia",
    "tts_voice": "79a125e8-cd45-4c13-8a67-188112f4dd22"
  }'
```

**Respuesta:**
```json
{
  "room_url": "https://yoursubdomain.daily.co/xxx",
  "token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "bot_pid": 12345,
  "status_endpoint": "/status/12345"
}
```

### GET /status/{pid}

Verifica el estado de un bot:

```bash
curl http://localhost:7860/status/12345
```

---

## Variables de Entorno

| Proveedor | Variable de Entorno | Descripción |
| :--- | :--- | :--- |
| **Daily** | `DAILY_API_KEY` | Requerido para crear salas |
| **Google** | `GOOGLE_API_KEY` | Para Gemini y Gemini Live |
| **OpenAI** | `OPENAI_API_KEY` | Para GPT-4o y Realtime |
| **Anthropic** | `ANTHROPIC_API_KEY` | Para Claude 3.5 |
| **Deepgram** | `DEEPGRAM_API_KEY` | Para STT/TTS |
| **Cartesia** | `CARTESIA_API_KEY` | Para TTS en español |
| **ElevenLabs** | `ELEVENLABS_API_KEY` | Para TTS de alta calidad |
| **Groq** | `GROQ_API_KEY` | Para LLM/STT ultra-rápidos |
| **PlayHT** | `PLAYHT_API_KEY`, `PLAYHT_USER_ID` | Para TTS |
| **AWS** | `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY` | Para Nova Sonic |
| **Ultravox** | `ULTRAVOX_API_KEY` | Para multimodal Ultravox |

### Configuración por Defecto

```bash
# Bot
BOT_NAME=Marissa
BOT_TYPE=flow  # flow, simple, multimodal

# Proveedores
LLM_PROVIDER=google
STT_PROVIDER=deepgram
TTS_PROVIDER=deepgram

# Modelos
LLM_MODEL=gemini-2.0-flash
LLM_TEMPERATURE=0.7
```

---

## Argumentos CLI del Runner

| Argumento | Descripción | Ejemplo |
| :--- | :--- | :--- |
| `-u, --room-url` | URL de la sala Daily (requerido) | `-u https://x.daily.co/sala` |
| `-t, --token` | Token de autenticación (requerido) | `-t eyJ...` |
| `-b, --bot-type` | Tipo de bot | `-b multimodal` |
| `-l, --llm-provider` | Proveedor LLM | `-l anthropic` |
| `-m, --llm-model` | Modelo específico | `-m gpt-4o-mini` |
| `-T, --llm-temperature` | Temperatura del LLM | `-T 0.5` |
| `-s, --stt-provider` | Proveedor STT | `-s gladia` |
| `-p, --tts-provider` | Proveedor TTS | `-p elevenlabs` |
| `-v, --tts-voice` | ID de voz TTS | `-v voice_id` |

---

## Ejecución con Docker

```bash
# Construir
docker build -t tito-ai-demo .

# Ejecutar
docker run --env-file .env -p 7860:7860 tito-ai-demo
```

---

## Tipos de Bot

### `simple`
Bot directo sin gestión de estados. Ideal para pruebas rápidas y conversaciones simples.

```bash
python runner.py -u [URL] -t [TOKEN] -b simple
```

### `flow`
Bot basado en nodos con `pipecat-flows`. Guía al usuario a través de un cuestionario estructurado con estados y transiciones.

```bash
python runner.py -u [URL] -t [TOKEN] -b flow
```

### `multimodal`
Bot de voz a voz nativo. El modelo (Gemini Live, OpenAI Realtime, etc.) recibe y genera audio directamente, sin pasar por STT/TTS intermedios.

```bash
python runner.py -u [URL] -t [TOKEN] -b multimodal -l google
```

---

## Troubleshooting

### Error: "Missing API key"

Asegúrate de que las variables de entorno estén configuradas en `.env`:

```bash
# Verificar que .env tiene las claves
cat .env | grep API_KEY
```

### Error: "Cannot import..."

Instala las dependencias correctamente:

```bash
uv sync
# o
pip install -e ".[dev]"
```

### El bot no responde

1. Verifica que `DAILY_API_KEY` esté configurado
2. Asegúrate de que el proveedor LLM tiene su API key
3. Revisa los logs del servidor para errores específicos

---

## Estructura del Proyecto

```
demo/
├── main.py          # Servidor FastAPI
├── runner.py        # CLI para ejecutar bots
├── bots/
│   ├── base_bot.py  # Clase base abstracta
│   ├── simple.py    # Bot simple
│   ├── flow.py      # Bot de flujo
│   └── multimodal.py # Bot multimodal
├── config/
│   └── bot.py       # Configuración centralizada
├── prompts/
│   └── *.py         # System prompts
├── .env.example     # Plantilla de variables
└── pyproject.toml   # Dependencias
```

---

## Recursos

- [Documentación de Pipecat](https://docs.pipecat.ai)
- [Pipecat Flows](https://github.com/pipecat-ai/pipecat-flows)
- [Daily.co API](https://docs.daily.co/reference)
