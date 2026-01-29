# Tito.AI API Documentation

## Descripción General

La API de Tito.AI proporciona endpoints para gestionar asistentes de voz con IA, crear llamadas, administrar campañas y recibir webhooks. Está construida con FastAPI y soporta múltiples proveedores de LLM, STT y TTS.

**Base URL:** `http://localhost:7860`

---

## Autenticación

La API actualmente no requiere autenticación. Todos los endpoints son públicos.

---

## Endpoints

### 1. Gestión de Asistentes

#### GET /assistants
Obtiene la lista de todos los asistentes disponibles.

**Respuesta:**
```json
[
  {
    "id": "uuid",
    "name": "Assistant Name",
    "description": "Description",
    "created_at": "2026-01-28T13:20:19.076Z",
    "version": "1.0.0",
    "architecture_type": "flow",
    "_links": [
      {
        "href": "/assistants/uuid",
        "method": "GET",
        "rel": "self"
      }
    ]
  }
]
```

#### POST /assistants
Crea un nuevo asistente.

**Body:**
```json
{
  "name": "Mi Asistente",
  "description": "Descripción del asistente",
  "architecture_type": "flow",
  "agent": {
    "system_prompt": "Eres un asistente útil",
    "provider": "google",
    "model": "gemini-2.0-flash",
    "temperature": 0.7
  },
  "io_layer": {
    "stt": {
      "provider": "deepgram",
      "language": "es"
    },
    "tts": {
      "provider": "deepgram",
      "voice_id": "aura-luna-es"
    }
  }
}
```

#### GET /assistants/{assistant_id}
Obtiene los detalles de un asistente específico.

**Parámetros:**
- `assistant_id` (string): ID del asistente

#### DELETE /assistants/{assistant_id}
Elimina un asistente.

**Respuesta:**
```json
{
  "status": "success",
  "message": "Assistant {assistant_id} deleted"
}
```

### 2. Gestión de Llamadas

#### GET /
Endpoint para acceso directo desde navegador. Crea una sala Daily y redirige al usuario.

**Respuesta:** Redirección HTTP a la URL de la sala.

#### POST /calls
Inicia una nueva llamada con un asistente específico.

**Body:**
```json
{
  "assistant_id": "uuid",
  "phone_number": "+1234567890",
  "variables": {
    "customer_name": "Juan Pérez",
    "product": "Software"
  },
  "secrets": {
    "api_key": "secret_value"
  },
  "dynamic_vocabulary": ["palabra1", "palabra2"]
}
```

**Respuesta:**
```json
{
  "id": "12345",
  "status": "initiated",
  "room_url": "https://domain.daily.co/room",
  "token": "jwt_token",
  "_links": [
    {
      "href": "/status/12345",
      "method": "GET",
      "rel": "status"
    }
  ]
}
```

#### POST /connect
Endpoint dinámico para conexión RTVI con configuración inline.

**Body:**
```json
{
  "bot_type": "flow",
  "bot_name": "Marissa",
  "llm_provider": "google",
  "llm_model": "gemini-2.0-flash",
  "llm_temperature": 0.7,
  "stt_provider": "deepgram",
  "tts_provider": "deepgram",
  "tts_voice": "aura-luna-es",
  "enable_stt_mute_filter": true
}
```

#### POST /assistants/{assistant_id}/connect
Conecta un cliente RTVI a un bot basado en un asistente.

**Parámetros:**
- `assistant_id` (string): ID del asistente

**Body (opcional):**
```json
{
  "variables": {
    "key": "value"
  }
}
```

#### GET /status/{pid}
Verifica el estado de un proceso de bot.

**Parámetros:**
- `pid` (integer): ID del proceso

**Respuesta:**
```json
{
  "bot_id": 12345,
  "status": "running"
}
```

### 3. Gestión de Campañas

#### GET /campaigns
Obtiene la lista de todas las campañas.

**Respuesta:**
```json
[
  {
    "id": "campaign_id",
    "name": "Campaign Name",
    "assistant_id": "uuid",
    "contacts": [
      {
        "name": "Contact Name",
        "phone": "+1234567890"
      }
    ]
  }
]
```

#### POST /campaigns
Crea una nueva campaña.

**Body:**
```json
{
  "id": "campaign_id",
  "name": "Mi Campaña",
  "assistant_id": "uuid",
  "contacts": [
    {
      "name": "Juan Pérez",
      "phone": "+1234567890"
    }
  ]
}
```

#### POST /campaigns/{campaign_id}/start
Inicia la ejecución de una campaña en segundo plano.

**Parámetros:**
- `campaign_id` (string): ID de la campaña

**Respuesta:**
```json
{
  "message": "Campaign campaign_id started in background"
}
```

#### GET /campaigns/{campaign_id}
Obtiene los detalles de una campaña específica.

**Parámetros:**
- `campaign_id` (string): ID de la campaña

### 4. Webhooks

#### POST /webhook
Recibe webhooks de servicios externos.

**Body:** Cualquier JSON válido

**Respuesta:**
```json
{
  "status": "received"
}
```

---

## Códigos de Estado HTTP

| Código | Descripción |
|--------|-------------|
| 200 | Éxito |
| 201 | Creado |
| 302 | Redirección |
| 400 | Solicitud incorrecta |
| 404 | No encontrado |
| 422 | Error de validación |
| 500 | Error interno del servidor |

---

## Modelos de Datos

### AssistantConfig
```json
{
  "id": "string (UUID)",
  "name": "string",
  "description": "string (opcional)",
  "created_at": "datetime",
  "version": "string",
  "architecture_type": "simple | flow | multimodal",
  "pipeline_settings": "PipelineSettings",
  "agent": "AgentConfig",
  "io_layer": "IOLayerConfig",
  "webhooks": "WebhookConfig (opcional)",
  "flow": "FlowConfig (requerido para architecture_type: flow)"
}
```

### AgentConfig
```json
{
  "system_prompt": "string",
  "provider": "google | openai | anthropic | groq",
  "model": "string (opcional)",
  "temperature": "float",
  "tools": "array (opcional)"
}
```

### IOLayerConfig
```json
{
  "stt": {
    "provider": "deepgram | gladia | assemblyai | groq",
    "language": "string",
    "enable_mute_filter": "boolean"
  },
  "tts": {
    "provider": "deepgram | cartesia | elevenlabs | playht",
    "voice_id": "string",
    "language": "string",
    "speed": "float"
  }
}
```

### FlowConfig (Para architecture_type: "flow")
```json
{
  "initial_node": "string",
  "nodes": {
    "node_name": {
      "messages": [
        {
          "role": "system | user | assistant",
          "content": "string"
        }
      ],
      "functions": [
        {
          "name": "string",
          "description": "string",
          "properties": {},
          "required": ["string"],
          "handler": "string",
          "transition_callback": "string (opcional)"
        }
      ],
      "post_actions": []
    }
  }
}
```

---

## Variables de Entorno Requeridas

| Variable | Descripción | Requerido |
|----------|-------------|-----------|
| `DAILY_API_KEY` | Clave API de Daily.co | Sí |
| `GOOGLE_API_KEY` | Clave API de Google | Para Gemini |
| `OPENAI_API_KEY` | Clave API de OpenAI | Para GPT |
| `ANTHROPIC_API_KEY` | Clave API de Anthropic | Para Claude |
| `DEEPGRAM_API_KEY` | Clave API de Deepgram | Para STT/TTS |
| `CARTESIA_API_KEY` | Clave API de Cartesia | Para TTS |
| `ELEVENLABS_API_KEY` | Clave API de ElevenLabs | Para TTS |
| `GROQ_API_KEY` | Clave API de Groq | Para LLM/STT |

---

## Ejemplos de Uso

### Crear un asistente y hacer una llamada
```bash
# 1. Crear asistente de tipo flow
curl -X POST http://localhost:7860/assistants \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Asistente de Ventas",
    "architecture_type": "flow",
    "agent": {
      "system_prompt": "Eres un asistente de ventas profesional",
      "provider": "google",
      "model": "gemini-2.0-flash"
    },
    "flow": {
      "initial_node": "greeting",
      "nodes": {
        "greeting": {
          "messages": [
            {
              "role": "system",
              "content": "Saluda al cliente y pregunta su nombre."
            }
          ],
          "functions": [
            {
              "name": "collect_name",
              "description": "Recopilar el nombre del cliente",
              "properties": {
                "name": {
                  "type": "string",
                  "description": "Nombre del cliente"
                }
              },
              "required": ["name"],
              "handler": "handle_name_collection",
              "transition_callback": "transition_to_qualification"
            }
          ]
        },
        "qualification": {
          "messages": [
            {
              "role": "system", 
              "content": "Califica al cliente preguntando sobre su empresa."
            }
          ],
          "functions": [
            {
              "name": "qualify_lead",
              "description": "Calificar el lead",
              "properties": {
                "company_size": {"type": "string"},
                "budget": {"type": "string"}
              },
              "required": ["company_size", "budget"],
              "handler": "handle_lead_qualification"
            }
          ]
        }
      }
    }
  }'

# 2. Iniciar llamada
curl -X POST http://localhost:7860/calls \
  -H "Content-Type: application/json" \
  -d '{
    "assistant_id": "uuid-del-asistente",
    "variables": {
      "customer_name": "Juan Pérez"
    }
  }'
```

### Conexión rápida con configuración inline
```bash
curl -X POST http://localhost:7860/connect \
  -H "Content-Type: application/json" \
  -d '{
    "bot_type": "multimodal",
    "llm_provider": "google"
  }'
```

---

## Configuración de Flows

Los asistentes de tipo `flow` utilizan una configuración basada en nodos que define el flujo de conversación estructurado.

### Estructura de Flow

```json
{
  "initial_node": "nombre_nodo_inicial",
  "nodes": {
    "nombre_nodo": {
      "messages": [...],
      "functions": [...],
      "post_actions": [...]
    }
  }
}
```

### Componentes de un Nodo

#### Messages
Define los mensajes del sistema para el nodo:
```json
"messages": [
  {
    "role": "system",
    "content": "Instrucciones específicas para este nodo"
  }
]
```

#### Functions
Define las funciones que el asistente puede llamar en este nodo:
```json
"functions": [
  {
    "name": "nombre_funcion",
    "description": "Descripción de la función",
    "properties": {
      "parametro": {
        "type": "string",
        "description": "Descripción del parámetro"
      }
    },
    "required": ["parametro"],
    "handler": "nombre_del_handler",
    "transition_callback": "callback_transicion_opcional"
  }
]
```

#### Handlers Disponibles
- `handle_name_collection` - Recopilar información del cliente
- `handle_lead_qualification` - Calificar leads
- `handle_followup_scheduling` - Programar seguimientos
- `handle_problem_description` - Describir problemas
- `handle_ticket_result` - Gestionar tickets

#### Ejemplo de Flow Completo
```json
{
  "initial_node": "greeting",
  "nodes": {
    "greeting": {
      "messages": [
        {
          "role": "system",
          "content": "Saluda cordialmente y pregunta el nombre del cliente."
        }
      ],
      "functions": [
        {
          "name": "collect_customer_info",
          "description": "Recopilar información básica del cliente",
          "properties": {
            "name": {
              "type": "string",
              "description": "Nombre completo del cliente"
            },
            "company": {
              "type": "string", 
              "description": "Nombre de la empresa"
            }
          },
          "required": ["name"],
          "handler": "handle_name_collection",
          "transition_callback": "transition_to_qualification"
        }
      ]
    },
    "qualification": {
      "messages": [
        {
          "role": "system",
          "content": "Califica al cliente preguntando sobre necesidades y presupuesto."
        }
      ],
      "functions": [
        {
          "name": "assess_needs",
          "description": "Evaluar las necesidades del cliente",
          "properties": {
            "pain_points": {
              "type": "array",
              "description": "Puntos de dolor identificados"
            },
            "budget_range": {
              "type": "string",
              "description": "Rango de presupuesto"
            },
            "timeline": {
              "type": "string",
              "description": "Cronograma esperado"
            }
          },
          "required": ["pain_points", "budget_range"],
          "handler": "handle_lead_qualification",
          "transition_callback": "transition_to_closing"
        }
      ]
    },
    "closing": {
      "messages": [
        {
          "role": "system",
          "content": "Cierra la conversación y programa próximos pasos."
        }
      ],
      "functions": [
        {
          "name": "schedule_next_steps",
          "description": "Programar próximos pasos",
          "properties": {
            "next_action": {
              "type": "string",
              "description": "Próxima acción a realizar"
            },
            "contact_preference": {
              "type": "string",
              "description": "Método de contacto preferido"
            }
          },
          "required": ["next_action"],
          "handler": "handle_followup_scheduling"
        }
      ]
    }
  }
}
```
