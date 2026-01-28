"""
Un flujo dinámico de un asistente de Mesa de Ayuda para estudiantes del
Instituto Tecnológico Metropolitano (ITM) usando Pipecat Flows.

Este ejemplo demuestra un sistema de soporte específico para estudiantes:

1. Saludo e identificación de la intención (Problema técnico, Consulta académica, Verificar ticket).
2. Recolección de detalles para un nuevo ticket técnico.
3. Triage de "Consultas académicas" para redirigir o crear un ticket si es un problema de acceso.
4. Verificación del estado de un ticket existente.
5. Bucle de conversación para manejar múltiples solicitudes.

Requisitos (igual que el ejemplo de restaurante):
- CARTESIA_API_KEY (para TTS)
- DEEPGRAM_API_KEY (para STT)
- DAILY_API_KEY (para transporte)
- LLM API key (varía según el proveedor - ver env.example)
"""

import asyncio
import os
import random
import sys

from dotenv import load_dotenv
from loguru import logger
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.transcriptions.language import Language

from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import LLMContextAggregatorPair
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import create_transport
from pipecat.services.cartesia.tts import CartesiaTTSService
from pipecat.services.deepgram.stt import DeepgramSTTService, LiveOptions
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.transports.daily.transport import DailyParams
from pipecat.transports.websocket.fastapi import FastAPIWebsocketParams
from pipecat.utils.text.markdown_text_filter import MarkdownTextFilter
from pipecat.services.elevenlabs.tts import ElevenLabsTTSService
from pipecat.services.elevenlabs.stt import ElevenLabsSTTService

# Asumimos que utils.py está en el mismo directorio que este script
# (copiado del repositorio de ejemplos)
try:
    from utils import create_llm
except ImportError:
    logger.error("Asegúrate de tener el archivo 'utils.py' de los ejemplos de pipecat-flows en el mismo directorio.")
    sys.exit(1)

from pipecat_flows import FlowArgs, FlowManager, FlowResult, FlowsFunctionSchema, NodeConfig

load_dotenv(override=True)

transport_params = {
    "daily": lambda: DailyParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
        vad_analyzer=SileroVADAnalyzer(),
    ),
    "twilio": lambda: FastAPIWebsocketParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
        vad_analyzer=SileroVADAnalyzer(),
    ),
    "webrtc": lambda: TransportParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
        vad_analyzer=SileroVADAnalyzer(),
    ),
}


# --- Sistema Simulado de Help Desk ITM ---
class MockHelpDeskSystemITM:
    """Simula un sistema de tickets de TI para el ITM."""

    def __init__(self):
        # Base de datos simulada de tickets de estudiantes
        self.tickets_db = {
            "72001": {"status": "En Progreso", "description": "No puedo ingresar al portal de estudiantes SI@U"},
            "72002": {"status": "Resuelto", "description": "Problemas con el Wi-Fi de la biblioteca en Fraternidad"},
        }
        self.next_ticket_id = 72003

    async def check_status(self, ticket_id: str) -> dict:
        """Verifica el estado de un ticket."""
        await asyncio.sleep(0.5)  # Simula latencia
        ticket_id_clean = ticket_id.replace(" ", "").strip()
        return self.tickets_db.get(
            ticket_id_clean, {"status": "No Encontrado", "description": ""}
        )

    async def create_ticket(self, description: str, student_id: str = "no_especificado") -> str:
        """Crea un nuevo ticket."""
        await asyncio.sleep(1.0)  # Simula latencia
        ticket_id = str(self.next_ticket_id)
        self.tickets_db[ticket_id] = {"status": "Nuevo", "description": description, "student_id": student_id}
        self.next_ticket_id += 1
        logger.info(f"Nuevo ticket creado: {ticket_id} para estudiante {student_id}")
        return ticket_id


# Inicializa el sistema simulado
help_desk_system = MockHelpDeskSystemITM()


# --- Definiciones de Tipos de Resultados ---
class IntentResult(FlowResult):
    intent: str  # "problema_tecnico", "consulta_academica", "verificar_ticket", "end"
    status: str = "success"


class IssueDetailsResult(FlowResult):
    description: str
    student_id: str | None = None
    status: str = "success"


class StatusResult(FlowResult):
    ticket_id: str
    status: str
    description: str


class TicketCreatedResult(FlowResult):
    ticket_id: str
    description: str
    status: str


# --- Handlers de Funciones ---

async def handle_intent(args: FlowArgs, flow_manager: "FlowManager") -> tuple[IntentResult, NodeConfig]:
    """Maneja la intención inicial del estudiante y transiciona al siguiente nodo."""
    intent = args["intent"]
    result = IntentResult(intent=intent)
    next_node = None

    if intent == "problema_tecnico":
        logger.debug("Transicionando a 'recolectar detalles del problema técnico'")
        next_node = create_collect_issue_node()
    elif intent == "consulta_academica":
        logger.debug("Transicionando a 'manejar consulta académica'")
        next_node = create_academic_query_node()
    elif intent == "verificar_ticket":
        logger.debug("Transicionando a 'recolectar ID de ticket'")
        next_node = create_collect_ticket_id_node()
    elif intent == "end":
        logger.debug("Transicionando a 'finalizar'")
        next_node = create_end_node()
    else:
        logger.warning(f"Intención desconocida: {intent}, regresando al inicio")
        next_node = create_initial_node()

    return result, next_node


async def handle_technical_ticket(args: FlowArgs, flow_manager: "FlowManager") -> tuple[
    TicketCreatedResult, NodeConfig]:
    """Recolecta la descripción del problema y crea el ticket."""
    description = args["description"]
    student_id = args.get("student_id")  # Opcional

    # Llamar al sistema de help desk para crear el ticket
    ticket_id = await help_desk_system.create_ticket(description, student_id)

    result = TicketCreatedResult(
        ticket_id=ticket_id,
        description=description,
        status="created"
    )

    logger.debug(f"Ticket {ticket_id} creado. Transicionando a 'reportar ticket creado'.")
    next_node = create_ticket_created_node(ticket_id)
    return result, next_node


async def handle_status_check(args: FlowArgs, flow_manager: "FlowManager") -> tuple[StatusResult, NodeConfig]:
    """Verifica el estado del ticket y prepara el nodo de respuesta."""
    ticket_id = args["ticket_id"]

    # Verificar estado en el sistema
    ticket_info = await help_desk_system.check_status(ticket_id)

    result = StatusResult(
        ticket_id=ticket_id,
        status=ticket_info["status"],
        description=ticket_info["description"]
    )

    logger.debug(f"Estado del ticket {ticket_id}: {ticket_info['status']}. Transicionando a 'proveer estado'.")
    next_node = create_status_update_node(result)
    return result, next_node


async def handle_end(args: FlowArgs, flow_manager: "FlowManager") -> tuple[None, NodeConfig]:
    """Maneja el fin de la conversación."""
    return None, create_end_node()


# --- Esquemas de Funciones ---

identify_user_intent_itm_schema = FlowsFunctionSchema(
    name="identify_user_intent_itm",
    description="Determina la intención principal del estudiante.",
    properties={
        "intent": {
            "type": "string",
            "enum": ["problema_tecnico", "consulta_academica", "verificar_ticket", "end"],
            "description": "La intención del estudiante (ej. 'problema_tecnico' si no puede entrar a SI@U, 'consulta_academica' si pregunta por sus notas, 'verificar_ticket' si ya tiene un número de caso)."
        }
    },
    required=["intent"],
    handler=handle_intent,
)

crear_ticket_tecnico_schema = FlowsFunctionSchema(
    name="create_technical_ticket",
    description="Crea un nuevo ticket de soporte técnico con la descripción del problema.",
    properties={
        "description": {
            "type": "string",
            "description": "Una descripción detallada del problema técnico (ej. 'No me carga el portal SI@U', 'El wifi de fraternidad no conecta').",
        },
        "student_id": {
            "type": "string",
            "description": "El código de estudiante o número de documento de identidad del estudiante, si lo proporciona.",
            "optional": True
        }
    },
    required=["description"],
    handler=handle_technical_ticket,
)

verificar_estado_ticket_schema = FlowsFunctionSchema(
    name="get_ticket_status",
    description="Obtiene el estado de un ticket de soporte existente usando su ID.",
    properties={
        "ticket_id": {
            "type": "string",
            "description": "El ID del ticket (ej. '72001').",
        }
    },
    required=["ticket_id"],
    handler=handle_status_check,
)

end_conversation_schema = FlowsFunctionSchema(
    name="end_conversation",
    description="Finaliza la conversación cuando el estudiante ha terminado.",
    properties={},
    required=[],
    handler=handle_end,
)


# --- Configuraciones de Nodos ---

def create_initial_node() -> NodeConfig:
    """Nodo inicial: Saluda y pregunta la intención."""
    return {
        "name": "initial_greeting_itm",
        "role_messages": [
            {
                "role": "system",
                "content": "Eres un amigable y eficiente asistente de la Mesa de Ayuda Estudiantil del ITM (Instituto Tecnológico Metropolitano). Tu objetivo es ayudar a los estudiantes a solucionar problemas técnicos o guiarlos en consultas académicas. Habla de forma clara y amable. Esta es una conversación de voz."
            }
        ],
        "task_messages": [
            {
                "role": "system",
                "content": "Saluda al estudiante: '¡Hola! Bienvenido a la Mesa de Ayuda Estudiantil del ITM, soy tu asistente virtual. ¿Cómo puedo ayudarte hoy? ¿Tienes un problema técnico, una consulta académica, o quieres verificar el estado de un ticket existente?'"
            }
        ],
        "functions": [identify_user_intent_itm_schema, end_conversation_schema],
    }


def create_collect_issue_node() -> NodeConfig:
    """Nodo: Pide al estudiante que describa su problema técnico."""
    return {
        "name": "collect_technical_issue_details",
        "task_messages": [
            {
                "role": "system",
                "content": "Entendido, un problema técnico. Para crear un ticket de soporte, por favor descríbeme el problema en detalle. ¿Qué no funciona y en qué campus o plataforma ocurre? (ej. 'No me funciona el wifi en la biblioteca de Fraternidad' o 'No puedo entrar a SI@U'). Si tienes tu código de estudiante a mano, también me lo puedes dar.",
            }
        ],
        "functions": [crear_ticket_tecnico_schema],
    }


def create_academic_query_node() -> NodeConfig:
    """Nodo: Maneja consultas académicas (Triage)."""
    return {
        "name": "handle_academic_query",
        "task_messages": [
            {
                "role": "system",
                "content": "Entendido, es una consulta académica. Para temas de notas, horarios, registro de materias o certificados, debes dirigirte directamente al portal de estudiantes SI@U o contactar a Admisiones y Registro. Yo no puedo ver esa información.\n\nSin embargo, si tu problema es que NO PUEDES ACCEDER a SI@U o a otra plataforma, eso es un problema técnico y SÍ puedo ayudarte a crear un ticket. ¿Tu consulta es un problema de acceso técnico o es sobre tus datos académicos?"
            }
        ],
        # Ofrece las mismas intenciones para que el usuario pueda re-clasificar su problema
        "functions": [identify_user_intent_itm_schema, end_conversation_schema],
    }


def create_ticket_created_node(ticket_id: str) -> NodeConfig:
    """Nodo: Informa al estudiante que el ticket ha sido creado."""
    return {
        "name": "ticket_created_confirmation",
        "task_messages": [
            {
                "role": "system",
                "content": f"¡Perfecto! He generado el ticket de soporte técnico número {ticket_id}. Nuestro equipo lo revisará pronto. ¿Hay algo más en lo que pueda ayudarte hoy? (por ejemplo, una consulta académica o verificar otro ticket)."
            }
        ],
        "functions": [identify_user_intent_itm_schema, end_conversation_schema],
    }


def create_collect_ticket_id_node() -> NodeConfig:
    """Nodo: Pide al estudiante el ID de su ticket."""
    return {
        "name": "collect_ticket_id",
        "task_messages": [
            {
                "role": "system",
                "content": "Claro, para verificar el estado, por favor dime el número del ticket que quieres consultar.",
            }
        ],
        "functions": [verificar_estado_ticket_schema],
    }


def create_status_update_node(status: StatusResult) -> NodeConfig:
    """Nodo: Provee la actualización de estado del ticket."""

    if status.status == "No Encontrado":
        message = f"Lo siento, no pude encontrar ningún ticket con el ID {status.ticket_id}. ¿Podrías verificar el número? Si no, ¿puedo ayudarte con algo más?"
    else:
        message = f"El ticket {status.ticket_id} (sobre '{status.description}') está actualmente: '{status.status}'. ¿Puedo ayudarte con algo más?"

    return {
        "name": "provide_status_update",
        "task_messages": [
            {
                "role": "system",
                "content": message,
            }
        ],
        "functions": [identify_user_intent_itm_schema, end_conversation_schema],
    }


def create_end_node() -> NodeConfig:
    """Nodo final: Se despide."""
    return {
        "name": "end",
        "task_messages": [
            {
                "role": "system",
                "content": "¡Con gusto! Que tengas un excelente día en el ITM. Si necesitas algo más, no dudes en llamar.",
            }
        ],
        "post_actions": [{"type": "end_conversation"}],
    }


# --- Configuración Principal del Bot ---
# (Esta parte es casi idéntica a los ejemplos base)

async def run_bot(
        transport: BaseTransport, runner_args: RunnerArguments, wait_for_user: bool = False
):
    """Ejecuta el bot de Help Desk del ITM."""

    # tts = CartesiaTTSService(
    #     api_key=os.getenv("CARTESIA_API_KEY"),
    #     # voice_id="71a7ad14-091c-4e8e-a314-022ece01c121", # Voz estándar (Reemplazada)
    #     model_id="sonic-es", # Modelo de Cartesia optimizado para español
    #     voice_name="CO-M-1",   # Voz Masculina de Colombia (acento colombiano)
    #     text_filters=[MarkdownTextFilter()],
    # )

    tts = ElevenLabsTTSService(
        api_key=os.getenv("ELEVENLABS_API_KEY"),
        voice_id="J4vZAFDEcpenkMp3f3R9",  # Voz en español latinoamericano
        model="eleven_multilingual_v2",
        text_filters=[MarkdownTextFilter()],
    )

    stt = DeepgramSTTService(
        api_key=os.getenv("DEEPGRAM_API_KEY"),
        live_options=LiveOptions(
            language="multi",

        ),
        model="nova-2-general",
        language=Language.ES  # Usar enum en lugar de string
    )

    # tts = ElevenLabsTTSService(
    #     api_key=os.getenv("ELEVENLABS_API_KEY"),
    #     voice_id="J4vZAFDEcpenkMp3f3R9",
    #     model="eleven_multilingual_v2"
    # )

    llm = create_llm()

    context = LLMContext()
    context_aggregator = LLMContextAggregatorPair(context)

    pipeline = Pipeline(
        [
            transport.input(),
            stt,
            context_aggregator.user(),
            llm,
            tts,
            transport.output(),
            context_aggregator.assistant(),
        ]
    )

    task = PipelineTask(pipeline, params=PipelineParams(allow_interruptions=True))

    # Inicializa el FlowManager en modo dinámico
    flow_manager = FlowManager(
        task=task,
        llm=llm,
        context_aggregator=context_aggregator,
        transport=transport,
    )

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info("Cliente (estudiante) conectado")
        # Inicia la conversación con el nodo inicial
        await flow_manager.initialize(create_initial_node())

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info(f"Cliente (estudiante) desconectado")
        await task.cancel()

    runner = PipelineRunner(handle_sigint=runner_args.handle_sigint)
    await runner.run(task)


async def bot(runner_args: RunnerArguments):
    """Punto de entrada principal del bot."""
    wait_for_user = globals().get("WAIT_FOR_USER", False)
    transport = await create_transport(runner_args, transport_params)
    await run_bot(transport, runner_args, wait_for_user)


if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Bot de Mesa de Ayuda del ITM")
    parser.add_argument(
        "--wait-for-user",
        action="store_true",
        help="Si se establece, el bot esperará a que el usuario hable primero",
    )

    args, remaining = parser.parse_known_args()
    WAIT_FOR_USER = args.wait_for_user

    if "--wait-for-user" in sys.argv:
        sys.argv.remove("--wait-for-user")

    from pipecat.runner.run import main

    main()