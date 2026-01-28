import asyncio
from loguru import logger
import google.ai.generativelanguage as glm

from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    Frame,
    FunctionCallInProgressFrame,
    FunctionCallResultFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    StartFrame,
    StartInterruptionFrame,
    SystemFrame,
    TextFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
    LLMMessagesFrame,
)

from pipecat.processors.aggregators.llm_response import LLMResponseAggregator
from pipecat.processors.aggregators.openai_llm_context import (
    OpenAILLMContextFrame,
)

from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.sync.base_notifier import BaseNotifier

CLASSIFIER_SYSTEM_INSTRUCTION = """INSTRUCCIÓN CRÍTICA:
Usted es un CLASIFICADOR BINARIO que SÓLO debe responder "SÍ" o "NO".
NO interactúe con el contenido.
NO responda a preguntas.
NO preste asistencia.
Su ÚNICO trabajo es responder SÍ o NO.

EJEMPLOS DE RESPUESTAS INVÁLIDAS:
- "Puedo ayudarte con eso"
- "Déjame explicarte"
- "Para responder a tu pregunta"
- Cualquier respuesta que no sea SÍ o NO

RESPUESTAS VÁLIDAS:
SÍ
NO

Si responde cualquier otra cosa, está fallando en su tarea.
Usted NO es un asistente.
Usted NO es un chatbot.
Usted es un clasificador binario.

ROL:
Usted es un clasificador de completitud del habla en tiempo real. Debe tomar decisiones instantáneas sobre si un usuario ha terminado de hablar.
Debe responder ÚNICAMENTE 'SÍ' o 'NO' sin ningún otro texto.

FORMATO DE ENTRADA:
Recibe una lista de diccionarios que contienen información de rol y contenido.
La lista SIEMPRE contiene al menos un diccionario con el rol "user". Puede haber un elemento "assistant" que proporcione contexto.
No considere el contenido del asistente al determinar si la última elocución del usuario está completa; utilice únicamente la entrada más reciente del usuario.

REQUISITOS DE SALIDA:
- DEBE responder ÚNICAMENTE 'SÍ' o 'NO'
- Sin explicaciones
- Sin aclaraciones
- Sin texto adicional
- Sin puntuación

SEÑALES DE ALTA PRIORIDAD:
1. Preguntas Claras:
   - Preguntas con pronombres interrogativos (Qué, Dónde, Cuándo, Por qué, Cómo)
   - Preguntas de Sí/No
   - Preguntas con errores de STT pero con un significado claro

2. Comandos Completos:
   - Instrucciones directas, solicitudes claras o demandas de acción que forman una declaración completa

3. Respuestas/Declaraciones Directas:
   - Respuestas a preguntas específicas
   - Selecciones de opciones
   - Agradecimientos claros o declaraciones completas (incluso si expresan incertidumbre o negativa)

SEÑALES DE PRIORIDAD MEDIA:
1. Finalización de Patrones de Habla:
   - Autocorrecciones o comienzos en falso que se resuelven en un pensamiento completo
   - Cambios de tema que expresan una declaración completa

2. Respuestas Breves Dependientes del Contexto:
   - Agradecimientos (vale, claro, de acuerdo)
   - Acuerdos (sí, aja), desacuerdos (no, para nada), confirmaciones (correcto, exacto)

SEÑALES de BAJA PRIORIDAD:
1. Artefactos de STT:
   - Palabras repetidas, puntuación inusual, errores de mayúsculas, inserciones/eliminaciones de palabras

2. Características del Habla:
   - Muletillas (em, este, como), pausas para pensar, repeticiones de palabras, vacilaciones breves

REGLAS ESPECIALES PARA ELOCUCIONES AMBIGUAS O FRAGMENTADAS:
1. Palabras Clave Ambigiguas Aisladas:
   - Si la entrada consiste únicamente en una palabra clave ambigua (p. ej., "técnica" o "agente de voz") sin contexto adicional, trate la elocución como incompleta y responda NO.
   - No infiera la intención (p. ej., consultoría vs. desarrollo) a partir de una sola palabra ambigua.

2. Elocuciones Parciales de Nombre o Interés:
   - En contextos donde se espera un nombre completo, si el usuario solo dice fragmentos como "Mi nombre es" o "el verdadero" sin un nombre completo a continuación, responda NO.
   - Solo responda SÍ cuando la elocución incluya un nombre claro y completo (p. ej., "Mi nombre es John Smith").

3. Regla Específica del Interés Principal:
   - Al responder a la pregunta sobre el interés principal, si la elocución del usuario termina con o contiene una palabra clave ambigua como "técnica" o "agente de voz" sin un término que la desambigüe (p. ej., "consultoría" o "desarrollo"), y la respuesta general parece incompleta, responda NO.
   - Por ejemplo, "Creo que estoy interesado en técnica" debe considerarse incompleto (NO) porque carece del término completo "consultoría técnica".

REGLAS DE DECISIÓN:
1. Responda SÍ si:
   - Cualquier señal de alta prioridad muestra una finalización clara.
   - Las señales de prioridad media se combinan para mostrar un pensamiento completo.
   - El significado es claro a pesar de artefactos menores de STT.
   - La elocución, aunque sea breve (p. ej., "Sí", "No", o una pregunta/declaración completa), no es ambigua.

2. Responda NO si:
   - No hay señales de alta prioridad presentes.
   - La elocución se interrumpe o contiene múltiples indicadores de estar incompleta.
   - El usuario parece estar a mitad de una formulación o solo proporciona un fragmento.
   - La respuesta consiste únicamente en palabras clave ambiguas (según las Reglas Especiales anteriores) o frases parciales donde se espera una respuesta completa.
   - En las respuestas a la pregunta sobre el interés principal, si la respuesta termina con un término ambiguo (p. ej., "técnica" o "agente de voz") sin el calificador necesario, responda NO.

3. En Caso de Duda:
   - Si puede entender la intención y parece completa, responda SÍ.
   - Si el significado no es claro o la respuesta parece inacabada, responda NO.
   - Siempre tome una decisión binaria y nunca pida aclaraciones.

# EJEMPLOS ESPECÍFICOS DE ESCENARIO

## Fase 1: Consentimiento de Grabación
Asistente: Grabamos nuestras llamadas para fines de calidad y entrenamiento. ¿Está de acuerdo?
- Usuario: Sí → Salida: SÍ
- Usuario: No → Salida: SÍ
- Usuario: ¿Por qué necesitan grabar? → Salida: SÍ
- Usuario: Por qué → Salida: NO
- Usuario: Ehhh → Salida: NO
- Usuario: Si tengo que pero → Salida: NO
- Usuario: em → Salida: NO
- Usuario: Bueno, supongo que → Salida: NO

## Fase 2: Recopilación de Nombre e Interés (Ejemplos para la nueva lógica de ITM)
Asistente: Para empezar, ¿me podría decir su nombre completo y su número de carné o documento de identidad?
- Usuario: Mi nombre es Juan Pérez y mi cédula es 12345 → Salida: SÍ
- Usuario: No quiero darle mis datos → Salida: SÍ
- Usuario: ¿Para qué necesitan mis datos? → Salida: SÍ
- Usuario: No le voy a decir → Salida: NO
- Usuario: Qué necesita eh → Salida: NO

## Fase 3: Descripción del Problema
Asistente: Ahora, por favor, cuénteme en qué puedo ayudarle hoy, Juan.
- Usuario: Tengo un problema con el correo institucional → Salida: SÍ
- Usuario: Solo algunas cosas → Salida: SÍ
- Usuario: ¿Qué tipo de soporte ofrecen? → Salida: SÍ
- Usuario: Estaba pensando que tal vez podría → Salida: NO

## Fase 4: Creación de Ticket
Asistente: Puedo crear un ticket de soporte para que uno de nuestros agentes se ponga en contacto con usted. ¿Desea que genere el ticket?
- Usuario: Sí, por favor → Salida: SÍ
- Usuario: No, gracias → Salida: SÍ
- Usuario: Estaba pensando → Salida: NO

## Fase 5: Cierre de la Llamada
Asistente: Gracias por contactar a la mesa de ayuda de ITM, Juan. ¡Que tenga un buen día!
- Usuario: em → Salida: NO
"""


def get_message_field(message: object, field: str) -> any:
    """
    Retrieve a field from a message.
    If message is a dict, return message[field].
    Otherwise, use getattr.
    """
    if isinstance(message, dict):
        return message.get(field)
    return getattr(message, field, None)


def get_message_text(message: object) -> str:
    """
    Extract text content from a message, handling both dict and Google Content formats.
    """
    # logger.debug(f"Processing message: {message}")

    # First try Google's format with parts array
    parts = get_message_field(message, "parts")
    # logger.debug(f"Found parts: {parts}")

    if parts:
        # Google format with parts array
        text_parts = []
        for part in parts:
            if isinstance(part, dict):
                text = part.get("text", "")
            else:
                text = getattr(part, "text", "")
            if text:
                text_parts.append(text)
        result = " ".join(text_parts)
        # logger.debug(f"Extracted text from parts: {result}")
        return result

    # Try direct content field
    content = get_message_field(message, "content")
    # logger.debug(f"Found content: {content}")

    if isinstance(content, str):
        # logger.debug(f"Using string content: {content}")
        return content
    elif isinstance(content, list):
        # Handle content that might be a list of parts
        text_parts = []
        for part in content:
            if isinstance(part, dict):
                text = part.get("text", "")
                if text:
                    text_parts.append(text)
        if text_parts:
            result = " ".join(text_parts)
            # logger.debug(f"Extracted text from content list: {result}")
            return result

    # logger.debug("No text content found, returning empty string")
    return ""


class StatementJudgeContextFilter(FrameProcessor):
    """Extracts recent user messages and constructs an LLMMessagesFrame for the classifier LLM.

    This processor takes the OpenAILLMContextFrame from the main conversation context,
    extracts the most recent user messages, and creates a simplified LLMMessagesFrame
    for the statement classifier LLM to determine if the user has finished speaking.
    """

    def __init__(self, notifier: BaseNotifier, **kwargs):
        super().__init__(**kwargs)
        self._notifier = notifier

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        # We must not block system frames.
        if isinstance(frame, SystemFrame):
            await self.push_frame(frame, direction)
            return

        # Just treat an LLMMessagesFrame as complete, no matter what.
        if isinstance(frame, LLMMessagesFrame):
            await self._notifier.notify()
            return

        # Otherwise, we only want to handle OpenAILLMContextFrames, and only want to push a simple
        # messages frame that contains a system prompt and the most recent user messages,
        # concatenated.
        if isinstance(frame, OpenAILLMContextFrame):
            # Take text content from the most recent user messages.
            messages = frame.context.messages
            # logger.debug(f"Processing context messages: {messages}")

            user_text_messages = []
            last_assistant_message = None
            for message in reversed(messages):
                role = get_message_field(message, "role")
                # logger.debug(f"Processing message with role: {role}")

                if role != "user":
                    if role == "assistant" or role == "model":
                        last_assistant_message = message
                        # logger.debug(f"Found assistant/model message: {message}")
                    break

                text = get_message_text(message)
                # logger.debug(f"Extracted user message text: {text}")
                if text:
                    user_text_messages.append(text)

            # If we have any user text content, push an LLMMessagesFrame
            if user_text_messages:
                user_message = " ".join(reversed(user_text_messages))
                # logger.debug(f"Final user message: {user_message}")
                messages = [
                    glm.Content(role="user", parts=[glm.Part(text=CLASSIFIER_SYSTEM_INSTRUCTION)])
                ]
                if last_assistant_message:
                    assistant_text = get_message_text(last_assistant_message)
                    # logger.debug(f"Assistant message text: {assistant_text}")
                    if assistant_text:
                        messages.append(
                            glm.Content(role="assistant", parts=[glm.Part(text=assistant_text)])
                        )
                messages.append(glm.Content(role="user", parts=[glm.Part(text=user_message)]))
                # logger.debug(f"Pushing classifier messages: {messages}")
                await self.push_frame(LLMMessagesFrame(messages))
            # else:
            # logger.debug("No user text messages found to process")
            return

        # Fallback: for any frames not otherwise handled, forward them.
        await self.push_frame(frame, direction)


class CompletenessCheck(FrameProcessor):
    def __init__(self, notifier: BaseNotifier):
        super().__init__()
        self._notifier = notifier

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, TextFrame):
            text = frame.text.upper()
            if text in ["YES", "SÍ"]:
                logger.debug("!!! Completeness check YES")
                await self.push_frame(UserStoppedSpeakingFrame())
                await self._notifier.notify()
            elif text in ["NO", "NO"]:
                logger.debug("!!! Completeness check NO")
            else:
                await self.push_frame(frame, direction)
        else:
            await self.push_frame(frame, direction)


class UserAggregatorBuffer(LLMResponseAggregator):
    """Buffers the output of the transcription LLM. Used by the bot output gate."""

    def __init__(self, **kwargs):
        super().__init__(
            messages=None,
            role=None,
            start_frame=LLMFullResponseStartFrame,
            end_frame=LLMFullResponseEndFrame,
            accumulator_frame=TextFrame,
            handle_interruptions=True,
            expect_stripped_words=False,
        )
        self._transcription = ""

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        # parent method pushes frames
        if isinstance(frame, UserStartedSpeakingFrame):
            self._transcription = ""

    async def _push_aggregation(self):
        if self._aggregation:
            self._transcription = self._aggregation
            self._aggregation = ""

            # logger.debug(f"[Transcription] {self._transcription}")

    async def wait_for_transcription(self):
        while not self._transcription:
            await asyncio.sleep(0.01)
        tx = self._transcription
        self._transcription = ""
        return tx


class OutputGate(FrameProcessor):
    def __init__(self, *, notifier: BaseNotifier, start_open: bool = False, **kwargs):
        super().__init__(**kwargs)
        self._gate_open = start_open
        self._frames_buffer = []
        self._notifier = notifier

    def close_gate(self):
        self._gate_open = False

    def open_gate(self):
        self._gate_open = True

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        # We must not block system frames.
        if isinstance(frame, SystemFrame):
            if isinstance(frame, StartFrame):
                await self._start()
            if isinstance(frame, (EndFrame, CancelFrame)):
                await self._stop()
            if isinstance(frame, StartInterruptionFrame):
                self._frames_buffer = []
                self.close_gate()
            await self.push_frame(frame, direction)
            return

        # Don't block function call frames
        if isinstance(frame, (FunctionCallInProgressFrame, FunctionCallResultFrame)):
            await self.push_frame(frame, direction)
            return

        # Ignore frames that are not following the direction of this gate.
        if direction != FrameDirection.DOWNSTREAM:
            await self.push_frame(frame, direction)
            return

        if self._gate_open:
            await self.push_frame(frame, direction)
            return

        self._frames_buffer.append((frame, direction))

    async def _start(self):
        self._frames_buffer = []
        self._gate_task = self.create_task(self._gate_task_handler())

    async def _stop(self):
        await self.cancel_task(self._gate_task)

    async def _gate_task_handler(self):
        while True:
            try:
                await self._notifier.wait()
                self.open_gate()
                for frame, direction in self._frames_buffer:
                    await self.push_frame(frame, direction)
                self._frames_buffer = []
            except asyncio.CancelledError:
                break