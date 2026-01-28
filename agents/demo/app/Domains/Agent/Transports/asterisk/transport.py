"""Asterisk WebSocket Transport for Pipecat.

This module provides a Pipecat transport implementation for Asterisk's 
chan_websocket module, enabling voice AI agents to handle phone calls.

Based on PR #3229 from pipecat-ai/pipecat with improvements for:
- XOFF/XON flow control
- Generation counter for clean interruptions
- Real-time audio pacing
- Proper chunk sizing for ptime matching
"""

import asyncio
import json
import struct
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

from loguru import logger
from pydantic import BaseModel

from pipecat.frames.frames import (
    AudioRawFrame,
    CancelFrame,
    EndFrame,
    Frame,
    StartFrame,
    StartInterruptionFrame,
    StopInterruptionFrame,
    SystemFrame,
    TransportMessageFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.audio.vad.vad_analyzer import VADAnalyzer

from app.Utils.audio import AudioResampler

try:
    import websockets
    from websockets.server import WebSocketServerProtocol
    HAS_WEBSOCKETS = True
except ImportError:
    HAS_WEBSOCKETS = False
    logger.warning("websockets not installed. Run: pip install websockets")


class AsteriskCodec(str, Enum):
    """Supported audio codecs for Asterisk."""
    ULAW = "ulaw"
    ALAW = "alaw"
    SLIN = "slin"      # 8kHz signed linear
    SLIN16 = "slin16"  # 16kHz signed linear


class AsteriskEventType(str, Enum):
    """Asterisk WebSocket event types."""
    MEDIA_START = "MEDIA_START"
    MEDIA = "MEDIA"
    MEDIA_END = "MEDIA_END"
    DTMF = "DTMF"
    XOFF = "XOFF"
    XON = "XON"
    FLUSH_MEDIA = "FLUSH_MEDIA"


class AsteriskWSParams(TransportParams):
    """Configuration parameters for Asterisk WebSocket transport.
    
    Attributes:
        host: Host address to bind the WebSocket server.
        port: Port number for WebSocket connections.
        pipeline_sample_rate: Audio sample rate used by AI services (16kHz).
        ptime_ms: Packet time in milliseconds (20ms default).
    """
    host: str = "0.0.0.0"
    port: int = 8765
    pipeline_sample_rate: int = 16000
    ptime_ms: int = 20  # 20ms chunks


@dataclass
class DTMFFrame(Frame):
    """Frame representing a DTMF digit received from Asterisk."""
    digit: str
    call_id: Optional[str] = None


@dataclass
class HangupFrame(SystemFrame):
    """Frame indicating the call has been hung up."""
    call_id: Optional[str] = None


@dataclass
class CallStartFrame(SystemFrame):
    """Frame indicating a new call has started."""
    call_id: str
    codec: str = "slin16"
    sample_rate: int = 16000


class AsteriskWSInputTransport(FrameProcessor):
    """Input transport for receiving audio from Asterisk via WebSocket."""
    
    def __init__(
        self,
        params: AsteriskWSParams,
        name: Optional[str] = None,
        **kwargs
    ):
        super().__init__(name=name or "AsteriskWSInput", **kwargs)
        self.params = params
        self._call_id: Optional[str] = None
        self._codec: str = "slin16"
        self._sample_rate: int = 16000
        self._resampler: Optional[AudioResampler] = None
        self._running = False
        
    def set_media_params(self, codec: str, sample_rate: int, call_id: str):
        """Set media parameters from MEDIA_START event."""
        self._codec = codec
        self._sample_rate = sample_rate
        self._call_id = call_id
        
        # Create resampler if needed
        if sample_rate != self.params.pipeline_sample_rate:
            self._resampler = AudioResampler(
                input_rate=sample_rate,
                output_rate=self.params.pipeline_sample_rate
            )
        else:
            self._resampler = None
            
        self._running = True
        logger.info(f"AsteriskWSInput: Media started - codec={codec}, rate={sample_rate}")
    
    async def handle_message(self, message: dict):
        """Handle a parsed WebSocket message from Asterisk."""
        event_type = message.get("event")
        
        if event_type == AsteriskEventType.MEDIA_START:
            codec = message.get("codec", "slin16")
            sample_rate = self._get_sample_rate(codec)
            call_id = message.get("channel", "unknown")
            self.set_media_params(codec, sample_rate, call_id)
            await self.push_frame(CallStartFrame(
                call_id=call_id,
                codec=codec,
                sample_rate=sample_rate
            ))
            
        elif event_type == AsteriskEventType.MEDIA:
            await self._handle_audio(message)
            
        elif event_type == AsteriskEventType.DTMF:
            digit = message.get("digit", "")
            if digit:
                logger.debug(f"AsteriskWSInput: DTMF received: {digit}")
                await self.push_frame(DTMFFrame(digit=digit, call_id=self._call_id))
                
        elif event_type == AsteriskEventType.MEDIA_END:
            logger.info("AsteriskWSInput: Media ended")
            self._running = False
            await self.push_frame(HangupFrame(call_id=self._call_id))
            await self.push_frame(EndFrame())
    
    async def _handle_audio(self, message: dict):
        """Process incoming audio from Asterisk."""
        import base64
        
        audio_b64 = message.get("media")
        if not audio_b64:
            return
            
        audio_data = base64.b64decode(audio_b64)
        
        # Decode codec if needed
        audio_pcm = self._decode_audio(audio_data)
        
        # Resample if needed
        if self._resampler:
            audio_pcm = self._resampler.resample(audio_pcm)
        
        frame = AudioRawFrame(
            audio=audio_pcm,
            sample_rate=self.params.pipeline_sample_rate,
            num_channels=1
        )
        await self.push_frame(frame)
    
    def _decode_audio(self, data: bytes) -> bytes:
        """Decode audio from Asterisk codec to PCM."""
        if self._codec in ("slin", "slin16"):
            return data  # Already PCM
        elif self._codec == "ulaw":
            return self._ulaw_to_pcm(data)
        elif self._codec == "alaw":
            return self._alaw_to_pcm(data)
        return data
    
    def _get_sample_rate(self, codec: str) -> int:
        """Get sample rate for codec."""
        rates = {
            "ulaw": 8000,
            "alaw": 8000,
            "slin": 8000,
            "slin16": 16000,
            "slin12": 12000,
        }
        return rates.get(codec, 8000)
    
    @staticmethod
    def _ulaw_to_pcm(data: bytes) -> bytes:
        """Convert μ-law to 16-bit PCM."""
        import audioop
        return audioop.ulaw2lin(data, 2)
    
    @staticmethod
    def _alaw_to_pcm(data: bytes) -> bytes:
        """Convert A-law to 16-bit PCM."""
        import audioop
        return audioop.alaw2lin(data, 2)
    
    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, CancelFrame):
            self._running = False
        await self.push_frame(frame, direction)


class AsteriskWSOutputTransport(FrameProcessor):
    """Output transport for sending audio to Asterisk via WebSocket.
    
    Implements production-ready features:
    - XOFF/XON flow control to prevent buffer overflow
    - Generation counter for clean interruptions
    - Real-time pacing to prevent XOFF/XON cycling
    """
    
    def __init__(
        self,
        params: AsteriskWSParams,
        name: Optional[str] = None,
        **kwargs
    ):
        super().__init__(name=name or "AsteriskWSOutput", **kwargs)
        self.params = params
        self._websocket: Optional[WebSocketServerProtocol] = None
        self._codec: str = "slin16"
        self._sample_rate: int = 16000
        self._resampler: Optional[AudioResampler] = None
        
        # Flow control
        self._flow_ready = asyncio.Event()
        self._flow_ready.set()
        
        # Generation counter for clean interruptions
        self._flush_generation = 0
        
        # Real-time pacing
        self._send_interval_sec = params.ptime_ms / 1000.0
        self._next_send_time = 0.0
        
        # Chunk size (samples per ptime)
        self._chunk_size = 0
        
    def set_websocket(self, ws: WebSocketServerProtocol):
        """Set the WebSocket connection."""
        self._websocket = ws
        
    def set_media_params(self, codec: str, sample_rate: int):
        """Set media parameters for output."""
        self._codec = codec
        self._sample_rate = sample_rate
        
        # Calculate chunk size in bytes (16-bit samples)
        samples_per_chunk = int(sample_rate * self.params.ptime_ms / 1000)
        self._chunk_size = samples_per_chunk * 2
        
        # Create resampler if needed
        if self.params.pipeline_sample_rate != sample_rate:
            self._resampler = AudioResampler(
                input_rate=self.params.pipeline_sample_rate,
                output_rate=sample_rate
            )
        else:
            self._resampler = None
            
        logger.info(f"AsteriskWSOutput: Ready - codec={codec}, chunk={self._chunk_size}b")
    
    def set_flow_control(self, paused: bool):
        """Handle XOFF/XON flow control from Asterisk."""
        if paused:
            logger.debug("AsteriskWSOutput: Flow control XOFF - pausing")
            self._flow_ready.clear()
        else:
            logger.debug("AsteriskWSOutput: Flow control XON - resuming")
            self._flow_ready.set()
    
    async def flush(self):
        """Flush audio buffer (called on interruption)."""
        self._flush_generation += 1
        if self._websocket:
            try:
                await self._websocket.send(json.dumps({"event": "FLUSH_MEDIA"}))
                logger.debug("AsteriskWSOutput: Sent FLUSH_MEDIA")
            except Exception as e:
                logger.error(f"AsteriskWSOutput: Error sending flush: {e}")
    
    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        
        if isinstance(frame, AudioRawFrame):
            await self._send_audio(frame)
        elif isinstance(frame, StartInterruptionFrame):
            await self.flush()
        elif isinstance(frame, (EndFrame, CancelFrame)):
            await self._send_end()
        else:
            await self.push_frame(frame, direction)
    
    async def _send_audio(self, frame: AudioRawFrame):
        """Send audio to Asterisk with flow control and pacing."""
        if not self._websocket:
            return
            
        my_generation = self._flush_generation
        
        # Resample if needed
        audio_data = frame.audio
        if self._resampler:
            audio_data = self._resampler.resample(audio_data)
        
        # Encode audio
        audio_encoded = self._encode_audio(audio_data)
        
        # Send in chunks matching ptime
        import base64
        
        for i in range(0, len(audio_encoded), self._chunk_size):
            # Check if we were flushed
            if my_generation != self._flush_generation:
                logger.debug("AsteriskWSOutput: Dropping stale audio after flush")
                return
            
            # Wait for flow control
            if not self._flow_ready.is_set():
                try:
                    await asyncio.wait_for(self._flow_ready.wait(), timeout=5.0)
                except asyncio.TimeoutError:
                    logger.warning("AsteriskWSOutput: Flow control timeout")
                    return
            
            # Real-time pacing
            now = time.monotonic()
            if self._next_send_time > now:
                await asyncio.sleep(self._next_send_time - now)
            
            # Prepare chunk
            chunk = audio_encoded[i:i + self._chunk_size]
            if len(chunk) < self._chunk_size:
                chunk = chunk + b"\x00" * (self._chunk_size - len(chunk))
            
            # Send
            try:
                message = json.dumps({
                    "event": "MEDIA",
                    "media": base64.b64encode(chunk).decode()
                })
                await self._websocket.send(message)
                self._next_send_time = time.monotonic() + self._send_interval_sec
            except Exception as e:
                logger.error(f"AsteriskWSOutput: Error sending audio: {e}")
                return
    
    def _encode_audio(self, data: bytes) -> bytes:
        """Encode PCM audio to Asterisk codec."""
        if self._codec in ("slin", "slin16"):
            return data  # Keep as PCM
        elif self._codec == "ulaw":
            import audioop
            return audioop.lin2ulaw(data, 2)
        elif self._codec == "alaw":
            import audioop
            return audioop.lin2alaw(data, 2)
        return data
    
    async def _send_end(self):
        """Send media end to Asterisk."""
        if self._websocket:
            try:
                await self._websocket.send(json.dumps({"event": "MEDIA_END"}))
            except Exception:
                pass


class AsteriskWSTransport(BaseTransport):
    """Main Asterisk WebSocket transport.
    
    Creates a WebSocket server that listens for connections from Asterisk's
    chan_websocket and manages the audio pipeline.
    
    Usage:
        transport = AsteriskWSTransport(AsteriskWSParams(port=8765))
        
        @transport.event_handler("on_call_start")
        async def on_call(transport, call_id):
            print(f"Call started: {call_id}")
        
        await transport.start_server()
    """
    
    def __init__(
        self,
        params: AsteriskWSParams,
        name: Optional[str] = None,
        **kwargs
    ):
        super().__init__(name=name or "AsteriskWSTransport", **kwargs)
        self.params = params
        self._input = AsteriskWSInputTransport(params, name=self._input_name)
        self._output = AsteriskWSOutputTransport(params, name=self._output_name)
        self._server = None
        self._event_handlers: Dict[str, List[Callable]] = {}
        
    def input(self) -> FrameProcessor:
        return self._input
    
    def output(self) -> FrameProcessor:
        return self._output
    
    def event_handler(self, event_name: str):
        """Decorator for registering event handlers."""
        def decorator(func: Callable):
            if event_name not in self._event_handlers:
                self._event_handlers[event_name] = []
            self._event_handlers[event_name].append(func)
            return func
        return decorator
    
    async def _trigger_event(self, event_name: str, *args, **kwargs):
        """Trigger registered event handlers."""
        for handler in self._event_handlers.get(event_name, []):
            try:
                result = handler(*args, **kwargs)
                if asyncio.iscoroutine(result):
                    await result
            except Exception as e:
                logger.error(f"Event handler error '{event_name}': {e}")
    
    async def start_server(self):
        """Start the WebSocket server."""
        if not HAS_WEBSOCKETS:
            raise ImportError("websockets package required: pip install websockets")
        
        async with websockets.serve(
            self._handle_connection,
            self.params.host,
            self.params.port
        ) as server:
            logger.info(f"AsteriskWSTransport: Server on ws://{self.params.host}:{self.params.port}")
            await asyncio.Future()  # Run forever
    
    async def _handle_connection(self, websocket: WebSocketServerProtocol):
        """Handle a WebSocket connection from Asterisk."""
        addr = websocket.remote_address
        logger.info(f"AsteriskWSTransport: Connection from {addr}")
        
        self._output.set_websocket(websocket)
        call_id = None
        
        try:
            async for message in websocket:
                try:
                    data = json.loads(message)
                    event_type = data.get("event")
                    
                    if event_type == AsteriskEventType.MEDIA_START:
                        codec = data.get("codec", "slin16")
                        sample_rate = self._input._get_sample_rate(codec)
                        call_id = data.get("channel", "unknown")
                        
                        self._input.set_media_params(codec, sample_rate, call_id)
                        self._output.set_media_params(codec, sample_rate)
                        
                        await self._trigger_event("on_call_start", self, call_id)
                        await self._input.push_frame(CallStartFrame(
                            call_id=call_id, codec=codec, sample_rate=sample_rate
                        ))
                        
                    elif event_type == AsteriskEventType.XOFF:
                        self._output.set_flow_control(paused=True)
                        
                    elif event_type == AsteriskEventType.XON:
                        self._output.set_flow_control(paused=False)
                        
                    else:
                        await self._input.handle_message(data)
                        
                except json.JSONDecodeError:
                    # Binary audio frame
                    import base64
                    await self._input.handle_message({
                        "event": "MEDIA",
                        "media": base64.b64encode(message).decode()
                    })
                    
        except websockets.ConnectionClosed:
            logger.info(f"AsteriskWSTransport: Connection closed from {addr}")
        except Exception as e:
            logger.error(f"AsteriskWSTransport: Error: {e}")
        finally:
            if call_id:
                await self._trigger_event("on_call_end", self, call_id)
            await self._input.push_frame(HangupFrame(call_id=call_id))
            await self._input.push_frame(EndFrame())
