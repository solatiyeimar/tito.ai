"""Asterisk AudioSocket Protocol Implementation.

The AudioSocket protocol is a simple TCP-based protocol for streaming audio 
between Asterisk PBX and external applications.

Message Format:
- 1 byte: Message type
- 2 bytes: Payload length (big-endian)
- N bytes: Payload data

Message Types:
- 0x01 UUID: 16-byte binary UUID identifying the call session
- 0x10 AUDIO: 320 bytes of 16-bit signed PCM audio (20ms at 8kHz mono)
- 0x03 DTMF: 1-byte ASCII digit (0-9, *, #, A-D)
- 0x00 HANGUP: No payload, indicates call termination
- 0xFF ERROR: Optional error code payload
"""

import struct
from dataclasses import dataclass
from enum import IntEnum
from typing import Optional, Tuple


class MessageType(IntEnum):
    """AudioSocket message types."""
    HANGUP = 0x00
    UUID = 0x01
    DTMF = 0x03
    AUDIO = 0x10
    ERROR = 0xFF


@dataclass
class AudioSocketMessage:
    """Parsed AudioSocket message."""
    msg_type: MessageType
    payload: bytes
    
    @property
    def uuid(self) -> Optional[str]:
        """Get UUID string if this is a UUID message."""
        if self.msg_type == MessageType.UUID and len(self.payload) == 16:
            import uuid
            return str(uuid.UUID(bytes=self.payload))
        return None
    
    @property
    def dtmf_digit(self) -> Optional[str]:
        """Get DTMF digit if this is a DTMF message."""
        if self.msg_type == MessageType.DTMF and len(self.payload) >= 1:
            return chr(self.payload[0])
        return None
    
    @property
    def audio_data(self) -> Optional[bytes]:
        """Get audio data if this is an AUDIO message."""
        if self.msg_type == MessageType.AUDIO:
            return self.payload
        return None


class AudioSocketProtocol:
    """AudioSocket protocol parser and builder."""
    
    HEADER_SIZE = 3  # 1 byte type + 2 bytes length
    AUDIO_CHUNK_SIZE = 320  # 20ms at 8kHz, 16-bit mono
    
    @staticmethod
    def parse_header(data: bytes) -> Tuple[MessageType, int]:
        """Parse message header.
        
        Args:
            data: At least 3 bytes of header data.
            
        Returns:
            Tuple of (message_type, payload_length)
            
        Raises:
            ValueError: If data is too short or message type is invalid.
        """
        if len(data) < AudioSocketProtocol.HEADER_SIZE:
            raise ValueError(f"Header too short: {len(data)} bytes")
        
        msg_type_byte = data[0]
        payload_length = struct.unpack(">H", data[1:3])[0]
        
        try:
            msg_type = MessageType(msg_type_byte)
        except ValueError:
            raise ValueError(f"Unknown message type: {msg_type_byte:#x}")
        
        return msg_type, payload_length
    
    @staticmethod
    def parse_message(data: bytes) -> Tuple[AudioSocketMessage, int]:
        """Parse a complete message from buffer.
        
        Args:
            data: Buffer containing at least one complete message.
            
        Returns:
            Tuple of (parsed_message, bytes_consumed)
            
        Raises:
            ValueError: If buffer doesn't contain a complete message.
        """
        if len(data) < AudioSocketProtocol.HEADER_SIZE:
            raise ValueError("Incomplete header")
        
        msg_type, payload_length = AudioSocketProtocol.parse_header(data)
        total_length = AudioSocketProtocol.HEADER_SIZE + payload_length
        
        if len(data) < total_length:
            raise ValueError(f"Incomplete message: need {total_length}, have {len(data)}")
        
        payload = data[AudioSocketProtocol.HEADER_SIZE:total_length]
        message = AudioSocketMessage(msg_type=msg_type, payload=payload)
        
        return message, total_length
    
    @staticmethod
    def create_audio_message(audio_data: bytes) -> bytes:
        """Create an AUDIO message.
        
        Args:
            audio_data: PCM audio data (16-bit signed, 8kHz, mono)
            
        Returns:
            Complete message bytes ready to send.
        """
        header = struct.pack(">BH", MessageType.AUDIO, len(audio_data))
        return header + audio_data
    
    @staticmethod
    def create_hangup_message() -> bytes:
        """Create a HANGUP message.
        
        Returns:
            Complete message bytes ready to send.
        """
        return struct.pack(">BH", MessageType.HANGUP, 0)
    
    @staticmethod
    def create_error_message(error_code: int = 0) -> bytes:
        """Create an ERROR message.
        
        Args:
            error_code: Optional application-specific error code.
            
        Returns:
            Complete message bytes ready to send.
        """
        payload = struct.pack(">B", error_code) if error_code else b""
        header = struct.pack(">BH", MessageType.ERROR, len(payload))
        return header + payload
