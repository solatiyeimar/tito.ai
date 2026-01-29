"""Audio Resampler for Asterisk Integration.

Asterisk AudioSocket uses 8kHz audio, while most AI services (STT/TTS) 
use 16kHz. This module provides efficient resampling between these rates.
"""

import struct
from typing import Optional

try:
    import numpy as np
    from scipy import signal
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


class AudioResampler:
    """Audio resampler for converting between sample rates.
    
    Supports both scipy-based high-quality resampling and a simple
    linear interpolation fallback.
    """
    
    def __init__(
        self,
        input_rate: int = 8000,
        output_rate: int = 16000,
        channels: int = 1,
        dtype: str = "int16"
    ):
        """Initialize the resampler.
        
        Args:
            input_rate: Input sample rate in Hz.
            output_rate: Output sample rate in Hz.
            channels: Number of audio channels.
            dtype: Audio data type (int16 for PCM).
        """
        self.input_rate = input_rate
        self.output_rate = output_rate
        self.channels = channels
        self.dtype = dtype
        self.ratio = output_rate / input_rate
        
    def resample(self, data: bytes) -> bytes:
        """Resample audio data.
        
        Args:
            data: Input audio as bytes (16-bit signed PCM).
            
        Returns:
            Resampled audio as bytes.
        """
        if self.input_rate == self.output_rate:
            return data
        
        if HAS_SCIPY:
            return self._resample_scipy(data)
        else:
            return self._resample_linear(data)
    
    def _resample_scipy(self, data: bytes) -> bytes:
        """High-quality resampling using scipy."""
        # Convert bytes to numpy array
        samples = np.frombuffer(data, dtype=np.int16)
        
        # Calculate output length
        output_length = int(len(samples) * self.ratio)
        
        # Resample using scipy
        resampled = signal.resample(samples, output_length)
        
        # Clip to int16 range and convert back to bytes
        resampled = np.clip(resampled, -32768, 32767).astype(np.int16)
        return resampled.tobytes()
    
    def _resample_linear(self, data: bytes) -> bytes:
        """Simple linear interpolation resampling (fallback)."""
        # Unpack input samples
        sample_count = len(data) // 2
        samples = struct.unpack(f"<{sample_count}h", data)
        
        # Calculate output length
        output_length = int(sample_count * self.ratio)
        output_samples = []
        
        for i in range(output_length):
            # Calculate position in input
            pos = i / self.ratio
            idx = int(pos)
            frac = pos - idx
            
            # Linear interpolation
            if idx + 1 < sample_count:
                sample = int(samples[idx] * (1 - frac) + samples[idx + 1] * frac)
            else:
                sample = samples[idx] if idx < sample_count else 0
            
            # Clip to int16 range
            sample = max(-32768, min(32767, sample))
            output_samples.append(sample)
        
        return struct.pack(f"<{len(output_samples)}h", *output_samples)


class UpsampleResampler(AudioResampler):
    """Resampler for upsampling from 8kHz to 16kHz (Asterisk → AI)."""
    
    def __init__(self):
        super().__init__(input_rate=8000, output_rate=16000)


class DownsampleResampler(AudioResampler):
    """Resampler for downsampling from 16kHz to 8kHz (AI → Asterisk)."""
    
    def __init__(self):
        super().__init__(input_rate=16000, output_rate=8000)


def create_silence(duration_ms: int, sample_rate: int = 8000) -> bytes:
    """Create silence audio of specified duration.
    
    Args:
        duration_ms: Duration in milliseconds.
        sample_rate: Sample rate in Hz.
        
    Returns:
        Silent audio as bytes (16-bit signed PCM).
    """
    sample_count = int(sample_rate * duration_ms / 1000)
    return b"\x00\x00" * sample_count
