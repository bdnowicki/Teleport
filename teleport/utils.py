"""
Utility functions for audio-over-RDP teleport system.
"""

import hashlib
import struct
from typing import List, Tuple, Optional
import numpy as np


def sha256_hash(data: bytes) -> bytes:
    """Calculate SHA-256 hash of data."""
    return hashlib.sha256(data).digest()


def crc32(data: bytes) -> int:
    """Calculate CRC32 checksum."""
    import zlib
    return zlib.crc32(data) & 0xffffffff


def bytes_to_bits(data: bytes) -> List[int]:
    """Convert bytes to list of bits (MSB first)."""
    bits = []
    for byte in data:
        for i in range(7, -1, -1):
            bits.append((byte >> i) & 1)
    return bits


def bits_to_bytes(bits: List[int]) -> bytes:
    """Convert list of bits to bytes (MSB first)."""
    if len(bits) % 8 != 0:
        bits.extend([0] * (8 - len(bits) % 8))
    
    result = []
    for i in range(0, len(bits), 8):
        byte = 0
        for j in range(8):
            if i + j < len(bits):
                byte |= (bits[i + j] << (7 - j))
        result.append(byte)
    return bytes(result)


def pack_uint32(value: int) -> bytes:
    """Pack 32-bit unsigned integer as big-endian bytes."""
    return struct.pack('>I', value)


def unpack_uint32(data: bytes) -> int:
    """Unpack 32-bit unsigned integer from big-endian bytes."""
    return struct.unpack('>I', data)[0]


def pack_uint16(value: int) -> bytes:
    """Pack 16-bit unsigned integer as big-endian bytes."""
    return struct.pack('>H', value)


def unpack_uint16(data: bytes) -> int:
    """Unpack 16-bit unsigned integer from big-endian bytes."""
    return struct.unpack('>H', data)[0]


def normalize_audio(audio: np.ndarray, target_level: float = 0.8) -> np.ndarray:
    """Normalize audio to target level, avoiding clipping."""
    max_val = np.max(np.abs(audio))
    if max_val > 0:
        scale = target_level / max_val
        return audio * scale
    return audio


def apply_raised_cosine_window(signal: np.ndarray, alpha: float = 0.35) -> np.ndarray:
    """Apply raised cosine window to reduce spectral leakage."""
    n = len(signal)
    window = np.ones(n)
    
    # Apply raised cosine taper at edges
    taper_len = int(n * 0.1)  # 10% taper
    if taper_len > 0:
        t = np.linspace(0, np.pi, taper_len)
        taper = 0.5 * (1 - np.cos(t))
        window[:taper_len] *= taper
        window[-taper_len:] *= taper[::-1]
    
    return signal * window


class Config:
    """Configuration parameters for the teleport system."""
    
    # Audio parameters
    SAMPLE_RATE = 48000
    CHANNELS = 1
    DTYPE = np.float32
    
    # Modulation parameters
    TONE_START = 1200  # Hz
    TONE_STEP = 120    # Hz
    SYMBOL_RATE = 1000  # symbols per second
    
    # Rate presets
    RATES = {
        'safe': {'tones': 8, 'symbol_rate': 800, 'bits_per_symbol': 3},
        'default': {'tones': 16, 'symbol_rate': 1000, 'bits_per_symbol': 4},
        'fast': {'tones': 32, 'symbol_rate': 1600, 'bits_per_symbol': 5}
    }
    
    # Frame parameters
    FRAME_SIZE = 1024  # payload bytes per frame
    PREAMBLE_LEN = 1000  # samples
    SYNC_LEN = 500     # samples
    
    # FEC parameters
    RS_N = 255
    RS_K = 223  # 32 parity symbols
    INTERLEAVER_DEPTH = 8
    
    # Chunking parameters
    CHUNK_SIZE = 64 * 1024  # 64KB chunks
    MAX_FILENAME_LEN = 255
