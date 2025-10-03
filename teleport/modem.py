"""
MFSK modulator and demodulator for audio-over-RDP transmission.
"""

import numpy as np
from typing import List, Tuple, Optional
from scipy import signal
from .utils import Config


class MFSKModulator:
    """MFSK modulator for transmitting digital data as audio tones."""
    
    def __init__(self, tones: int, symbol_rate: int, tone_start: int, tone_step: int, 
                 sample_rate: int = 48000):
        self.tones = tones
        self.symbol_rate = symbol_rate
        self.tone_start = tone_start
        self.tone_step = tone_step
        self.sample_rate = sample_rate
        self.symbol_duration = sample_rate // symbol_rate
        
        # Generate tone frequencies
        self.frequencies = [tone_start + i * tone_step for i in range(tones)]
        
        # Generate Barker code for sync (13-bit)
        self.barker_code = [1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1]
    
    def generate_preamble(self) -> np.ndarray:
        """Generate preamble: pilot tone + alternating tones."""
        # 1 second pilot tone at first frequency
        pilot_duration = int(self.sample_rate * 1.0)
        pilot = self._generate_tone(self.frequencies[0], pilot_duration)
        
        # 0.5 seconds alternating tones
        alt_duration = int(self.sample_rate * 0.5)
        alt_samples = []
        for i in range(0, alt_duration, self.symbol_duration):
            tone_idx = (i // self.symbol_duration) % 2
            tone_duration = min(self.symbol_duration, alt_duration - i)
            alt_samples.append(self._generate_tone(self.frequencies[tone_idx], tone_duration))
        
        alternating = np.concatenate(alt_samples) if alt_samples else np.array([])
        
        return np.concatenate([pilot, alternating])
    
    def generate_sync_word(self) -> np.ndarray:
        """Generate sync word using Barker code."""
        sync_samples = []
        for bit in self.barker_code:
            # Use first two frequencies for sync
            freq = self.frequencies[0] if bit == 1 else self.frequencies[1]
            sync_samples.append(self._generate_tone(freq, self.symbol_duration))
        
        return np.concatenate(sync_samples)
    
    def modulate_symbols(self, symbols: List[int]) -> np.ndarray:
        """Modulate a list of symbols into audio."""
        audio_samples = []
        
        for symbol in symbols:
            if 0 <= symbol < self.tones:
                tone_audio = self._generate_tone(self.frequencies[symbol], self.symbol_duration)
                audio_samples.append(tone_audio)
        
        return np.concatenate(audio_samples) if audio_samples else np.array([])
    
    def _generate_tone(self, frequency: float, duration: int) -> np.ndarray:
        """Generate a tone with raised cosine windowing."""
        t = np.linspace(0, duration / self.sample_rate, duration, False)
        tone = np.sin(2 * np.pi * frequency * t)
        
        # Apply raised cosine window
        window = np.ones(duration)
        taper_len = int(duration * 0.1)  # 10% taper
        if taper_len > 0:
            taper = np.linspace(0, np.pi, taper_len)
            window[:taper_len] = 0.5 * (1 - np.cos(taper))
            window[-taper_len:] = 0.5 * (1 - np.cos(taper[::-1]))
        
        return tone * window


class MFSKDemodulator:
    """MFSK demodulator for receiving digital data from audio tones."""
    
    def __init__(self, tones: int, symbol_rate: int, tone_start: int, tone_step: int,
                 sample_rate: int = 48000):
        self.tones = tones
        self.symbol_rate = symbol_rate
        self.tone_start = tone_start
        self.tone_step = tone_step
        self.sample_rate = sample_rate
        self.symbol_duration = sample_rate // symbol_rate
        
        # Generate tone frequencies
        self.frequencies = [tone_start + i * tone_step for i in range(tones)]
        
        # Barker code for sync detection
        self.barker_code = [1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1]
        
        # Demodulation state
        self.sync_found = False
        self.symbol_buffer = []
        self.last_sync_pos = 0
    
    def find_preamble(self, audio: np.ndarray) -> Optional[int]:
        """Find preamble in audio and return position."""
        # Look for pilot tone (1 second at first frequency)
        pilot_freq = self.frequencies[0]
        pilot_duration = self.sample_rate  # 1 second
        
        # Use FFT to detect pilot tone
        window_size = min(len(audio), pilot_duration)
        if window_size < pilot_duration // 2:
            return None
        
        # Check for pilot tone in first part of audio
        pilot_audio = audio[:window_size]
        freqs, psd = signal.periodogram(pilot_audio, self.sample_rate)
        
        # Find peak near pilot frequency
        freq_idx = np.argmin(np.abs(freqs - pilot_freq))
        pilot_power = psd[freq_idx]
        
        # Check if pilot tone is strong enough
        avg_power = np.mean(psd)
        if pilot_power > 10 * avg_power:
            return 0  # Preamble starts at beginning
        
        return None
    
    def find_sync_word(self, audio: np.ndarray, start_pos: int = 0) -> Optional[int]:
        """Find sync word in audio and return position."""
        # Look for Barker code pattern
        sync_len = len(self.barker_code) * self.symbol_duration
        
        if start_pos + sync_len > len(audio):
            return None
        
        # Check each possible position
        for pos in range(start_pos, len(audio) - sync_len, self.symbol_duration // 4):
            sync_audio = audio[pos:pos + sync_len]
            if self._detect_barker_code(sync_audio):
                return pos
        
        return None
    
    def demodulate_symbols(self, audio: np.ndarray, start_pos: int, num_symbols: int) -> List[int]:
        """Demodulate symbols from audio starting at given position."""
        symbols = []
        
        for i in range(num_symbols):
            symbol_start = start_pos + i * self.symbol_duration
            symbol_end = symbol_start + self.symbol_duration
            
            if symbol_end > len(audio):
                break
            
            symbol_audio = audio[symbol_start:symbol_end]
            symbol = self._demodulate_symbol(symbol_audio)
            symbols.append(symbol)
        
        return symbols
    
    def _detect_barker_code(self, audio: np.ndarray) -> bool:
        """Detect Barker code in audio segment."""
        if len(audio) < len(self.barker_code) * self.symbol_duration:
            return False
        
        detected_bits = []
        for i in range(len(self.barker_code)):
            symbol_start = i * self.symbol_duration
            symbol_end = symbol_start + self.symbol_duration
            symbol_audio = audio[symbol_start:symbol_end]
            
            # Demodulate this symbol
            symbol = self._demodulate_symbol(symbol_audio)
            
            # Convert to bit (0 or 1)
            bit = 1 if symbol == 0 else 0  # First two frequencies for sync
            detected_bits.append(bit)
        
        # Check if it matches Barker code
        return detected_bits == self.barker_code
    
    def _demodulate_symbol(self, audio: np.ndarray) -> int:
        """Demodulate a single symbol from audio."""
        if len(audio) == 0:
            return 0
        
        # Use FFT to find strongest frequency
        freqs, psd = signal.periodogram(audio, self.sample_rate)
        
        # Find peak near our tone frequencies
        best_symbol = 0
        best_power = 0
        
        for i, freq in enumerate(self.frequencies):
            # Find closest frequency bin
            freq_idx = np.argmin(np.abs(freqs - freq))
            power = psd[freq_idx]
            
            if power > best_power:
                best_power = power
                best_symbol = i
        
        return best_symbol
    
    def reset_sync(self):
        """Reset sync state."""
        self.sync_found = False
        self.symbol_buffer = []
        self.last_sync_pos = 0


class MFSKModem:
    """Combined MFSK modem for transmission and reception."""
    
    def __init__(self, rate: str = 'default', tone_start: int = 1200, tone_step: int = 120,
                 sample_rate: int = 48000):
        self.rate = rate
        self.sample_rate = sample_rate
        
        # Get rate parameters
        if rate not in Config.RATES:
            raise ValueError(f"Invalid rate: {rate}. Must be one of {list(Config.RATES.keys())}")
        
        rate_params = Config.RATES[rate]
        self.tones = rate_params['tones']
        self.symbol_rate = rate_params['symbol_rate']
        self.bits_per_symbol = rate_params['bits_per_symbol']
        
        # Create modulator and demodulator
        self.modulator = MFSKModulator(
            tones=self.tones,
            symbol_rate=self.symbol_rate,
            tone_start=tone_start,
            tone_step=tone_step,
            sample_rate=sample_rate
        )
        
        self.demodulator = MFSKDemodulator(
            tones=self.tones,
            symbol_rate=self.symbol_rate,
            tone_start=tone_start,
            tone_step=tone_step,
            sample_rate=sample_rate
        )
    
    def get_throughput(self) -> float:
        """Get theoretical throughput in bits per second."""
        return self.symbol_rate * self.bits_per_symbol
    
    def encode_data(self, data: bytes) -> np.ndarray:
        """Encode data into audio signal."""
        # Convert bytes to bits
        from .utils import bytes_to_bits
        bits = bytes_to_bits(data)
        
        # Pad to multiple of bits_per_symbol
        while len(bits) % self.bits_per_symbol != 0:
            bits.append(0)
        
        # Convert bits to symbols
        symbols = []
        for i in range(0, len(bits), self.bits_per_symbol):
            symbol_bits = bits[i:i + self.bits_per_symbol]
            symbol = 0
            for j, bit in enumerate(symbol_bits):
                symbol |= (bit << (self.bits_per_symbol - 1 - j))
            symbols.append(symbol)
        
        # Generate preamble
        preamble = self.modulator.generate_preamble()
        
        # Generate sync word
        sync_word = self.modulator.generate_sync_word()
        
        # Modulate data symbols
        data_audio = self.modulator.modulate_symbols(symbols)
        
        # Combine all parts
        return np.concatenate([preamble, sync_word, data_audio])
    
    def decode_audio(self, audio: np.ndarray) -> Optional[bytes]:
        """Decode audio signal back to data."""
        # Find preamble
        preamble_pos = self.demodulator.find_preamble(audio)
        if preamble_pos is None:
            return None
        
        # Find sync word
        sync_pos = self.demodulator.find_sync_word(audio, preamble_pos + self.sample_rate)
        if sync_pos is None:
            return None
        
        # Skip sync word
        data_start = sync_pos + len(self.demodulator.barker_code) * self.demodulator.symbol_duration
        
        # Estimate number of symbols (this is a simplified approach)
        # In practice, you'd need proper framing to know the exact length
        remaining_audio = audio[data_start:]
        num_symbols = len(remaining_audio) // self.demodulator.symbol_duration
        
        # Demodulate symbols
        symbols = self.demodulator.demodulate_symbols(remaining_audio, 0, num_symbols)
        
        # Convert symbols back to bits
        bits = []
        for symbol in symbols:
            for i in range(self.bits_per_symbol):
                bit = (symbol >> (self.bits_per_symbol - 1 - i)) & 1
                bits.append(bit)
        
        # Convert bits to bytes
        from .utils import bits_to_bytes
        return bits_to_bytes(bits)
