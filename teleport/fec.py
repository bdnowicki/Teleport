"""
Forward Error Correction (FEC) using Reed-Solomon codes and interleaving.
"""

import numpy as np
from typing import List, Tuple, Optional
from reedsolo import RSCodec
from .utils import Config


class ReedSolomonFEC:
    """Reed-Solomon Forward Error Correction."""
    
    def __init__(self, n: int = None, k: int = None):
        self.n = n or Config.RS_N
        self.k = k or Config.RS_K
        self.rs = RSCodec(self.n - self.k)  # Number of parity symbols
    
    def encode(self, data: bytes) -> bytes:
        """Encode data with Reed-Solomon FEC."""
        try:
            return self.rs.encode(data)
        except Exception as e:
            print(f"RS encoding error: {e}")
            return data
    
    def decode(self, data: bytes) -> Tuple[bytes, bool]:
        """Decode data with Reed-Solomon FEC.
        
        Returns:
            Tuple of (decoded_data, success)
        """
        try:
            decoded = self.rs.decode(data)
            return decoded, True
        except Exception as e:
            print(f"RS decoding error: {e}")
            return data, False


class Interleaver:
    """Interleaver for spreading burst errors."""
    
    def __init__(self, depth: int = None):
        self.depth = depth or Config.INTERLEAVER_DEPTH
    
    def interleave(self, data: bytes) -> bytes:
        """Interleave data to spread burst errors."""
        if len(data) == 0:
            return data
        
        # Pad data to multiple of depth
        padded_len = ((len(data) - 1) // self.depth + 1) * self.depth
        padded_data = data + b'\x00' * (padded_len - len(data))
        
        # Reshape and transpose
        matrix = np.frombuffer(padded_data, dtype=np.uint8).reshape(-1, self.depth)
        interleaved_matrix = matrix.T
        interleaved_data = interleaved_matrix.flatten().tobytes()
        
        return interleaved_data
    
    def deinterleave(self, data: bytes) -> bytes:
        """Deinterleave data to restore original order."""
        if len(data) == 0:
            return data
        
        # Reshape and transpose back
        matrix = np.frombuffer(data, dtype=np.uint8).reshape(self.depth, -1)
        deinterleaved_matrix = matrix.T
        deinterleaved_data = deinterleaved_matrix.flatten().tobytes()
        
        return deinterleaved_data


class FECEncoder:
    """Combined FEC encoder with Reed-Solomon and interleaving."""
    
    def __init__(self, rs_n: int = None, rs_k: int = None, interleaver_depth: int = None):
        self.rs_fec = ReedSolomonFEC(rs_n, rs_k)
        self.interleaver = Interleaver(interleaver_depth)
    
    def encode(self, data: bytes) -> bytes:
        """Encode data with FEC and interleaving."""
        # First interleave
        interleaved = self.interleaver.interleave(data)
        
        # Then apply Reed-Solomon
        fec_encoded = self.rs_fec.encode(interleaved)
        
        return fec_encoded
    
    def get_overhead(self) -> float:
        """Get FEC overhead ratio."""
        rs_overhead = (self.rs_fec.n - self.rs_fec.k) / self.rs_fec.k
        return rs_overhead


class FECDecoder:
    """Combined FEC decoder with Reed-Solomon and deinterleaving."""
    
    def __init__(self, rs_n: int = None, rs_k: int = None, interleaver_depth: int = None):
        self.rs_fec = ReedSolomonFEC(rs_n, rs_k)
        self.interleaver = Interleaver(interleaver_depth)
    
    def decode(self, data: bytes) -> Tuple[bytes, bool]:
        """Decode data with FEC and deinterleaving.
        
        Returns:
            Tuple of (decoded_data, success)
        """
        # First try Reed-Solomon decoding
        rs_decoded, rs_success = self.rs_fec.decode(data)
        
        if not rs_success:
            return data, False
        
        # Then deinterleave
        deinterleaved = self.interleaver.deinterleave(rs_decoded)
        
        return deinterleaved, True


class AdaptiveFEC:
    """Adaptive FEC that adjusts based on channel conditions."""
    
    def __init__(self):
        self.encoders = {
            'light': FECEncoder(255, 239, 4),    # Light FEC
            'medium': FECEncoder(255, 223, 8),  # Medium FEC
            'heavy': FECEncoder(255, 191, 16)   # Heavy FEC
        }
        self.decoders = {
            'light': FECDecoder(255, 239, 4),
            'medium': FECDecoder(255, 223, 8),
            'heavy': FECDecoder(255, 191, 16)
        }
        self.current_mode = 'medium'
        self.error_count = 0
        self.success_count = 0
    
    def encode(self, data: bytes) -> bytes:
        """Encode data with current FEC mode."""
        return self.encoders[self.current_mode].encode(data)
    
    def decode(self, data: bytes) -> Tuple[bytes, bool]:
        """Decode data with current FEC mode."""
        decoded, success = self.decoders[self.current_mode].decode(data)
        
        # Update statistics
        if success:
            self.success_count += 1
        else:
            self.error_count += 1
        
        # Adapt FEC mode based on error rate
        self._adapt_mode()
        
        return decoded, success
    
    def _adapt_mode(self):
        """Adapt FEC mode based on error statistics."""
        total = self.error_count + self.success_count
        if total < 10:  # Need more samples
            return
        
        error_rate = self.error_count / total
        
        if error_rate > 0.1:  # High error rate
            if self.current_mode != 'heavy':
                self.current_mode = 'heavy'
                print("Switching to heavy FEC mode")
        elif error_rate > 0.05:  # Medium error rate
            if self.current_mode != 'medium':
                self.current_mode = 'medium'
                print("Switching to medium FEC mode")
        else:  # Low error rate
            if self.current_mode != 'light':
                self.current_mode = 'light'
                print("Switching to light FEC mode")
        
        # Reset counters
        self.error_count = 0
        self.success_count = 0


def test_fec():
    """Test FEC functionality."""
    print("Testing FEC...")
    
    # Test data
    test_data = b"Hello, World! This is a test of FEC encoding and decoding."
    
    # Test Reed-Solomon
    rs_fec = ReedSolomonFEC(255, 223)
    encoded = rs_fec.encode(test_data)
    decoded, success = rs_fec.decode(encoded)
    
    print(f"RS FEC test: {'PASS' if success and decoded == test_data else 'FAIL'}")
    
    # Test interleaver
    interleaver = Interleaver(8)
    interleaved = interleaver.interleave(test_data)
    deinterleaved = interleaver.deinterleave(interleaved)
    
    print(f"Interleaver test: {'PASS' if deinterleaved == test_data else 'FAIL'}")
    
    # Test combined FEC
    fec_encoder = FECEncoder()
    fec_decoder = FECDecoder()
    
    encoded = fec_encoder.encode(test_data)
    decoded, success = fec_decoder.decode(encoded)
    
    print(f"Combined FEC test: {'PASS' if success and decoded == test_data else 'FAIL'}")
    
    # Test with errors
    if len(encoded) > 10:
        # Introduce some errors
        corrupted = bytearray(encoded)
        corrupted[5] ^= 0xFF
        corrupted[15] ^= 0xFF
        
        decoded, success = fec_decoder.decode(bytes(corrupted))
        print(f"Error correction test: {'PASS' if success and decoded == test_data else 'FAIL'}")


if __name__ == "__main__":
    test_fec()
