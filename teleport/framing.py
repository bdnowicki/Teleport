"""
Frame structure and framing for audio-over-RDP transmission.
"""

import struct
import numpy as np
from typing import List, Tuple, Optional
from .utils import Config, crc32, pack_uint32, unpack_uint32, pack_uint16, unpack_uint16


class Frame:
    """A single data frame with header and payload."""
    
    def __init__(self, frame_id: int, payload: bytes, is_last: bool = False):
        self.frame_id = frame_id
        self.payload = payload
        self.is_last = is_last
        self.crc = crc32(payload)
    
    def to_bytes(self) -> bytes:
        """Serialize frame to bytes."""
        # Frame format: [frame_id(4)] [is_last(1)] [payload_len(2)] [payload] [crc(4)]
        header = struct.pack('>IBH', self.frame_id, 1 if self.is_last else 0, len(self.payload))
        return header + self.payload + struct.pack('>I', self.crc)
    
    @classmethod
    def from_bytes(cls, data: bytes) -> Optional['Frame']:
        """Deserialize frame from bytes."""
        if len(data) < 7:  # Minimum header size
            return None
        
        try:
            frame_id, is_last, payload_len = struct.unpack('>IBH', data[:7])
            is_last = bool(is_last)
            
            if len(data) < 7 + payload_len + 4:  # Header + payload + CRC
                return None
            
            payload = data[7:7 + payload_len]
            crc = struct.unpack('>I', data[7 + payload_len:7 + payload_len + 4])[0]
            
            # Verify CRC
            if crc32(payload) != crc:
                return None
            
            return cls(frame_id, payload, is_last)
        except struct.error:
            return None
    
    def __len__(self):
        return 7 + len(self.payload) + 4  # Header + payload + CRC


class FrameAssembler:
    """Assembles frames from data."""
    
    def __init__(self, frame_size: int = None):
        self.frame_size = frame_size or Config.FRAME_SIZE
    
    def create_frames(self, data: bytes) -> List[Frame]:
        """Split data into frames."""
        frames = []
        frame_id = 0
        
        for i in range(0, len(data), self.frame_size):
            chunk = data[i:i + self.frame_size]
            is_last = (i + len(chunk) == len(data))
            frames.append(Frame(frame_id, chunk, is_last))
            frame_id += 1
        
        return frames
    
    def reassemble_data(self, frames: List[Frame]) -> Optional[bytes]:
        """Reassemble data from frames."""
        if not frames:
            return None
        
        # Sort frames by frame_id
        frames.sort(key=lambda f: f.frame_id)
        
        # Check for missing frames
        expected_ids = set(range(len(frames)))
        actual_ids = set(f.frame_id for f in frames)
        missing_ids = expected_ids - actual_ids
        
        if missing_ids:
            print(f"Warning: Missing frames: {sorted(missing_ids)}")
        
        # Reassemble data
        data_parts = []
        for frame in frames:
            data_parts.append(frame.payload)
        
        return b''.join(data_parts)


class FileHeader:
    """File header with metadata."""
    
    def __init__(self, filename: str, file_size: int, file_hash: bytes):
        self.filename = filename
        self.file_size = file_size
        self.file_hash = file_hash
    
    def to_bytes(self) -> bytes:
        """Serialize file header to bytes."""
        filename_bytes = self.filename.encode('utf-8')
        if len(filename_bytes) > Config.MAX_FILENAME_LEN:
            raise ValueError(f"Filename too long: {len(filename_bytes)} > {Config.MAX_FILENAME_LEN}")
        
        # Header format: [filename_len(1)] [filename] [file_size(8)] [file_hash(32)]
        header = struct.pack('B', len(filename_bytes))
        header += filename_bytes
        header += struct.pack('>Q', self.file_size)
        header += self.file_hash
        
        return header
    
    @classmethod
    def from_bytes(cls, data: bytes) -> Optional['FileHeader']:
        """Deserialize file header from bytes."""
        if len(data) < 1:
            return None
        
        try:
            filename_len = data[0]
            if len(data) < 1 + filename_len + 8 + 32:
                return None
            
            filename = data[1:1 + filename_len].decode('utf-8')
            file_size = struct.unpack('>Q', data[1 + filename_len:1 + filename_len + 8])[0]
            file_hash = data[1 + filename_len + 8:1 + filename_len + 8 + 32]
            
            return cls(filename, file_size, file_hash)
        except (struct.error, UnicodeDecodeError):
            return None


class ChunkHeader:
    """Chunk header for file chunking."""
    
    def __init__(self, chunk_id: int, chunk_size: int, is_last: bool = False):
        self.chunk_id = chunk_id
        self.chunk_size = chunk_size
        self.is_last = is_last
    
    def to_bytes(self) -> bytes:
        """Serialize chunk header to bytes."""
        # Chunk header format: [chunk_id(4)] [chunk_size(4)] [is_last(1)]
        return struct.pack('>IIB', self.chunk_id, self.chunk_size, 1 if self.is_last else 0)
    
    @classmethod
    def from_bytes(cls, data: bytes) -> Optional['ChunkHeader']:
        """Deserialize chunk header from bytes."""
        if len(data) < 9:
            return None
        
        try:
            chunk_id, chunk_size, is_last = struct.unpack('>IIB', data[:9])
            return cls(chunk_id, chunk_size, bool(is_last))
        except struct.error:
            return None


class Chunk:
    """A file chunk with header and data."""
    
    def __init__(self, chunk_id: int, data: bytes, is_last: bool = False):
        self.chunk_id = chunk_id
        self.data = data
        self.is_last = is_last
        self.crc = crc32(data)
    
    def to_bytes(self) -> bytes:
        """Serialize chunk to bytes."""
        header = ChunkHeader(self.chunk_id, len(self.data), self.is_last)
        return header.to_bytes() + self.data + struct.pack('>I', self.crc)
    
    @classmethod
    def from_bytes(cls, data: bytes) -> Optional['Chunk']:
        """Deserialize chunk from bytes."""
        if len(data) < 9:
            return None
        
        header = ChunkHeader.from_bytes(data[:9])
        if header is None:
            return None
        
        if len(data) < 9 + header.chunk_size + 4:
            return None
        
        chunk_data = data[9:9 + header.chunk_size]
        crc = struct.unpack('>I', data[9 + header.chunk_size:9 + header.chunk_size + 4])[0]
        
        # Verify CRC
        if crc32(chunk_data) != crc:
            return None
        
        return cls(header.chunk_id, chunk_data, header.is_last)


class FileChunker:
    """Handles file chunking and reassembly."""
    
    def __init__(self, chunk_size: int = None):
        self.chunk_size = chunk_size or Config.CHUNK_SIZE
    
    def chunk_file(self, filepath: str) -> Tuple[FileHeader, List[Chunk]]:
        """Chunk a file into smaller pieces."""
        import os
        from .utils import sha256_hash
        
        # Read file
        with open(filepath, 'rb') as f:
            file_data = f.read()
        
        # Create file header
        filename = os.path.basename(filepath)
        file_hash = sha256_hash(file_data)
        file_header = FileHeader(filename, len(file_data), file_hash)
        
        # Create chunks
        chunks = []
        chunk_id = 0
        
        for i in range(0, len(file_data), self.chunk_size):
            chunk_data = file_data[i:i + self.chunk_size]
            is_last = (i + len(chunk_data) == len(file_data))
            chunks.append(Chunk(chunk_id, chunk_data, is_last))
            chunk_id += 1
        
        return file_header, chunks
    
    def reassemble_file(self, file_header: FileHeader, chunks: List[Chunk], output_dir: str) -> bool:
        """Reassemble file from chunks."""
        import os
        
        # Sort chunks by ID
        chunks.sort(key=lambda c: c.chunk_id)
        
        # Check for missing chunks
        expected_ids = set(range(len(chunks)))
        actual_ids = set(c.chunk_id for c in chunks)
        missing_ids = expected_ids - actual_ids
        
        if missing_ids:
            print(f"Warning: Missing chunks: {sorted(missing_ids)}")
            return False
        
        # Reassemble data
        file_data = b''.join(chunk.data for chunk in chunks)
        
        # Verify file size
        if len(file_data) != file_header.file_size:
            print(f"Error: File size mismatch: expected {file_header.file_size}, got {len(file_data)}")
            return False
        
        # Verify file hash
        from .utils import sha256_hash
        if sha256_hash(file_data) != file_header.file_hash:
            print("Error: File hash mismatch")
            return False
        
        # Write file
        output_path = os.path.join(output_dir, file_header.filename)
        with open(output_path, 'wb') as f:
            f.write(file_data)
        
        print(f"File reassembled: {output_path}")
        return True


class FrameTransmitter:
    """Transmits frames using MFSK modulation."""
    
    def __init__(self, modem, frame_assembler: FrameAssembler):
        self.modem = modem
        self.frame_assembler = frame_assembler
    
    def transmit_frames(self, frames: List[Frame]) -> np.ndarray:
        """Transmit frames as audio."""
        audio_parts = []
        
        for frame in frames:
            # Convert frame to bytes
            frame_bytes = frame.to_bytes()
            
            # Modulate frame
            frame_audio = self.modem.encode_data(frame_bytes)
            audio_parts.append(frame_audio)
        
        return np.concatenate(audio_parts) if audio_parts else np.array([])


class FrameReceiver:
    """Receives and decodes frames from audio."""
    
    def __init__(self, modem, frame_assembler: FrameAssembler):
        self.modem = modem
        self.frame_assembler = frame_assembler
        self.received_frames = []
        self.sync_found = False
    
    def process_audio(self, audio: np.ndarray) -> List[Frame]:
        """Process audio and extract frames."""
        frames = []
        
        if not self.sync_found:
            # Look for sync
            sync_pos = self.modem.demodulator.find_sync_word(audio)
            if sync_pos is not None:
                self.sync_found = True
                audio = audio[sync_pos:]
        
        if self.sync_found:
            # Try to decode frames
            try:
                frame_data = self.modem.decode_audio(audio)
                if frame_data:
                    frame = Frame.from_bytes(frame_data)
                    if frame:
                        frames.append(frame)
            except Exception as e:
                print(f"Frame decode error: {e}")
        
        return frames
