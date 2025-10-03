"""
Test script for teleport system.
"""

import os
import sys
import time
import tempfile
import threading
from pathlib import Path

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from teleport.audio_io import test_audio_devices
from teleport.fec import test_fec
from teleport.modem import MFSKModem
from teleport.framing import Frame, FrameAssembler, FileChunker
from teleport.utils import Config


def test_basic_functionality():
    """Test basic functionality without audio I/O."""
    print("=== Testing Basic Functionality ===")
    
    # Test FEC
    print("\n1. Testing FEC...")
    test_fec()
    
    # Test modem
    print("\n2. Testing MFSK Modem...")
    modem = MFSKModem(rate='default')
    test_data = b"Hello, World! This is a test."
    
    # Encode and decode
    audio = modem.encode_data(test_data)
    decoded = modem.decode_audio(audio)
    
    print(f"Modem test: {'PASS' if decoded == test_data else 'FAIL'}")
    print(f"Audio length: {len(audio)} samples ({len(audio)/48000:.2f} seconds)")
    
    # Test framing
    print("\n3. Testing Framing...")
    frame_assembler = FrameAssembler()
    frames = frame_assembler.create_frames(test_data)
    
    print(f"Created {len(frames)} frames")
    
    # Test frame serialization
    for i, frame in enumerate(frames):
        frame_bytes = frame.to_bytes()
        reconstructed = Frame.from_bytes(frame_bytes)
        
        if reconstructed and reconstructed.frame_id == frame.frame_id and reconstructed.payload == frame.payload:
            print(f"Frame {i}: PASS")
        else:
            print(f"Frame {i}: FAIL")
    
    # Test file chunking
    print("\n4. Testing File Chunking...")
    
    # Create a test file
    test_file = "test_file.txt"
    test_content = b"This is a test file for teleport system. " * 100  # ~4KB
    with open(test_file, 'wb') as f:
        f.write(test_content)
    
    try:
        chunker = FileChunker()
        file_header, chunks = chunker.chunk_file(test_file)
        
        print(f"File: {file_header.filename} ({file_header.file_size} bytes)")
        print(f"Chunks: {len(chunks)}")
        
        # Test chunk serialization
        for i, chunk in enumerate(chunks):
            chunk_bytes = chunk.to_bytes()
            reconstructed = Chunk.from_bytes(chunk_bytes)
            
            if reconstructed and reconstructed.chunk_id == chunk.chunk_id and reconstructed.data == chunk.data:
                print(f"Chunk {i}: PASS")
            else:
                print(f"Chunk {i}: FAIL")
        
        # Test file reassembly
        output_dir = "test_output"
        os.makedirs(output_dir, exist_ok=True)
        
        success = chunker.reassemble_file(file_header, chunks, output_dir)
        if success:
            # Verify file
            output_file = os.path.join(output_dir, file_header.filename)
            if os.path.exists(output_file):
                with open(output_file, 'rb') as f:
                    reassembled_content = f.read()
                
                if reassembled_content == test_content:
                    print("File reassembly: PASS")
                else:
                    print("File reassembly: FAIL - Content mismatch")
            else:
                print("File reassembly: FAIL - File not created")
        else:
            print("File reassembly: FAIL")
    
    finally:
        # Cleanup
        if os.path.exists(test_file):
            os.remove(test_file)
        if os.path.exists("test_output"):
            import shutil
            shutil.rmtree("test_output")


def test_audio_devices():
    """Test audio device functionality."""
    print("\n=== Testing Audio Devices ===")
    try:
        test_audio_devices()
    except Exception as e:
        print(f"Audio device test failed: {e}")


def test_end_to_end():
    """Test end-to-end functionality with audio."""
    print("\n=== Testing End-to-End ===")
    
    # Create test file
    test_file = "e2e_test.txt"
    test_content = b"End-to-end test file for teleport system. " * 50
    with open(test_file, 'wb') as f:
        f.write(test_content)
    
    try:
        # Test transmit
        print("Testing transmit...")
        from teleport.cli import transmit_file
        
        # Run in separate thread to avoid blocking
        tx_success = [False]
        def tx_thread():
            try:
                tx_success[0] = transmit_file(test_file, rate='safe', repeat=1)
            except Exception as e:
                print(f"Transmit error: {e}")
                tx_success[0] = False
        
        tx_thread_obj = threading.Thread(target=tx_thread)
        tx_thread_obj.start()
        
        # Wait a bit for transmission
        time.sleep(2)
        
        # Stop transmission
        tx_thread_obj.join(timeout=1)
        
        print(f"Transmit test: {'PASS' if tx_success[0] else 'FAIL'}")
        
    except Exception as e:
        print(f"End-to-end test error: {e}")
    
    finally:
        # Cleanup
        if os.path.exists(test_file):
            os.remove(test_file)


def main():
    """Run all tests."""
    print("Teleport System Test Suite")
    print("=" * 50)
    
    # Test basic functionality
    test_basic_functionality()
    
    # Test audio devices
    test_audio_devices()
    
    # Test end-to-end (optional, requires audio)
    print("\n=== End-to-End Test (Optional) ===")
    print("This test requires audio devices and may produce sound.")
    response = input("Run end-to-end test? (y/N): ").strip().lower()
    
    if response == 'y':
        test_end_to_end()
    else:
        print("Skipping end-to-end test")
    
    print("\nTest suite completed!")


if __name__ == '__main__':
    main()
