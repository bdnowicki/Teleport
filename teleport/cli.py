"""
Command-line interface for audio-over-RDP file teleport.
"""

import argparse
import sys
import os
import time
import threading
from pathlib import Path
from typing import Optional

import numpy as np

from .audio_io import AudioTransmitter, AudioReceiver, list_audio_devices, find_loopback_device, test_device_capabilities
from .modem import MFSKModem
from .framing import FrameAssembler, FrameTransmitter, FrameReceiver, FileChunker, FileHeader, Chunk
from .fec import FECEncoder, FECDecoder
from .utils import Config, normalize_audio


def list_devices():
    """List available audio devices."""
    print("Available audio devices:")
    devices = list_audio_devices()
    
    for i, device in enumerate(devices):
        print(f"{i}: {device}")
    
    print(f"\nRecommended loopback device: {find_loopback_device()}")
    
    # Test devices for loopback capability
    print("\nTesting devices for loopback capability:")
    working_devices = []
    for i, device in enumerate(devices):
        if device.is_output:  # Only test output devices for loopback
            if test_device_capabilities(i):
                working_devices.append(i)
    
    if working_devices:
        print(f"\nWorking loopback devices: {working_devices}")
    else:
        print("\nNo devices found that support WASAPI loopback")
        print("Try using a different device or check your audio drivers")


def transmit_file(filepath: str, rate: str = 'default', repeat: int = 1, 
                 tone_start: int = 1200, tone_step: int = 120, device: Optional[int] = None):
    """Transmit a file as audio."""
    if not os.path.exists(filepath):
        print(f"Error: File not found: {filepath}")
        return False
    
    print(f"Transmitting file: {filepath}")
    print(f"Rate: {rate}, Repeat: {repeat}")
    
    # Create modem
    modem = MFSKModem(rate=rate, tone_start=tone_start, tone_step=tone_step)
    print(f"Theoretical throughput: {modem.get_throughput():.1f} bps")
    
    # Create FEC encoder
    fec_encoder = FECEncoder()
    print(f"FEC overhead: {fec_encoder.get_overhead():.1%}")
    
    # Chunk file
    chunker = FileChunker()
    file_header, chunks = chunker.chunk_file(filepath)
    print(f"File: {file_header.filename} ({file_header.file_size} bytes)")
    print(f"Chunks: {len(chunks)}")
    
    # Create frame assembler
    frame_assembler = FrameAssembler()
    
    # Create frame transmitter
    frame_tx = FrameTransmitter(modem, frame_assembler)
    
    # Create audio transmitter
    audio_tx = AudioTransmitter()
    
    try:
        # Start audio transmission
        audio_tx.start(device)
        print("Audio transmission started")
        
        # Transmit file header
        print("Transmitting file header...")
        header_frames = frame_assembler.create_frames(file_header.to_bytes())
        for frame in header_frames:
            frame_bytes = frame.to_bytes()
            fec_encoded = fec_encoder.encode(frame_bytes)
            frame_audio = modem.encode_data(fec_encoded)
            audio_tx.play(frame_audio)
            time.sleep(0.1)  # Small delay between frames
        
        # Transmit chunks
        for chunk in chunks:
            print(f"Transmitting chunk {chunk.chunk_id}...")
            
            for _ in range(repeat):
                chunk_bytes = chunk.to_bytes()
                fec_encoded = fec_encoder.encode(chunk_bytes)
                chunk_audio = modem.encode_data(fec_encoded)
                
                # Normalize audio
                chunk_audio = normalize_audio(chunk_audio)
                audio_tx.play(chunk_audio)
                
                if repeat > 1:
                    time.sleep(0.5)  # Delay between repeats
        
        print("Transmission completed")
        return True
        
    except KeyboardInterrupt:
        print("\nTransmission interrupted")
        return False
    except Exception as e:
        print(f"Transmission error: {e}")
        return False
    finally:
        audio_tx.stop()


def receive_file(output_dir: str, device: Optional[int] = None, timeout: float = 300.0):
    """Receive a file from audio."""
    print(f"Receiving file to: {output_dir}")
    print(f"Timeout: {timeout} seconds")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Create modem
    modem = MFSKModem()
    
    # Create FEC decoder
    fec_decoder = FECDecoder()
    
    # Create frame assembler
    frame_assembler = FrameAssembler()
    
    # Create frame receiver
    frame_rx = FrameReceiver(modem, frame_assembler)
    
    # Create audio receiver
    audio_rx = AudioReceiver()
    
    # State
    file_header = None
    chunks = []
    received_frames = []
    start_time = time.time()
    
    try:
        # Start audio reception
        audio_rx.start(device)
        print("Audio reception started")
        
        while time.time() - start_time < timeout:
            # Read audio
            audio = audio_rx.read(timeout=1.0)
            if audio is None:
                continue
            
            # Process audio for frames
            frames = frame_rx.process_audio(audio)
            
            for frame in frames:
                print(f"Received frame {frame.frame_id}")
                
                # Try to decode frame
                try:
                    frame_bytes = frame.to_bytes()
                    decoded_bytes, success = fec_decoder.decode(frame_bytes)
                    
                    if success:
                        # Try to parse as file header
                        if file_header is None:
                            file_header = FileHeader.from_bytes(decoded_bytes)
                            if file_header:
                                print(f"File header received: {file_header.filename} ({file_header.file_size} bytes)")
                                continue
                        
                        # Try to parse as chunk
                        chunk = Chunk.from_bytes(decoded_bytes)
                        if chunk:
                            chunks.append(chunk)
                            print(f"Chunk {chunk.chunk_id} received ({len(chunk.data)} bytes)")
                            
                            # Check if all chunks received
                            if chunk.is_last:
                                print("Last chunk received, reassembling file...")
                                
                                # Reassemble file
                                chunker = FileChunker()
                                success = chunker.reassemble_file(file_header, chunks, output_dir)
                                
                                if success:
                                    print("File received successfully!")
                                    return True
                                else:
                                    print("File reassembly failed")
                                    return False
                    
                except Exception as e:
                    print(f"Frame processing error: {e}")
                    continue
        
        print("Receive timeout")
        return False
        
    except KeyboardInterrupt:
        print("\nReception interrupted")
        return False
    except Exception as e:
        print(f"Reception error: {e}")
        return False
    finally:
        audio_rx.stop()


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description='Audio-over-RDP File Teleport')
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # List devices command
    list_parser = subparsers.add_parser('list', help='List audio devices')
    
    # Transmit command
    tx_parser = subparsers.add_parser('tx', help='Transmit file')
    tx_parser.add_argument('file', help='File to transmit')
    tx_parser.add_argument('--rate', choices=['safe', 'default', 'fast'], 
                          default='default', help='Transmission rate')
    tx_parser.add_argument('--repeat', type=int, default=1, help='Number of repeats')
    tx_parser.add_argument('--tone-start', type=int, default=1200, help='Starting tone frequency (Hz)')
    tx_parser.add_argument('--tone-step', type=int, default=120, help='Tone step frequency (Hz)')
    tx_parser.add_argument('--device', type=int, help='Audio device index')
    
    # Receive command
    rx_parser = subparsers.add_parser('rx', help='Receive file')
    rx_parser.add_argument('output_dir', help='Output directory')
    rx_parser.add_argument('--device', type=int, help='Audio device index')
    rx_parser.add_argument('--timeout', type=float, default=300.0, help='Receive timeout (seconds)')
    
    args = parser.parse_args()
    
    if args.command == 'list':
        list_devices()
    elif args.command == 'tx':
        success = transmit_file(
            filepath=args.file,
            rate=args.rate,
            repeat=args.repeat,
            tone_start=args.tone_start,
            tone_step=args.tone_step,
            device=args.device
        )
        sys.exit(0 if success else 1)
    elif args.command == 'rx':
        success = receive_file(
            output_dir=args.output_dir,
            device=args.device,
            timeout=args.timeout
        )
        sys.exit(0 if success else 1)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == '__main__':
    main()
