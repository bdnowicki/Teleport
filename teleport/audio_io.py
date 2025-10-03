"""
Audio I/O handling for WASAPI loopback capture and playback.
"""

import sounddevice as sd
import numpy as np
from typing import Optional, List, Tuple
import threading
import queue
import time


class AudioDevice:
    """Audio device information."""
    
    def __init__(self, index: int, name: str, is_input: bool, is_output: bool):
        self.index = index
        self.name = name
        self.is_input = is_input
        self.is_output = is_output
    
    def __str__(self):
        return f"{self.name} ({'input' if self.is_input else ''}{'output' if self.is_output else ''})"


def list_audio_devices() -> List[AudioDevice]:
    """List all available audio devices."""
    devices = []
    try:
        device_list = sd.query_devices()
        for i, device in enumerate(device_list):
            if device['max_input_channels'] > 0 or device['max_output_channels'] > 0:
                devices.append(AudioDevice(
                    index=i,
                    name=device['name'],
                    is_input=device['max_input_channels'] > 0,
                    is_output=device['max_output_channels'] > 0
                ))
    except Exception as e:
        print(f"Error listing devices: {e}")
    
    return devices


def find_loopback_device() -> Optional[AudioDevice]:
    """Find the best loopback device for RDP audio capture."""
    devices = list_audio_devices()
    
    # Look for RDP-specific devices first
    for device in devices:
        if 'remote audio' in device.name.lower() or 'rdp' in device.name.lower():
            return device
    
    # Look for default output device
    try:
        default_output = sd.query_devices(kind='output')
        for device in devices:
            if device.index == default_output['index']:
                return device
    except:
        pass
    
    # Fall back to first output device
    for device in devices:
        if device.is_output:
            return device
    
    return None


class AudioTransmitter:
    """Handles audio transmission (playback)."""
    
    def __init__(self, sample_rate: int = 48000, channels: int = 1):
        self.sample_rate = sample_rate
        self.channels = channels
        self.stream = None
        self.is_playing = False
    
    def start(self, device: Optional[int] = None):
        """Start audio transmission."""
        if self.is_playing:
            return
        
        try:
            self.stream = sd.OutputStream(
                samplerate=self.sample_rate,
                channels=self.channels,
                dtype=np.float32,
                device=device
            )
            self.stream.start()
            self.is_playing = True
        except Exception as e:
            raise RuntimeError(f"Failed to start audio transmission: {e}")
    
    def play(self, audio: np.ndarray):
        """Play audio data."""
        if not self.is_playing or self.stream is None:
            return
        
        try:
            # Ensure audio is in correct format and dtype
            audio = audio.astype(np.float32)
            if audio.ndim == 1:
                audio = audio.reshape(-1, 1)
            
            self.stream.write(audio)
        except Exception as e:
            print(f"Audio playback error: {e}")
    
    def stop(self):
        """Stop audio transmission."""
        if self.stream is not None:
            self.stream.stop()
            self.stream.close()
            self.stream = None
        self.is_playing = False


class AudioReceiver:
    """Handles audio reception via WASAPI loopback."""
    
    def __init__(self, sample_rate: int = 48000, channels: int = 1, buffer_size: int = 4096):
        self.sample_rate = sample_rate
        self.channels = channels
        self.buffer_size = buffer_size
        self.stream = None
        self.is_recording = False
        self.audio_queue = queue.Queue()
        self.callback_thread = None
    
    def _audio_callback(self, indata, frames, time, status):
        """Callback for audio input."""
        if status:
            print(f"Audio callback status: {status}")
        
        if self.is_recording:
            # Convert to mono if needed
            if indata.ndim > 1:
                audio = np.mean(indata, axis=1)
            else:
                audio = indata.flatten()
            
            self.audio_queue.put(audio.copy())
    
    def start(self, device: Optional[int] = None):
        """Start audio reception."""
        if self.is_recording:
            return
        
        try:
            # Use WASAPI loopback if available
            extra_settings = None
            if device is not None:
                # Try to use WASAPI loopback for the specified device
                try:
                    extra_settings = sd.WasapiSettings(loopback=True)
                except:
                    pass  # Fall back to regular input
            
            self.stream = sd.InputStream(
                samplerate=self.sample_rate,
                channels=self.channels,
                dtype=np.float32,
                device=device,
                extra_settings=extra_settings,
                callback=self._audio_callback,
                blocksize=self.buffer_size
            )
            self.stream.start()
            self.is_recording = True
        except Exception as e:
            raise RuntimeError(f"Failed to start audio reception: {e}")
    
    def read(self, timeout: float = 1.0) -> Optional[np.ndarray]:
        """Read audio data from the queue."""
        try:
            return self.audio_queue.get(timeout=timeout)
        except queue.Empty:
            return None
    
    def stop(self):
        """Stop audio reception."""
        self.is_recording = False
        if self.stream is not None:
            self.stream.stop()
            self.stream.close()
            self.stream = None
        
        # Clear the queue
        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
            except queue.Empty:
                break


def test_audio_devices():
    """Test audio device functionality."""
    print("Available audio devices:")
    devices = list_audio_devices()
    
    for i, device in enumerate(devices):
        print(f"{i}: {device}")
    
    print(f"\nRecommended loopback device: {find_loopback_device()}")
    
    # Test basic audio I/O
    print("\nTesting audio I/O...")
    
    # Generate test tone
    duration = 1.0
    frequency = 1000
    sample_rate = 48000
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    test_audio = 0.3 * np.sin(2 * np.pi * frequency * t)
    
    # Test transmitter
    tx = AudioTransmitter(sample_rate)
    try:
        tx.start()
        print("Transmitter started successfully")
        tx.play(test_audio)
        time.sleep(0.5)
        tx.stop()
        print("Transmitter test completed")
    except Exception as e:
        print(f"Transmitter test failed: {e}")
    
    # Test receiver
    rx = AudioReceiver(sample_rate)
    try:
        rx.start()
        print("Receiver started successfully")
        time.sleep(0.5)
        rx.stop()
        print("Receiver test completed")
    except Exception as e:
        print(f"Receiver test failed: {e}")


if __name__ == "__main__":
    test_audio_devices()
