# Audio-over-RDP File Teleport

A Python CLI application for transmitting files as audio signals over RDP connections. This system allows you to transfer files from a remote Windows 11 host (via RDP) to a local Windows 11 machine using audio signals.

## Features

- **Audio-over-RDP Transmission**: Transmits files as audio signals that can be captured via WASAPI loopback
- **Robust MFSK Modulation**: Uses Multiple Frequency Shift Keying with configurable rates
- **Forward Error Correction**: Reed-Solomon FEC with interleaving for error resilience
- **File Chunking**: Supports large files by splitting them into manageable chunks
- **WASAPI Loopback**: Captures audio from RDP sessions without over-the-air transmission

## Installation

### Prerequisites

- Windows 11 (both host and local machine)
- Python 3.8 or higher
- Audio devices with WASAPI support

### Install Dependencies

```powershell
# Install Python dependencies
pip install -r requirements.txt
```

## Usage

### List Audio Devices

```powershell
python -m teleport list
```

### Transmit File (on remote host via RDP)

```powershell
# Basic transmission
python -m teleport tx "path/to/file.txt"

# Advanced options
python -m teleport tx "path/to/file.txt" --rate fast --repeat 2 --device 0
```

**Transmit Options:**
- `--rate`: Transmission rate (`safe`, `default`, `fast`)
- `--repeat`: Number of times to repeat transmission
- `--tone-start`: Starting tone frequency in Hz (default: 1200)
- `--tone-step`: Tone step frequency in Hz (default: 120)
- `--device`: Audio device index (use `list` command to find)

### Receive File (on local machine)

```powershell
# Basic reception
python -m teleport rx "output_directory"

# Advanced options
python -m teleport rx "output_directory" --device 1 --timeout 300
```

**Receive Options:**
- `--device`: Audio device index for loopback capture
- `--timeout`: Receive timeout in seconds (default: 300)

## Transmission Rates

| Rate | Tones | Symbol Rate | Throughput | Use Case |
|------|-------|-------------|------------|----------|
| `safe` | 8 | 800 sps | ~2.4 kbps | Poor audio quality |
| `default` | 16 | 1000 sps | ~4 kbps | Balanced performance |
| `fast` | 32 | 1600 sps | ~8 kbps | Good audio quality |

## How It Works

1. **File Preparation**: File is split into chunks with metadata headers
2. **FEC Encoding**: Reed-Solomon error correction and interleaving applied
3. **Framing**: Data is organized into frames with CRC32 checksums
4. **MFSK Modulation**: Digital data converted to audio tones
5. **Audio Transmission**: Tones played through speakers (RDP redirects to local)
6. **Audio Reception**: WASAPI loopback captures the audio signal
7. **Demodulation**: Audio tones converted back to digital data
8. **FEC Decoding**: Error correction and deinterleaving applied
9. **File Reassembly**: Chunks reassembled into original file

## Audio Device Setup

### For Transmission (Remote Host)
- Use default audio output device
- RDP will redirect audio to local machine
- No special setup required

### For Reception (Local Machine)
- Use WASAPI loopback device
- Look for "Speakers (Remote Audio)" or similar
- Use `python -m teleport list` to find available devices

## Troubleshooting

### Common Issues

1. **No Audio Devices Found**
   - Ensure audio drivers are installed
   - Check Windows audio settings
   - Try running as administrator

2. **Poor Reception Quality**
   - Use `safe` rate for better reliability
   - Increase `--repeat` for redundancy
   - Check audio levels and device selection

3. **File Transfer Fails**
   - Verify file size (very large files may timeout)
   - Check audio device selection
   - Ensure stable RDP connection

### Testing

Run the test suite to verify functionality:

```powershell
python test_teleport.py
```

## Technical Details

### Audio Parameters
- Sample Rate: 48 kHz
- Channels: Mono
- Format: 32-bit float

### Modulation
- MFSK (Multiple Frequency Shift Keying)
- Tone spacing: 120 Hz
- Frequency range: 900-4500 Hz
- Raised cosine windowing

### Error Correction
- Reed-Solomon (255, 223) code
- Interleaving depth: 8
- CRC32 frame checksums

### File Format
- Chunk size: 64 KB
- File header with metadata
- SHA-256 integrity verification

## Limitations

- Requires stable RDP connection
- Audio quality affects transmission reliability
- Large files may take significant time
- Not suitable for real-time applications

## License

This project is provided as-is for educational and experimental purposes.
