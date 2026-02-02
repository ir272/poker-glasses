# CLAUDE.md - Poker Glasses Implementation Plan

## Project Overview

AI-powered poker card detection demo using IR-marked cards. The system detects invisible IR marks on playing cards and announces detected cards via audio - demonstrating potential AI misuse for awareness purposes.

## Architecture

```
┌──────────────────────┐     ┌──────────────────────┐     ┌──────────────────────┐
│   Arducam B0506      │     │   Detection Engine   │     │    Audio Output      │
│   USB IR Camera      │────▶│   (Roboflow Local)   │────▶│    (pyttsx3/TTS)     │
│   850nm IR LEDs      │     │                      │     │                      │
└──────────────────────┘     └──────────────────────┘     └──────────────────────┘
                                       │
                                       ▼
                              ┌──────────────────────┐
                              │    Mark Registry     │
                              │  (config/cards.json) │
                              └──────────────────────┘
```

## Hardware

- **Camera**: Arducam 1080P Day/Night Vision USB Camera (SKU: B0506)
  - 2MP, 1080p @ 30fps
  - Built-in 850nm IR LEDs (6x)
  - Auto IR-cut filter switching
  - UVC compliant (standard webcam interface)
- **Computer**: Any laptop (Windows/Mac/Linux)
- **Audio**: Built-in speakers, headphones, or Bluetooth earpiece
- **Cards**: Standard playing cards marked with IR-visible ink

## Project Structure

```
poker-glasses/
├── CLAUDE.md              # This file - implementation guide
├── README.md              # Project overview
├── LICENSE
├── pyproject.toml         # Dependencies (use uv)
├── config/
│   └── cards.json         # Mark-to-card mapping
├── src/
│   ├── __init__.py
│   ├── main.py            # Entry point
│   ├── camera.py          # USB camera capture
│   ├── detector.py        # Roboflow inference
│   ├── audio.py           # TTS announcement engine
│   └── registry.py        # Mark-to-card lookup
├── models/                # Downloaded Roboflow model (gitignored)
│   └── .gitkeep
├── scripts/
│   ├── test_camera.py     # Verify camera works
│   ├── capture_training.py # Helper to capture training images
│   └── test_audio.py      # Verify TTS works
└── tests/
    └── test_registry.py
```

## Implementation Phases

### Phase 1: Project Setup & Camera Verification

**Goal**: Get frames from the IR camera displayed on screen.

**Files to create**:

1. `pyproject.toml`:
```toml
[project]
name = "poker-glasses"
version = "0.1.0"
description = "IR card detection demo"
requires-python = ">=3.10"
dependencies = [
    "opencv-python>=4.8.0",
    "numpy>=1.24.0",
    "pyttsx3>=2.90",
    "inference>=0.9.0",  # Roboflow local inference
]

[project.optional-dependencies]
dev = ["pytest>=7.0.0", "ruff>=0.1.0"]
```

2. `src/camera.py`:
```python
"""
USB Camera interface using OpenCV.

The Arducam B0506 presents as a standard UVC webcam.
Key considerations:
- Camera index may vary (usually 0 or 1)
- May need to disable auto IR-cut filter for consistent IR capture
- Resolution should be set explicitly for consistent detection
"""

import cv2
import numpy as np
from dataclasses import dataclass
from typing import Optional, Generator

@dataclass
class CameraConfig:
    device_index: int = 0
    width: int = 1280
    height: int = 720
    fps: int = 30

class Camera:
    def __init__(self, config: CameraConfig = None):
        self.config = config or CameraConfig()
        self._cap: Optional[cv2.VideoCapture] = None
    
    def open(self) -> bool:
        """Initialize camera. Returns True if successful."""
        # Implementation: open capture, set resolution, verify frames
        pass
    
    def read(self) -> Optional[np.ndarray]:
        """Read single frame. Returns None if failed."""
        pass
    
    def stream(self) -> Generator[np.ndarray, None, None]:
        """Yield frames continuously."""
        pass
    
    def close(self) -> None:
        """Release camera resources."""
        pass
    
    def __enter__(self):
        self.open()
        return self
    
    def __exit__(self, *args):
        self.close()
```

3. `scripts/test_camera.py`:
```python
"""
Quick test to verify camera is working.
Run: python scripts/test_camera.py

Expected: Window showing camera feed. Press 'q' to quit.
If IR LEDs are on, you should see them glowing in the preview.
"""
# Implementation: open camera, show frames in window, print FPS
```

**Acceptance criteria**:
- [ ] `uv sync` installs all dependencies
- [ ] `python scripts/test_camera.py` shows live camera feed
- [ ] Can see IR LED illumination in dark/low-light
- [ ] Achieves 25+ FPS

---

### Phase 2: Audio Engine

**Goal**: Text-to-speech that announces card names clearly.

**Files to create**:

1. `src/audio.py`:
```python
"""
Audio announcement engine.

Requirements:
- Non-blocking: announcements shouldn't freeze detection
- Debounced: don't repeat same card within N seconds
- Queue-based: handle rapid detections gracefully
"""

import pyttsx3
import threading
import queue
from dataclasses import dataclass
from typing import Optional
import time

@dataclass 
class AudioConfig:
    rate: int = 150          # Words per minute
    volume: float = 1.0      # 0.0 to 1.0
    debounce_seconds: float = 3.0  # Don't repeat same card within this window

class AudioEngine:
    def __init__(self, config: AudioConfig = None):
        self.config = config or AudioConfig()
        self._engine: Optional[pyttsx3.Engine] = None
        self._queue: queue.Queue = queue.Queue()
        self._thread: Optional[threading.Thread] = None
        self._running: bool = False
        self._recent: dict[str, float] = {}  # card -> last_announced_time
    
    def start(self) -> None:
        """Start the audio engine in background thread."""
        pass
    
    def announce(self, card_name: str) -> None:
        """Queue a card announcement (debounced)."""
        pass
    
    def stop(self) -> None:
        """Stop the audio engine."""
        pass
    
    def _worker(self) -> None:
        """Background thread that processes announcements."""
        pass
```

2. `scripts/test_audio.py`:
```python
"""
Test TTS output.
Run: python scripts/test_audio.py

Expected: Hear "King of Hearts", "Ace of Spades" announced.
"""
```

**Acceptance criteria**:
- [ ] `python scripts/test_audio.py` produces clear audio
- [ ] Announcements don't block main thread
- [ ] Same card isn't repeated within debounce window

---

### Phase 3: Mark Registry

**Goal**: Map detected mark classes to card names.

**Files to create**:

1. `config/cards.json`:
```json
{
  "marks": {
    "circle": {
      "card": "King of Hearts",
      "abbreviation": "Kh"
    },
    "triangle": {
      "card": "Ace of Spades", 
      "abbreviation": "As"
    },
    "square": {
      "card": "Queen of Diamonds",
      "abbreviation": "Qd"
    },
    "cross": {
      "card": "Jack of Clubs",
      "abbreviation": "Jc"
    }
  },
  "settings": {
    "confidence_threshold": 0.7
  }
}
```

2. `src/registry.py`:
```python
"""
Mark-to-card registry.

Loads config/cards.json and provides lookup from 
Roboflow detection class names to human-readable card names.
"""

import json
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

@dataclass
class Card:
    name: str           # "King of Hearts"
    abbreviation: str   # "Kh"
    mark_class: str     # "circle" (Roboflow class name)

class Registry:
    def __init__(self, config_path: Path = None):
        self.config_path = config_path or Path("config/cards.json")
        self._marks: dict[str, Card] = {}
        self._confidence_threshold: float = 0.7
    
    def load(self) -> None:
        """Load configuration from JSON file."""
        pass
    
    def lookup(self, mark_class: str) -> Optional[Card]:
        """Get card info from mark class name."""
        pass
    
    @property
    def confidence_threshold(self) -> float:
        return self._confidence_threshold
    
    @property
    def known_marks(self) -> list[str]:
        """List of mark classes we can detect."""
        return list(self._marks.keys())
```

**Acceptance criteria**:
- [ ] Registry loads from JSON without errors
- [ ] Lookup returns correct card for known marks
- [ ] Lookup returns None for unknown marks
- [ ] Unit tests pass

---

### Phase 4: Detection Engine (Roboflow Integration)

**Goal**: Run inference on camera frames using locally-deployed Roboflow model.

**Prerequisites**: 
- Roboflow account with trained model
- Model exported for local inference

**Files to create**:

1. `src/detector.py`:
```python
"""
Roboflow local inference wrapper.

Uses the `inference` SDK to run detection locally (no API calls).
Model is downloaded once and cached in models/ directory.
"""

from dataclasses import dataclass
from typing import Optional
import numpy as np

@dataclass
class Detection:
    class_name: str      # Mark class from Roboflow (e.g., "circle")
    confidence: float    # 0.0 to 1.0
    bbox: tuple[int, int, int, int]  # x1, y1, x2, y2

@dataclass
class DetectorConfig:
    model_id: str        # e.g., "poker-marks/1" 
    confidence_threshold: float = 0.7

class Detector:
    def __init__(self, config: DetectorConfig):
        self.config = config
        self._model = None
    
    def load(self) -> None:
        """
        Load model for local inference.
        
        Uses: from inference import get_model
        Model is cached after first download.
        """
        pass
    
    def detect(self, frame: np.ndarray) -> list[Detection]:
        """
        Run inference on a frame.
        
        Returns list of detections above confidence threshold.
        """
        pass
    
    def detect_and_draw(self, frame: np.ndarray) -> tuple[np.ndarray, list[Detection]]:
        """
        Run inference and draw bounding boxes on frame.
        
        Useful for demo visualization.
        """
        pass
```

**Roboflow Setup Instructions** (for the human):
1. Go to app.roboflow.com, create project "poker-marks"
2. Upload training images of marked cards
3. Annotate with class names matching config/cards.json (e.g., "circle", "triangle")
4. Train model (Roboflow's AutoML is fine for this)
5. Note your model ID (format: "workspace/project/version")
6. The `inference` SDK will download and cache the model automatically

**Acceptance criteria**:
- [ ] Model loads without errors
- [ ] Detection returns results on test images
- [ ] Bounding boxes draw correctly
- [ ] Inference runs at 15+ FPS

---

### Phase 5: Main Application Loop

**Goal**: Integrate all components into working demo.

**Files to create**:

1. `src/main.py`:
```python
"""
Main application entry point.

Usage:
    python -m src.main                    # Run detection
    python -m src.main --no-audio         # Visual only (for testing)
    python -m src.main --camera-index 1   # Use different camera
"""

import argparse
import cv2
from pathlib import Path

from .camera import Camera, CameraConfig
from .detector import Detector, DetectorConfig
from .audio import AudioEngine, AudioConfig
from .registry import Registry

def parse_args():
    parser = argparse.ArgumentParser(description="Poker Glasses - IR Card Detection")
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--model-id", type=str, required=True, 
                        help="Roboflow model ID (e.g., 'your-workspace/poker-marks/1')")
    parser.add_argument("--no-audio", action="store_true", help="Disable audio announcements")
    parser.add_argument("--show-video", action="store_true", default=True,
                        help="Show detection overlay window")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Initialize components
    registry = Registry()
    registry.load()
    
    camera_config = CameraConfig(device_index=args.camera_index)
    detector_config = DetectorConfig(
        model_id=args.model_id,
        confidence_threshold=registry.confidence_threshold
    )
    
    # Main loop
    with Camera(camera_config) as camera:
        detector = Detector(detector_config)
        detector.load()
        
        audio = None
        if not args.no_audio:
            audio = AudioEngine()
            audio.start()
        
        try:
            for frame in camera.stream():
                # Detect marks
                annotated_frame, detections = detector.detect_and_draw(frame)
                
                # Announce detected cards
                for det in detections:
                    card = registry.lookup(det.class_name)
                    if card and audio:
                        audio.announce(card.name)
                
                # Display
                if args.show_video:
                    cv2.imshow("Poker Glasses", annotated_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
        finally:
            if audio:
                audio.stop()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
```

**Acceptance criteria**:
- [ ] `python -m src.main --model-id <id>` runs without errors
- [ ] Detected cards are announced via audio
- [ ] Video overlay shows bounding boxes
- [ ] System runs at 15+ FPS
- [ ] Clean shutdown on 'q' press or Ctrl+C

---

### Phase 6: Demo Polish

**Goal**: Make the demo presentation-ready.

**Enhancements**:

1. **Demo overlay** (`src/overlay.py`):
   - Show FPS counter
   - Display last detected card prominently
   - Add "DETECTED: King of Hearts" banner when card found
   - Optional: record demo video

2. **Configuration CLI**:
   - Easy camera selection if multiple cameras
   - Volume/rate adjustment for TTS
   - Confidence threshold tuning

3. **Error handling**:
   - Graceful camera disconnect handling
   - Model loading failure messages
   - Audio device not found fallback

**Nice-to-have**:
- [ ] Demo recording mode (save annotated video)
- [ ] Fullscreen mode for presentations
- [ ] Statistics display (cards detected this session)

---

## Commands Reference

```bash
# Setup
uv sync                                    # Install dependencies

# Testing individual components
python scripts/test_camera.py              # Verify camera
python scripts/test_audio.py               # Verify TTS

# Run main application
python -m src.main --model-id "workspace/project/1"

# Run without audio (visual testing)
python -m src.main --model-id "workspace/project/1" --no-audio

# Use specific camera
python -m src.main --model-id "workspace/project/1" --camera-index 1
```

## Training Data Tips

When capturing training images for Roboflow:

1. **Capture variety**: Different angles, distances, lighting
2. **Include negatives**: Plain cards without marks
3. **Mark visibility**: Ensure IR marks are clearly visible in IR camera
4. **Consistent naming**: Use exact class names from config/cards.json
5. **Minimum images**: 30-50 per mark type for decent accuracy

Use `scripts/capture_training.py` to help capture images:
```python
"""
Press 's' to save current frame to training_data/ folder.
Press 'q' to quit.
"""
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Camera not found | Try different `--camera-index` (0, 1, 2) |
| No IR illumination | Camera may need low-light to trigger IR LEDs |
| Model not loading | Check model ID format, verify Roboflow API key |
| Audio not working | Run `test_audio.py`, check system audio settings |
| Low FPS | Reduce resolution in CameraConfig, ensure no other camera apps running |

## Environment Variables

```bash
# Optional: Roboflow API key (only needed for private models)
export ROBOFLOW_API_KEY="your_key_here"
```

## Notes for Demo

1. **Lighting**: Demo works in normal room lighting - IR is additive to visible light
2. **Mark placement**: Mark corner of card face (visible when cards dealt face-up to opponent)
3. **Audio latency**: ~200-500ms from card visible to announcement
4. **Talking points**: Emphasize this is for awareness, real cheating scenarios could be more sophisticated
