#!/usr/bin/env python3
"""
EOG Gaze Control Demo
=====================
A simple demonstration of using EOG signals for cursor control.
Students can see how eye movements translate to screen coordinates.

This demo shows:
- Real-time cursor control using HEOG (horizontal) and VEOG (vertical)
- Blink detection for "clicking"
- Simple calibration procedure
- Target hitting game mode

Usage:
    python eog_gaze_control.py              # Use LSL stream (default)
    python eog_gaze_control.py --port COM9  # Direct BioRadio connection
    python eog_gaze_control.py --simulate   # Simulate with mouse (for testing)

Controls:
    SPACE  - Start/restart calibration
    G      - Toggle game mode (hit targets)
    R      - Reset calibration
    S      - Toggle smoothing
    ESC/Q  - Quit

Author: BioRobotics Lab
"""

import numpy as np
import time
import argparse
import threading
from collections import deque
from dataclasses import dataclass, field
from typing import Optional, Tuple, List
import sys

# Try to import pygame for GUI
try:
    import pygame
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False
    print("pygame not found. Install with: pip install pygame")

# Try to import pylsl for LSL streaming
try:
    from pylsl import StreamInlet, resolve_streams as lsl_resolve_streams
    LSL_AVAILABLE = True
except ImportError:
    LSL_AVAILABLE = False
    print("pylsl not found. Install with: pip install pylsl")


# =============================================================================
# Configuration
# =============================================================================
@dataclass
class Config:
    """Configuration parameters for the gaze control demo."""
    # Window settings
    window_width: int = 1024
    window_height: int = 768
    fullscreen: bool = False
    
    # Signal processing
    sample_rate: int = 250
    buffer_size: int = 50  # samples for smoothing
    
    # Calibration defaults (will be updated during calibration)
    heog_min: float = -200.0  # μV for looking left
    heog_max: float = 200.0   # μV for looking right
    heog_center: float = 0.0
    veog_min: float = -200.0  # μV for looking down
    veog_max: float = 200.0   # μV for looking up
    veog_center: float = 0.0
    
    # Cursor settings
    cursor_size: int = 20
    cursor_smoothing: float = 0.15  # Lower = smoother but more lag
    dead_zone: float = 0.1  # Fraction of range to ignore near center
    
    # Blink detection
    blink_threshold: float = 300.0  # μV - large VEOG spike
    blink_duration_min: float = 0.05  # seconds
    blink_duration_max: float = 0.4   # seconds
    blink_cooldown: float = 0.5  # seconds between blinks
    
    # Game settings
    target_size: int = 50
    target_dwell_time: float = 0.5  # seconds to "click" a target
    num_targets: int = 5


# =============================================================================
# Color Scheme (RIT Orange/Black theme)
# =============================================================================
class Colors:
    BLACK = (26, 26, 26)
    WHITE = (255, 255, 255)
    ORANGE = (247, 105, 2)
    DARK_ORANGE = (180, 80, 0)
    LIGHT_GRAY = (200, 200, 200)
    DARK_GRAY = (80, 80, 80)
    GREEN = (0, 200, 100)
    RED = (220, 50, 50)
    BLUE = (50, 100, 200)


# =============================================================================
# Signal Processing
# =============================================================================
class SignalProcessor:
    """Processes raw EOG signals for cursor control."""
    
    def __init__(self, config: Config):
        self.config = config
        self.heog_buffer = deque(maxlen=config.buffer_size)
        self.veog_buffer = deque(maxlen=config.buffer_size)
        
        # Calibration values
        self.heog_min = config.heog_min
        self.heog_max = config.heog_max
        self.heog_center = config.heog_center
        self.veog_min = config.veog_min
        self.veog_max = config.veog_max
        self.veog_center = config.veog_center
        
        # Blink detection state
        self.last_blink_time = 0
        self.blink_start_time = 0
        self.in_blink = False
        
        # Current position (0-1 range)
        self.cursor_x = 0.5
        self.cursor_y = 0.5
        
        # Smoothing enabled
        self.smoothing_enabled = True
    
    def add_sample(self, heog: float, veog: float) -> Tuple[float, float, bool]:
        """
        Add a new sample and return normalized cursor position and blink state.
        
        Returns:
            (x, y, blink_detected) where x,y are in range [0, 1]
        """
        self.heog_buffer.append(heog)
        self.veog_buffer.append(veog)
        
        # Get smoothed values
        if self.smoothing_enabled and len(self.heog_buffer) > 5:
            heog_smooth = np.mean(list(self.heog_buffer)[-10:])
            veog_smooth = np.mean(list(self.veog_buffer)[-10:])
        else:
            heog_smooth = heog
            veog_smooth = veog
        
        # Check for blink (large positive VEOG spike)
        blink_detected = self._detect_blink(veog)
        
        # Normalize to 0-1 range
        # HEOG: positive = right, negative = left
        heog_range = self.heog_max - self.heog_min
        if heog_range > 0:
            raw_x = (heog_smooth - self.heog_min) / heog_range
        else:
            raw_x = 0.5
        
        # VEOG: positive = up, negative = down (inverted for screen coords)
        veog_range = self.veog_max - self.veog_min
        if veog_range > 0:
            raw_y = 1.0 - (veog_smooth - self.veog_min) / veog_range
        else:
            raw_y = 0.5
        
        # Apply dead zone around center
        raw_x = self._apply_dead_zone(raw_x)
        raw_y = self._apply_dead_zone(raw_y)
        
        # Clamp to valid range
        raw_x = max(0.0, min(1.0, raw_x))
        raw_y = max(0.0, min(1.0, raw_y))
        
        # Smooth cursor movement
        alpha = self.config.cursor_smoothing
        self.cursor_x = self.cursor_x * (1 - alpha) + raw_x * alpha
        self.cursor_y = self.cursor_y * (1 - alpha) + raw_y * alpha
        
        return self.cursor_x, self.cursor_y, blink_detected
    
    def _apply_dead_zone(self, value: float) -> float:
        """Apply dead zone around center (0.5)."""
        dz = self.config.dead_zone
        if abs(value - 0.5) < dz:
            return 0.5
        elif value > 0.5:
            return 0.5 + (value - 0.5 - dz) / (0.5 - dz) * 0.5
        else:
            return 0.5 - (0.5 - value - dz) / (0.5 - dz) * 0.5
    
    def _detect_blink(self, veog: float) -> bool:
        """Detect blink based on large VEOG spike."""
        current_time = time.time()
        
        # Check cooldown
        if current_time - self.last_blink_time < self.config.blink_cooldown:
            return False
        
        # Check for blink start
        if not self.in_blink and veog > self.config.blink_threshold:
            self.in_blink = True
            self.blink_start_time = current_time
            return False
        
        # Check for blink end
        if self.in_blink and veog < self.config.blink_threshold * 0.5:
            self.in_blink = False
            duration = current_time - self.blink_start_time
            
            # Valid blink if duration is in expected range
            if self.config.blink_duration_min < duration < self.config.blink_duration_max:
                self.last_blink_time = current_time
                return True
        
        return False
    
    def reset_calibration(self):
        """Reset calibration to defaults."""
        self.heog_min = self.config.heog_min
        self.heog_max = self.config.heog_max
        self.heog_center = self.config.heog_center
        self.veog_min = self.config.veog_min
        self.veog_max = self.config.veog_max
        self.veog_center = self.config.veog_center


# =============================================================================
# Calibration Manager
# =============================================================================
class CalibrationManager:
    """Handles the calibration procedure."""
    
    STATES = ['idle', 'center', 'left', 'right', 'up', 'down', 'complete']
    
    def __init__(self, config: Config):
        self.config = config
        self.state = 'idle'
        self.state_start_time = 0
        self.state_duration = 2.0  # seconds per calibration point
        self.collected_samples: List[Tuple[float, float]] = []
        
        # Calibration results
        self.center_heog = 0.0
        self.center_veog = 0.0
        self.left_heog = 0.0
        self.right_heog = 0.0
        self.up_veog = 0.0
        self.down_veog = 0.0
    
    def start(self):
        """Start calibration procedure."""
        self.state = 'center'
        self.state_start_time = time.time()
        self.collected_samples = []
    
    def add_sample(self, heog: float, veog: float):
        """Add a sample during calibration."""
        if self.state != 'idle' and self.state != 'complete':
            self.collected_samples.append((heog, veog))
    
    def update(self) -> bool:
        """
        Update calibration state machine.
        Returns True if calibration is still in progress.
        """
        if self.state == 'idle' or self.state == 'complete':
            return False
        
        elapsed = time.time() - self.state_start_time
        
        if elapsed >= self.state_duration:
            # Process collected samples
            self._process_state()
            
            # Move to next state
            current_idx = self.STATES.index(self.state)
            if current_idx < len(self.STATES) - 1:
                self.state = self.STATES[current_idx + 1]
                self.state_start_time = time.time()
                self.collected_samples = []
            
            if self.state == 'complete':
                return False
        
        return True
    
    def _process_state(self):
        """Process samples collected during current state."""
        if not self.collected_samples:
            return
        
        heog_values = [s[0] for s in self.collected_samples]
        veog_values = [s[1] for s in self.collected_samples]
        
        # Use median for robustness
        heog_median = np.median(heog_values)
        veog_median = np.median(veog_values)
        
        if self.state == 'center':
            self.center_heog = heog_median
            self.center_veog = veog_median
        elif self.state == 'left':
            self.left_heog = heog_median
        elif self.state == 'right':
            self.right_heog = heog_median
        elif self.state == 'up':
            self.up_veog = veog_median
        elif self.state == 'down':
            self.down_veog = veog_median
    
    def get_calibration(self) -> dict:
        """Get calibration results."""
        return {
            'heog_min': self.left_heog,
            'heog_max': self.right_heog,
            'heog_center': self.center_heog,
            'veog_min': self.down_veog,
            'veog_max': self.up_veog,
            'veog_center': self.center_veog,
        }
    
    def get_progress(self) -> float:
        """Get calibration progress (0-1)."""
        if self.state == 'idle':
            return 0.0
        if self.state == 'complete':
            return 1.0
        
        state_idx = self.STATES.index(self.state)
        elapsed = time.time() - self.state_start_time
        state_progress = min(elapsed / self.state_duration, 1.0)
        
        return (state_idx - 1 + state_progress) / (len(self.STATES) - 2)
    
    def get_instruction(self) -> str:
        """Get current calibration instruction."""
        instructions = {
            'idle': "Press SPACE to start calibration",
            'center': "Look at the CENTER of the screen",
            'left': "Look LEFT",
            'right': "Look RIGHT",
            'up': "Look UP",
            'down': "Look DOWN",
            'complete': "Calibration complete!"
        }
        return instructions.get(self.state, "")


# =============================================================================
# Game Mode (Target Hitting)
# =============================================================================
class TargetGame:
    """Simple target-hitting game to demonstrate gaze control."""
    
    def __init__(self, config: Config, screen_width: int, screen_height: int):
        self.config = config
        self.screen_width = screen_width
        self.screen_height = screen_height
        
        self.targets: List[Tuple[int, int]] = []
        self.current_target_idx = 0
        self.dwell_start_time = 0
        self.on_target = False
        self.score = 0
        self.total_targets = 0
        self.active = False
        
    def start(self):
        """Start a new game."""
        self.generate_targets()
        self.current_target_idx = 0
        self.score = 0
        self.total_targets = 0
        self.active = True
        self.dwell_start_time = 0
        self.on_target = False
    
    def stop(self):
        """Stop the game."""
        self.active = False
        self.targets = []
    
    def generate_targets(self):
        """Generate random target positions."""
        margin = self.config.target_size * 2
        self.targets = []
        
        for _ in range(self.config.num_targets):
            x = np.random.randint(margin, self.screen_width - margin)
            y = np.random.randint(margin, self.screen_height - margin)
            self.targets.append((x, y))
    
    def update(self, cursor_x: int, cursor_y: int, blink: bool) -> Optional[str]:
        """
        Update game state based on cursor position.
        Returns event string if something happened.
        """
        if not self.active or self.current_target_idx >= len(self.targets):
            return None
        
        target = self.targets[self.current_target_idx]
        distance = np.sqrt((cursor_x - target[0])**2 + (cursor_y - target[1])**2)
        
        # Check if cursor is on target
        if distance < self.config.target_size:
            if not self.on_target:
                self.on_target = True
                self.dwell_start_time = time.time()
            
            # Check dwell time or blink
            dwell_time = time.time() - self.dwell_start_time
            if dwell_time >= self.config.target_dwell_time or blink:
                # Target hit!
                self.score += 1
                self.total_targets += 1
                self.current_target_idx += 1
                self.on_target = False
                
                if self.current_target_idx >= len(self.targets):
                    self.generate_targets()
                    self.current_target_idx = 0
                
                return "hit"
        else:
            self.on_target = False
            self.dwell_start_time = 0
        
        return None
    
    def get_dwell_progress(self) -> float:
        """Get progress towards dwell activation (0-1)."""
        if not self.on_target:
            return 0.0
        return min((time.time() - self.dwell_start_time) / self.config.target_dwell_time, 1.0)
    
    def get_current_target(self) -> Optional[Tuple[int, int]]:
        """Get current target position."""
        if not self.active or self.current_target_idx >= len(self.targets):
            return None
        return self.targets[self.current_target_idx]


# =============================================================================
# Data Sources
# =============================================================================
class DataSource:
    """Base class for data sources."""
    
    def get_sample(self) -> Optional[Tuple[float, float]]:
        """Get next sample (heog, veog). Returns None if no data."""
        raise NotImplementedError
    
    def close(self):
        """Clean up resources."""
        pass


class LSLDataSource(DataSource):
    """Receive EOG data from LSL stream."""

    def __init__(self, stream_info):
        """Create from a pylsl StreamInfo object (as returned by resolve_streams)."""
        self.inlet = StreamInlet(stream_info)
        print(f"Connected to LSL stream: {stream_info.name()}")
    
    def get_sample(self) -> Optional[Tuple[float, float]]:
        sample, timestamp = self.inlet.pull_sample(timeout=0.0)
        if sample:
            # Assume channel 0 = HEOG, channel 1 = VEOG
            return (sample[0], sample[1])
        return None
    
    def close(self):
        self.inlet.close_stream()


class SimulatedDataSource(DataSource):
    """Simulate EOG data using mouse position (for testing without hardware)."""
    
    def __init__(self, screen_width: int, screen_height: int):
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.last_time = time.time()
    
    def get_sample(self) -> Optional[Tuple[float, float]]:
        # Simulate at ~250 Hz
        current_time = time.time()
        if current_time - self.last_time < 0.004:
            return None
        self.last_time = current_time
        
        # Get mouse position and convert to EOG-like values
        if PYGAME_AVAILABLE:
            mx, my = pygame.mouse.get_pos()
            
            # Convert to μV range (-200 to 200)
            heog = (mx / self.screen_width - 0.5) * 400
            veog = (0.5 - my / self.screen_height) * 400  # Inverted
            
            # Add some noise
            heog += np.random.randn() * 5
            veog += np.random.randn() * 5
            
            # Simulate blink on mouse click
            if pygame.mouse.get_pressed()[0]:
                veog = 400  # Large spike
            
            return (heog, veog)
        return None


class BioRadioDataSource(DataSource):
    """Direct connection to BioRadio device."""
    
    def __init__(self, port: str):
        # Import here to avoid dependency if not using direct connection
        try:
            from bioradio import BioRadio
        except ImportError:
            raise RuntimeError(
                "bioradio module not found. Make sure bioradio.py is in the same directory "
                "or use LSL mode instead (run bioradio_lsl_bridge.py first)."
            )
        
        print(f"Connecting to BioRadio on {port}...")
        self.device = BioRadio(port=port)
        self.device.start()
        print("BioRadio connected and streaming!")
    
    def get_sample(self) -> Optional[Tuple[float, float]]:
        data = self.device.read()
        if data is not None and len(data) >= 2:
            return (data[0], data[1])
        return None
    
    def close(self):
        self.device.stop()
        self.device.close()


# =============================================================================
# Stream Browser (LSL stream selection UI)
# =============================================================================
class StreamBrowser:
    """Pygame UI for discovering and selecting an LSL stream."""

    def __init__(self, screen, fonts):
        self.screen = screen
        self.font_large, self.font_medium, self.font_small = fonts
        self.width = screen.get_width()
        self.height = screen.get_height()
        self.clock = pygame.time.Clock()

        self.streams = []           # list of StreamInfo
        self.selected_idx = None    # index of highlighted stream
        self.scanning = False
        self.scan_thread = None
        self.status_msg = "Press  Scan  to search for LSL streams"

    # -- background scan ---------------------------------------------------
    def _scan_worker(self):
        try:
            found = lsl_resolve_streams(wait_time=2.0)
            self.streams = found
            if found:
                self.status_msg = f"Found {len(found)} stream(s). Click one then press Connect."
                self.selected_idx = 0
            else:
                self.status_msg = "No streams found. Make sure your device is streaming."
        except Exception as e:
            self.status_msg = f"Scan error: {e}"
        self.scanning = False

    def start_scan(self):
        if self.scanning:
            return
        self.scanning = True
        self.streams = []
        self.selected_idx = None
        self.status_msg = "Scanning..."
        self.scan_thread = threading.Thread(target=self._scan_worker, daemon=True)
        self.scan_thread.start()

    # -- button helpers ----------------------------------------------------
    @staticmethod
    def _point_in_rect(pos, rect):
        return rect[0] <= pos[0] <= rect[0] + rect[2] and rect[1] <= pos[1] <= rect[1] + rect[3]

    def _draw_button(self, text, rect, enabled=True):
        color = Colors.ORANGE if enabled else Colors.DARK_GRAY
        pygame.draw.rect(self.screen, color, rect, border_radius=6)
        pygame.draw.rect(self.screen, Colors.BLACK, rect, 2, border_radius=6)
        label = self.font_medium.render(text, True, Colors.WHITE if enabled else Colors.LIGHT_GRAY)
        label_rect = label.get_rect(center=(rect[0] + rect[2] // 2, rect[1] + rect[3] // 2))
        self.screen.blit(label, label_rect)

    # -- main loop ---------------------------------------------------------
    def run(self):
        """Run the stream browser. Returns a StreamInfo or None (user quit)."""
        # Button rects
        btn_w, btn_h = 160, 44
        scan_rect = (self.width // 2 - btn_w - 20, self.height - 80, btn_w, btn_h)
        connect_rect = (self.width // 2 + 20, self.height - 80, btn_w, btn_h)
        sim_rect = (self.width // 2 - btn_w // 2, self.height - 130, btn_w, btn_h)

        # Kick off an initial scan automatically
        self.start_scan()

        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return None
                if event.type == pygame.KEYDOWN:
                    if event.key in (pygame.K_ESCAPE, pygame.K_q):
                        return None
                    if event.key == pygame.K_RETURN and self.selected_idx is not None:
                        return self.streams[self.selected_idx]
                    if event.key == pygame.K_UP and self.selected_idx is not None and self.selected_idx > 0:
                        self.selected_idx -= 1
                    if event.key == pygame.K_DOWN and self.selected_idx is not None and self.selected_idx < len(self.streams) - 1:
                        self.selected_idx += 1
                if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    pos = event.pos
                    # Scan button
                    if self._point_in_rect(pos, scan_rect) and not self.scanning:
                        self.start_scan()
                    # Connect button
                    if self._point_in_rect(pos, connect_rect) and self.selected_idx is not None:
                        return self.streams[self.selected_idx]
                    # Simulate button
                    if self._point_in_rect(pos, sim_rect):
                        return "SIMULATE"
                    # Click on a stream row
                    for i, row_rect in enumerate(self._row_rects):
                        if self._point_in_rect(pos, row_rect):
                            self.selected_idx = i

            # -- draw --
            self.screen.fill(Colors.WHITE)

            # Title
            title = self.font_large.render("LSL Stream Browser", True, Colors.ORANGE)
            self.screen.blit(title, title.get_rect(center=(self.width // 2, 40)))

            # Status
            status = self.font_small.render(self.status_msg, True, Colors.DARK_GRAY)
            self.screen.blit(status, status.get_rect(center=(self.width // 2, 80)))

            # Stream list
            self._row_rects = []
            list_x = 80
            list_y = 110
            row_h = 50
            row_w = self.width - 160
            for i, s in enumerate(self.streams):
                rect = (list_x, list_y + i * (row_h + 4), row_w, row_h)
                self._row_rects.append(rect)
                bg = Colors.ORANGE if i == self.selected_idx else Colors.LIGHT_GRAY
                pygame.draw.rect(self.screen, bg, rect, border_radius=4)
                pygame.draw.rect(self.screen, Colors.DARK_GRAY, rect, 1, border_radius=4)
                text_color = Colors.WHITE if i == self.selected_idx else Colors.BLACK
                line1 = self.font_medium.render(s.name(), True, text_color)
                line2 = self.font_small.render(
                    f"Type: {s.type()}   Channels: {s.channel_count()}   Rate: {s.nominal_srate():.0f} Hz",
                    True, text_color)
                self.screen.blit(line1, (rect[0] + 12, rect[1] + 4))
                self.screen.blit(line2, (rect[0] + 12, rect[1] + 28))

            # Scanning indicator
            if self.scanning:
                dots = "." * (int(time.time() * 3) % 4)
                scanning_text = self.font_medium.render(f"Scanning{dots}", True, Colors.DARK_GRAY)
                self.screen.blit(scanning_text, scanning_text.get_rect(center=(self.width // 2, list_y + 10)))

            # Buttons
            self._draw_button("Scan", scan_rect, enabled=not self.scanning)
            self._draw_button("Connect", connect_rect, enabled=self.selected_idx is not None)
            self._draw_button("Simulate", sim_rect, enabled=True)

            # Help text
            help_text = self.font_small.render("Arrow keys to select  |  Enter to connect  |  ESC to quit", True, Colors.DARK_GRAY)
            self.screen.blit(help_text, help_text.get_rect(center=(self.width // 2, self.height - 20)))

            pygame.display.flip()
            self.clock.tick(30)


# =============================================================================
# Main Application
# =============================================================================
class GazeControlApp:
    """Main application class."""
    
    def __init__(self, config: Config, data_source: DataSource):
        self.config = config
        self.data_source = data_source

        # Initialize pygame (reuse existing display if already created)
        if not pygame.get_init():
            pygame.init()

        existing = pygame.display.get_surface()
        if existing is not None:
            self.screen = existing
        elif config.fullscreen:
            self.screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
            config.window_width = self.screen.get_width()
            config.window_height = self.screen.get_height()
        else:
            self.screen = pygame.display.set_mode(
                (config.window_width, config.window_height)
            )
        
        pygame.display.set_caption("EOG Gaze Control Demo")
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.Font(None, 48)
        self.font_medium = pygame.font.Font(None, 32)
        self.font_small = pygame.font.Font(None, 24)
        
        # Components
        self.processor = SignalProcessor(config)
        self.calibration = CalibrationManager(config)
        self.game = TargetGame(config, config.window_width, config.window_height)
        
        # State
        self.running = True
        self.cursor_x = config.window_width // 2
        self.cursor_y = config.window_height // 2
        self.last_blink = False
        self.blink_indicator_time = 0
        
        # Signal display
        self.heog_history = deque(maxlen=200)
        self.veog_history = deque(maxlen=200)
        self.raw_heog = 0.0
        self.raw_veog = 0.0
    
    def run(self):
        """Main application loop."""
        while self.running:
            self.handle_events()
            self.update()
            self.draw()
            self.clock.tick(60)
        
        self.cleanup()
    
    def handle_events(self):
        """Handle pygame events."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE or event.key == pygame.K_q:
                    self.running = False
                
                elif event.key == pygame.K_SPACE:
                    if self.calibration.state == 'idle' or self.calibration.state == 'complete':
                        self.calibration.start()
                        self.game.stop()
                
                elif event.key == pygame.K_g:
                    if self.calibration.state == 'complete':
                        if self.game.active:
                            self.game.stop()
                        else:
                            self.game.start()
                
                elif event.key == pygame.K_r:
                    self.processor.reset_calibration()
                    self.calibration.state = 'idle'
                
                elif event.key == pygame.K_s:
                    self.processor.smoothing_enabled = not self.processor.smoothing_enabled
    
    def update(self):
        """Update application state."""
        # Get samples from data source
        blink = False
        sample = self.data_source.get_sample()
        
        while sample is not None:
            heog, veog = sample
            self.raw_heog = heog
            self.raw_veog = veog
            self.heog_history.append(heog)
            self.veog_history.append(veog)
            
            # During calibration, collect samples
            if self.calibration.state not in ['idle', 'complete']:
                self.calibration.add_sample(heog, veog)
            
            # Process for cursor control
            norm_x, norm_y, blink_detected = self.processor.add_sample(heog, veog)
            
            if blink_detected:
                blink = True
                self.blink_indicator_time = time.time()
            
            # Convert to screen coordinates
            self.cursor_x = int(norm_x * self.config.window_width)
            self.cursor_y = int(norm_y * self.config.window_height)
            
            # Get next sample
            sample = self.data_source.get_sample()
        
        # Update calibration
        if self.calibration.update() == False and self.calibration.state == 'complete':
            # Apply calibration results
            cal = self.calibration.get_calibration()
            self.processor.heog_min = cal['heog_min']
            self.processor.heog_max = cal['heog_max']
            self.processor.heog_center = cal['heog_center']
            self.processor.veog_min = cal['veog_min']
            self.processor.veog_max = cal['veog_max']
            self.processor.veog_center = cal['veog_center']
        
        # Update game
        if self.game.active:
            event = self.game.update(self.cursor_x, self.cursor_y, blink)
            if event == "hit":
                # Play sound or visual feedback
                pass
        
        self.last_blink = blink
    
    def draw(self):
        """Draw the application."""
        # Clear screen
        self.screen.fill(Colors.WHITE)
        
        # Draw signal traces (top-left corner)
        self.draw_signal_traces()
        
        # Draw calibration UI or game
        if self.calibration.state not in ['idle', 'complete']:
            self.draw_calibration()
        else:
            if self.game.active:
                self.draw_game()
            else:
                self.draw_idle()
        
        # Draw cursor
        self.draw_cursor()
        
        # Draw status bar
        self.draw_status_bar()
        
        # Blink indicator
        if time.time() - self.blink_indicator_time < 0.3:
            self.draw_blink_indicator()
        
        pygame.display.flip()
    
    def draw_signal_traces(self):
        """Draw small signal traces in corner."""
        trace_width = 200
        trace_height = 60
        margin = 10
        
        # HEOG trace
        pygame.draw.rect(self.screen, Colors.LIGHT_GRAY,
                        (margin, margin, trace_width, trace_height))
        pygame.draw.rect(self.screen, Colors.DARK_GRAY,
                        (margin, margin, trace_width, trace_height), 1)
        
        if len(self.heog_history) > 1:
            # Normalize to trace height
            values = list(self.heog_history)
            min_val, max_val = min(values), max(values)
            if max_val - min_val > 0:
                points = []
                for i, v in enumerate(values):
                    x = margin + i * trace_width // len(values)
                    y = margin + trace_height - int((v - min_val) / (max_val - min_val) * trace_height)
                    points.append((x, y))
                if len(points) > 1:
                    pygame.draw.lines(self.screen, Colors.ORANGE, False, points, 2)
        
        label = self.font_small.render(f"HEOG: {self.raw_heog:.0f} μV", True, Colors.BLACK)
        self.screen.blit(label, (margin + 5, margin + 5))
        
        # VEOG trace
        pygame.draw.rect(self.screen, Colors.LIGHT_GRAY,
                        (margin, margin + trace_height + 5, trace_width, trace_height))
        pygame.draw.rect(self.screen, Colors.DARK_GRAY,
                        (margin, margin + trace_height + 5, trace_width, trace_height), 1)
        
        if len(self.veog_history) > 1:
            values = list(self.veog_history)
            min_val, max_val = min(values), max(values)
            if max_val - min_val > 0:
                points = []
                for i, v in enumerate(values):
                    x = margin + i * trace_width // len(values)
                    y = margin + trace_height + 5 + trace_height - int((v - min_val) / (max_val - min_val) * trace_height)
                    points.append((x, y))
                if len(points) > 1:
                    pygame.draw.lines(self.screen, Colors.BLUE, False, points, 2)
        
        label = self.font_small.render(f"VEOG: {self.raw_veog:.0f} μV", True, Colors.BLACK)
        self.screen.blit(label, (margin + 5, margin + trace_height + 10))
    
    def draw_calibration(self):
        """Draw calibration UI."""
        cx, cy = self.config.window_width // 2, self.config.window_height // 2
        
        # Draw instruction
        instruction = self.calibration.get_instruction()
        text = self.font_large.render(instruction, True, Colors.BLACK)
        text_rect = text.get_rect(center=(cx, cy - 100))
        self.screen.blit(text, text_rect)
        
        # Draw target point based on state
        target_positions = {
            'center': (cx, cy),
            'left': (100, cy),
            'right': (self.config.window_width - 100, cy),
            'up': (cx, 100),
            'down': (cx, self.config.window_height - 100),
        }
        
        if self.calibration.state in target_positions:
            tx, ty = target_positions[self.calibration.state]
            
            # Draw pulsing target
            pulse = abs(np.sin(time.time() * 4)) * 10
            pygame.draw.circle(self.screen, Colors.ORANGE, (tx, ty), int(30 + pulse))
            pygame.draw.circle(self.screen, Colors.WHITE, (tx, ty), 15)
            pygame.draw.circle(self.screen, Colors.ORANGE, (tx, ty), 5)
        
        # Draw progress bar
        progress = self.calibration.get_progress()
        bar_width = 400
        bar_height = 20
        bar_x = cx - bar_width // 2
        bar_y = cy + 150
        
        pygame.draw.rect(self.screen, Colors.LIGHT_GRAY,
                        (bar_x, bar_y, bar_width, bar_height))
        pygame.draw.rect(self.screen, Colors.ORANGE,
                        (bar_x, bar_y, int(bar_width * progress), bar_height))
        pygame.draw.rect(self.screen, Colors.DARK_GRAY,
                        (bar_x, bar_y, bar_width, bar_height), 2)
        
        text = self.font_small.render(f"{int(progress * 100)}%", True, Colors.BLACK)
        text_rect = text.get_rect(center=(cx, bar_y + bar_height + 20))
        self.screen.blit(text, text_rect)
    
    def draw_game(self):
        """Draw game mode."""
        # Draw current target
        target = self.game.get_current_target()
        if target:
            tx, ty = target
            
            # Draw target ring
            pygame.draw.circle(self.screen, Colors.LIGHT_GRAY, (tx, ty),
                             self.config.target_size + 10)
            pygame.draw.circle(self.screen, Colors.ORANGE, (tx, ty),
                             self.config.target_size)
            
            # Draw dwell progress
            progress = self.game.get_dwell_progress()
            if progress > 0:
                # Draw progress arc
                rect = pygame.Rect(tx - self.config.target_size, ty - self.config.target_size,
                                  self.config.target_size * 2, self.config.target_size * 2)
                pygame.draw.arc(self.screen, Colors.GREEN, rect,
                              -np.pi/2, -np.pi/2 + progress * 2 * np.pi, 5)
            
            # Draw center dot
            pygame.draw.circle(self.screen, Colors.WHITE, (tx, ty), 10)
        
        # Draw score
        score_text = self.font_medium.render(f"Score: {self.game.score}", True, Colors.BLACK)
        self.screen.blit(score_text, (self.config.window_width - 150, 20))
    
    def draw_idle(self):
        """Draw idle state with instructions."""
        cx, cy = self.config.window_width // 2, self.config.window_height // 2
        
        # Draw crosshair at center
        pygame.draw.line(self.screen, Colors.LIGHT_GRAY,
                        (cx - 50, cy), (cx + 50, cy), 1)
        pygame.draw.line(self.screen, Colors.LIGHT_GRAY,
                        (cx, cy - 50), (cx, cy + 50), 1)
        
        # Draw instructions
        instructions = [
            "EOG Gaze Control Demo",
            "",
            "Controls:",
            "  SPACE - Start calibration",
            "  G - Toggle game mode",
            "  S - Toggle smoothing",
            "  R - Reset calibration",
            "  ESC - Quit",
        ]
        
        y = cy + 100
        for i, line in enumerate(instructions):
            if i == 0:
                text = self.font_large.render(line, True, Colors.ORANGE)
            else:
                text = self.font_small.render(line, True, Colors.DARK_GRAY)
            text_rect = text.get_rect(center=(cx, y))
            self.screen.blit(text, text_rect)
            y += 30 if i == 0 else 25
    
    def draw_cursor(self):
        """Draw the gaze cursor."""
        # Outer ring
        pygame.draw.circle(self.screen, Colors.DARK_ORANGE,
                          (self.cursor_x, self.cursor_y),
                          self.config.cursor_size)
        # Inner dot
        pygame.draw.circle(self.screen, Colors.ORANGE,
                          (self.cursor_x, self.cursor_y),
                          self.config.cursor_size - 5)
        # Center
        pygame.draw.circle(self.screen, Colors.WHITE,
                          (self.cursor_x, self.cursor_y), 3)
    
    def draw_status_bar(self):
        """Draw status bar at bottom."""
        bar_height = 30
        bar_y = self.config.window_height - bar_height
        
        pygame.draw.rect(self.screen, Colors.BLACK,
                        (0, bar_y, self.config.window_width, bar_height))
        
        # Status text
        status_parts = []
        
        if self.calibration.state == 'complete':
            status_parts.append("Calibrated")
        elif self.calibration.state != 'idle':
            status_parts.append("Calibrating...")
        else:
            status_parts.append("Not calibrated")
        
        if self.game.active:
            status_parts.append("Game ON")
        
        if self.processor.smoothing_enabled:
            status_parts.append("Smoothing ON")
        else:
            status_parts.append("Smoothing OFF")
        
        status_text = " | ".join(status_parts)
        text = self.font_small.render(status_text, True, Colors.WHITE)
        self.screen.blit(text, (10, bar_y + 5))
        
        # Position
        pos_text = f"Cursor: ({self.cursor_x}, {self.cursor_y})"
        text = self.font_small.render(pos_text, True, Colors.LIGHT_GRAY)
        self.screen.blit(text, (self.config.window_width - 200, bar_y + 5))
    
    def draw_blink_indicator(self):
        """Draw blink indicator."""
        pygame.draw.circle(self.screen, Colors.GREEN,
                          (self.config.window_width - 50, 50), 20)
        text = self.font_small.render("BLINK", True, Colors.WHITE)
        text_rect = text.get_rect(center=(self.config.window_width - 50, 50))
        self.screen.blit(text, text_rect)
    
    def cleanup(self):
        """Clean up resources."""
        self.data_source.close()
        pygame.quit()


# =============================================================================
# Main Entry Point
# =============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="EOG Gaze Control Demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python eog_gaze_control.py              # Use LSL stream
  python eog_gaze_control.py --port COM9  # Direct BioRadio (Windows)
  python eog_gaze_control.py --port /dev/tty.BioRadioAYA  # Direct (macOS)
  python eog_gaze_control.py --simulate   # Test with mouse
        """
    )
    
    parser.add_argument('--port', type=str, default=None,
                       help='Serial port for direct BioRadio connection')
    parser.add_argument('--simulate', action='store_true',
                       help='Simulate EOG with mouse (for testing)')
    parser.add_argument('--lsl-stream', type=str, default='BioRadio',
                       help='Name of LSL stream to connect to')
    parser.add_argument('--fullscreen', action='store_true',
                       help='Run in fullscreen mode')
    parser.add_argument('--width', type=int, default=1024,
                       help='Window width')
    parser.add_argument('--height', type=int, default=768,
                       help='Window height')
    
    args = parser.parse_args()
    
    # Check dependencies
    if not PYGAME_AVAILABLE:
        print("Error: pygame is required. Install with: pip install pygame")
        sys.exit(1)
    
    # Create configuration
    config = Config(
        window_width=args.width,
        window_height=args.height,
        fullscreen=args.fullscreen
    )
    
    # Create data source
    data_source = None

    if args.simulate:
        print("Running in simulation mode (use mouse to control)")
        data_source = SimulatedDataSource(config.window_width, config.window_height)
    elif args.port:
        data_source = BioRadioDataSource(args.port)
    else:
        if not LSL_AVAILABLE:
            print("Error: pylsl is required for LSL mode.")
            print("Install with: pip install pylsl")
            print("Or use --simulate for testing, or --port for direct connection")
            sys.exit(1)

        # Show stream browser UI
        pygame.init()
        if config.fullscreen:
            screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
            config.window_width = screen.get_width()
            config.window_height = screen.get_height()
        else:
            screen = pygame.display.set_mode(
                (config.window_width, config.window_height)
            )
        pygame.display.set_caption("EOG Gaze Control - Stream Browser")
        fonts = (
            pygame.font.Font(None, 48),
            pygame.font.Font(None, 32),
            pygame.font.Font(None, 24),
        )

        browser = StreamBrowser(screen, fonts)
        result = browser.run()

        if result is None:
            pygame.quit()
            print("Cancelled.")
            sys.exit(0)
        elif result == "SIMULATE":
            data_source = SimulatedDataSource(config.window_width, config.window_height)
        else:
            data_source = LSLDataSource(result)

        # Hand the already-created screen to GazeControlApp
        pygame.display.set_caption("EOG Gaze Control Demo")

    # Run application
    app = GazeControlApp(config, data_source)

    print("\n" + "="*50)
    print("EOG Gaze Control Demo")
    print("="*50)
    print("Press SPACE to start calibration")
    print("Press G to toggle game mode")
    print("Press S to toggle smoothing")
    print("Press ESC to quit")
    print("="*50 + "\n")

    try:
        app.run()
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    finally:
        app.cleanup()


if __name__ == "__main__":
    main()
