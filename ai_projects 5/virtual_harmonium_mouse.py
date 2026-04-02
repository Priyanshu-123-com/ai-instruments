"""
Virtual AI-Driven Harmonium with Mouse Support
Uses computer vision (MediaPipe) for hand tracking and gesture-based harmonium control.
Also includes mouse-clickable keys for interactive playing.
"""

import cv2
import mediapipe as mp
import pygame
import numpy as np
import time
import math


class HarmoniumKey:
    """Represents a virtual harmonium key with position, visuals, and sound."""
    
    def __init__(self, note_name, x, y, width, height, frequency, is_black=False):
        """
        Initialize a harmonium key.
        
        Args:
            note_name: Note name (Sa, Re, Ga, Ma, Pa, Dha, Ni, Sa)
            x, y: Top-left corner coordinates
            width, height: Key dimensions
            frequency: Sound frequency in Hz
            is_black: True for komal (flat) keys
        """
        self.note_name = note_name
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.frequency = frequency
        self.is_black = is_black
        self.is_pressed = False
        self.last_press_time = 0
        self.press_intensity = 0
        self.cooldown = 0.1
        self.is_playing = False
        
        # Colors
        if is_black:
            self.color = (80, 60, 120)  # Dark purple for komal notes
            self.pressed_color = (150, 100, 200)
        else:
            self.color = (220, 200, 240)  # Light purple for shuddha notes
            self.pressed_color = (255, 230, 255)
        
        # Generate harmonium sound
        self.sound = self._generate_harmonium_sound(frequency)
        self.sound.set_volume(0.0)  # Start silent, controlled by bellows
    
    def _generate_harmonium_sound(self, frequency):
        """Generate realistic harmonium sound using reed-like synthesis."""
        sample_rate = 44100
        duration = 5.0  # Continuous sound
        samples = int(sample_rate * duration)
        wave = np.zeros(samples)
        
        # Harmonium has rich harmonics with tremolo effect
        harmonics = [
            (1.0, 1.0),      # Fundamental
            (2.0, 0.8),      # 2nd harmonic
            (3.0, 0.6),      # 3rd harmonic
            (4.0, 0.4),      # 4th harmonic
            (5.0, 0.3),      # 5th harmonic
            (6.0, 0.2),      # 6th harmonic
        ]
        
        for i in range(samples):
            t = i / sample_rate
            
            # Tremolo effect (typical of harmonium)
            tremolo = 1.0 + 0.15 * np.sin(2 * np.pi * 4.5 * t)  # 4.5Hz tremolo
            
            # Vibrato effect
            vibrato = 0.5 * np.sin(2 * np.pi * 6.0 * t)  # 6Hz vibrato
            
            # Add harmonics with modulation
            for harmonic_num, amplitude in harmonics:
                mod_freq = frequency * harmonic_num * (1 + 0.001 * vibrato)
                wave[i] += amplitude * np.sin(2 * np.pi * mod_freq * t)
            
            wave[i] *= tremolo
        
        # Add breath noise for authenticity
        breath_noise = np.random.randn(samples) * 0.05  # Reduced noise level
        # Apply low-pass filter to breath noise
        for i in range(1, samples):
            breath_noise[i] = 0.9 * breath_noise[i-1] + 0.1 * breath_noise[i]
        
        wave += breath_noise
        
        # Normalize and add warmth
        wave = np.tanh(wave * 0.7)
        max_val = np.max(np.abs(wave))
        if max_val > 0:
            wave = wave / max_val
        
        # Convert to stereo
        wave = np.int16(wave * 32767 * 0.6)  # Reduced volume to prevent distortion
        stereo_wave = np.column_stack((wave, wave))
        
        return pygame.sndarray.make_sound(stereo_wave)

    def check_collision(self, x, y):
        """Check if a point intersects with this key."""
        return (self.x <= x <= self.x + self.width and 
                self.y <= y <= self.y + self.height)

    def press(self, intensity=1.0):
        """Activate key for playing."""
        current_time = time.time()
        if current_time - self.last_press_time >= self.cooldown:
            self.is_pressed = True
            self.press_intensity = intensity
            self.last_press_time = current_time
            return True
        return False

    def release(self):
        """Release key."""
        self.is_pressed = False

    def update_volume(self, bellows_pressure):
        """Update sound volume based on bellows pressure."""
        if self.is_pressed and self.sound:
            # Scale volume with bellows pressure and press intensity
            raw_volume = bellows_pressure * self.press_intensity
            # Apply logarithmic scaling to prevent distortion at high volumes
            scaled_volume = raw_volume * raw_volume  # Square for gentle curve
            # Clamp volume to prevent clipping
            volume = max(0.0, min(1.0, scaled_volume))
            self.sound.set_volume(volume)
            if not self.is_playing:
                self.sound.play(loops=-1)  # Loop indefinitely
                self.is_playing = True
        elif self.is_playing:
            self.sound.stop()
            self.is_playing = False

    def draw(self, frame):
        """Draw harmonium key with realistic appearance."""
        color = self.pressed_color if self.is_pressed else self.color
        
        # Shadow
        cv2.rectangle(frame, (int(self.x + 4), int(self.y + 4)), 
                     (int(self.x + self.width + 4), int(self.y + self.height + 4)), 
                     (0, 0, 0), -1)
        
        # Main key body with gradient for 3D effect
        for i in range(int(self.height)):
            y_pos = int(self.y + i)
            # Create gradient from top to bottom
            gradient_factor = 1.0 - (i / self.height) * 0.4
            grad_color = tuple(int(c * gradient_factor) for c in color)
            cv2.line(frame, (int(self.x), y_pos), 
                    (int(self.x + self.width), y_pos), grad_color, 1)
        
        # Highlight at top for 3D effect
        highlight_height = int(self.height * 0.15)
        for i in range(highlight_height):
            alpha = 1.0 - (i / highlight_height)
            highlight_color = tuple(min(255, int(c + 50 * alpha)) for c in color)
            cv2.line(frame, (int(self.x + 2), int(self.y + i)), 
                    (int(self.x + self.width - 2), int(self.y + i)), highlight_color, 1)
        
        # Beveled edges
        cv2.line(frame, (int(self.x), int(self.y)), (int(self.x + self.width), int(self.y)), 
                (150, 130, 180), 2)
        cv2.line(frame, (int(self.x), int(self.y + self.height)), 
                (int(self.x + self.width), int(self.y + self.height)), (80, 60, 120), 2)
        
        # Note label with better styling
        text_size = cv2.getTextSize(self.note_name, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
        text_x = int(self.x + (self.width - text_size[0]) // 2)
        text_y = int(self.y + self.height - 15)
        
        # Text background for better visibility
        cv2.rectangle(frame, 
                     (int(text_x - 8), int(text_y - text_size[1] - 5)),
                     (int(text_x + text_size[0] + 8), int(text_y + 5)),
                     (0, 0, 0), -1)
        
        cv2.putText(frame, self.note_name, (text_x, text_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (240, 220, 255), 2)
        
        # Glow effect when pressed
        if self.is_pressed:
            glow_overlay = frame.copy()
            cv2.rectangle(glow_overlay, (int(self.x), int(self.y)), 
                         (int(self.x + self.width), int(self.y + self.height)), 
                         (255, 240, 255), -1)
            cv2.addWeighted(glow_overlay, 0.3 * self.press_intensity, 
                           frame, 0.7, 0, frame)


class Bellows:
    """Represents the harmonium bellows for pressure control."""
    
    def __init__(self, x, y, width, height):
        """Initialize bellows control area."""
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.pressure = 0.0
        self.target_pressure = 0.0
        self.is_active = False
    
    def update_pressure(self, hand_x, hand_y):
        """Update bellows pressure based on hand position."""
        if (self.x <= hand_x <= self.x + self.width and 
            self.y <= hand_y <= self.y + self.height):
            # Pressure based on vertical position within bellows area
            relative_y = (hand_y - self.y) / self.height
            self.target_pressure = 1.0 - relative_y  # Top = high pressure, bottom = low
            self.is_active = True
            return True
        else:
            self.is_active = False
            return False
    
    def smooth_update(self):
        """Smoothly interpolate to target pressure."""
        if self.is_active:
            self.pressure += (self.target_pressure - self.pressure) * 0.1
        else:
            # Gradually decrease pressure when not active
            self.pressure *= 0.95
    
    def draw(self, frame):
        """Draw bellows control area with realistic accordion appearance."""
        # Outer frame
        cv2.rectangle(frame, (int(self.x), int(self.y)), 
                     (int(self.x + self.width), int(self.y + self.height)), 
                     (80, 60, 40), -1)
        cv2.rectangle(frame, (int(self.x), int(self.y)), 
                     (int(self.x + self.width), int(self.y + self.height)), 
                     (120, 90, 60), 3)
        
        # Accordion pleats
        pleat_count = 12
        pleat_width = self.width // (pleat_count * 2)
        for i in range(pleat_count):
            pleat_x = int(self.x + i * pleat_width * 2)
            # Draw pleat
            cv2.rectangle(frame, (pleat_x, int(self.y + 5)), 
                        (int(pleat_x + pleat_width), int(self.y + self.height - 5)), 
                        (60, 45, 30), -1)
            # Pleat shadow
            cv2.rectangle(frame, (int(pleat_x + 2), int(self.y + 7)), 
                        (int(pleat_x + pleat_width - 2), int(self.y + self.height - 7)), 
                        (40, 30, 20), -1)
        
        # Pressure indicator bar
        indicator_height = int(self.height * 0.8 * self.pressure)
        if indicator_height > 0:
            indicator_y = int(self.y + self.height * 0.9 - indicator_height)
            cv2.rectangle(frame, (int(self.x + 10), indicator_y), 
                         (int(self.x + self.width - 10), int(self.y + self.height * 0.9)), 
                         (200, 150, 100), -1)
            # Gradient for indicator
            for i in range(indicator_height):
                y_pos = int(indicator_y + i)
                alpha = 1.0 - (i / indicator_height) * 0.7
                color = (int(200 * alpha), int(150 * alpha), int(100 * alpha))
                cv2.line(frame, (int(self.x + 10), y_pos), 
                        (int(self.x + self.width - 10), y_pos), color, 1)
        
        # Label
        cv2.putText(frame, "BELLOWS", (int(self.x + 15), int(self.y + 40)), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 240, 200), 2)
        cv2.putText(frame, "Volume Control", (int(self.x + 10), int(self.y + 70)), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 240, 200), 1)
        
        # Air flow indicator
        if self.is_active:
            # Animated air particles
            for i in range(5):
                particle_x = int(self.x + 20 + (i * 15))
                particle_y = int(self.y + self.height - 30 + np.sin(time.time() * 5 + i) * 5)
                cv2.circle(frame, (particle_x, particle_y), 3, (220, 180, 140), -1)


class InteractiveKey:
    """Represents an interactive key for mouse control."""
    
    def __init__(self, x, y, width, height, note_name, frequency):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.note_name = note_name
        self.frequency = frequency
        self.is_pressed = False
        
        # Colors
        if 'k' in note_name:  # komal notes
            self.color = (80, 60, 120)  # Dark purple
            self.pressed_color = (150, 100, 200)
        else:
            self.color = (220, 200, 240)  # Light purple
            self.pressed_color = (255, 230, 255)

    def check_collision(self, pos):
        """Check if a position collides with this key."""
        x, y = pos
        return (self.x <= x <= self.x + self.width and 
                self.y <= y <= self.y + self.height)

    def press(self):
        """Press the key."""
        self.is_pressed = True

    def release(self):
        """Release the key."""
        self.is_pressed = False

    def update(self):
        """Update the key state."""
        pass

    def draw(self, surface):
        """Draw the key on the pygame surface."""
        color = self.pressed_color if self.is_pressed else self.color
        
        # Draw key with pressed effect
        if self.is_pressed:
            # Draw pressed key with darker color and shadow
            pygame.draw.rect(surface, (100, 70, 140), (self.x, self.y, self.width, self.height))
            pygame.draw.rect(surface, (60, 40, 80), (self.x, self.y, self.width, self.height), 2)
        else:
            # Draw normal key with gradient effect
            for i in range(self.height):
                # Create gradient from top to bottom
                gradient_factor = 1.0 - (i / self.height) * 0.3
                r = int(color[0] * gradient_factor)
                g = int(color[1] * gradient_factor)
                b = int(color[2] * gradient_factor)
                pygame.draw.line(surface, (r, g, b), (self.x, self.y + i), (self.x + self.width, self.y + i))
            
            # Draw key outline
            pygame.draw.rect(surface, (100, 70, 120), (self.x, self.y, self.width, self.height), 2)
        
        # Draw note name
        font = pygame.font.SysFont('Arial', 14)
        text = font.render(self.note_name, True, (0, 0, 0))
        text_rect = text.get_rect(center=(self.x + self.width // 2, self.y + self.height // 2))
        surface.blit(text, text_rect)


class VirtualHarmonium:
    """Main application class for the virtual harmonium."""
    
    def __init__(self):
        """Initialize the virtual harmonium."""
        # Initialize MediaPipe Hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6
        )
        self.mp_draw = mp.solutions.drawing_utils
        
        # Initialize camera
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        # Get frame dimensions
        ret, frame = self.cap.read()
        if not ret or frame is None:
            print("Warning: Could not read from camera, using default dimensions")
            self.frame_width, self.frame_height = 1280, 720
        else:
            self.frame_height, self.frame_width = frame.shape[:2]
        
        # Initialize pygame for mouse input
        pygame.mixer.pre_init(44100, -16, 2, 512)
        pygame.mixer.init()
        pygame.init()
        
        # Create harmonium keys (for sound generation only, not visual)
        self.keys = self._create_harmonium_keys()
        
        # Create interactive keys for mouse control
        self.interactive_keys = []
        self.setup_interactive_keys()
        
        # Create bellows control (LEFT side for volume control)
        self.bellows = Bellows(
            int(self.frame_width * 0.05), 
            int(self.frame_height * 0.2),
            int(self.frame_width * 0.2),
            int(self.frame_height * 0.6)
        )
        
        # Track hand positions
        self.prev_hand_positions = {}
        
        # Create pygame window for interactive keys
        self.pygame_window = pygame.display.set_mode((800, 200))
        pygame.display.set_caption("Virtual Harmonium - Interactive Keys")
        self.clock = pygame.time.Clock()
        
        print("Virtual Harmonium initialized!")
        print("Left hand: Control bellows (volume) | Right hand: Play notes")
        print("Click on keys in the pygame window to play with mouse")
        print("Press 'q' to quit.")

    def _create_harmonium_keys(self):
        """Create harmonium keys for sound generation (no visual representation)."""
        keys = []
        
        # Indian classical notes (Sa Re Ga Ma Pa Dha Ni Sa)
        # Using fundamental frequency of Sa = 261.63 Hz (middle C)
        sa_frequency = 261.63
        
        # Shuddha (natural) notes
        shuddha_notes = ['Sa', 'Re', 'Ga', 'Ma', 'Pa', 'Dha', 'Ni', 'Sa']
        shuddha_ratios = [1.0, 9/8, 5/4, 4/3, 3/2, 5/3, 15/8, 2.0]
        
        # Komal (flat) notes
        komal_notes = ['rek', 'gak', 'mak', 'dhak', 'nik']
        komal_ratios = [16/15, 6/5, 32/27, 8/5, 9/5]
        
        # Create shuddha (natural) keys (invisible, for sound only)
        for i, (note, ratio) in enumerate(zip(shuddha_notes, shuddha_ratios)):
            frequency = sa_frequency * ratio
            # Create invisible key objects for sound generation
            keys.append(HarmoniumKey(
                note, 
                0, 0, 1, 1,  # Invisible positioning
                frequency,
                is_black=False
            ))
        
        # Create komal (flat) keys (invisible, for sound only)
        for i, (note, ratio) in enumerate(zip(komal_notes, komal_ratios)):
            frequency = sa_frequency * ratio
            keys.append(HarmoniumKey(
                note,
                0, 0, 1, 1,  # Invisible positioning
                frequency,
                is_black=True
            ))
        
        return keys
    
    def setup_interactive_keys(self):
        """Setup interactive keys for mouse control."""
        # Calculate key dimensions based on pygame window size
        total_width = 700
        num_keys = len(self.keys)
        key_width = total_width // num_keys
        key_height = 150
        start_x = (800 - total_width) // 2
        start_y = 25
        
        for i, harmonium_key in enumerate(self.keys):
            interactive_key = InteractiveKey(
                start_x + i * key_width, start_y,
                key_width, key_height,
                harmonium_key.note_name, harmonium_key.frequency
            )
            self.interactive_keys.append(interactive_key)

    def handle_mouse_input(self, pos, event_type):
        """Handle mouse input for interactive keys."""
        for key in self.interactive_keys:
            if key.check_collision(pos):
                if event_type == 'press':
                    key.press()
                    # Find corresponding harmonium key and trigger it
                    for h_key in self.keys:
                        if h_key.note_name == key.note_name:
                            h_key.press()
                            h_key.update_volume(self.bellows.pressure)
                            break
                elif event_type == 'release':
                    key.release()
                    # Release corresponding harmonium key
                    for h_key in self.keys:
                        if h_key.note_name == key.note_name:
                            h_key.release()
                            break
                return True
        return False

    def draw_interactive_keys(self):
        """Draw interactive keys on the pygame surface."""
        self.pygame_window.fill((240, 240, 220))  # Light cream background
        
        # Draw harmonium body background
        pygame.draw.rect(self.pygame_window, (139, 69, 19), (50, 20, 700, 160))
        pygame.draw.rect(self.pygame_window, (101, 67, 33), (50, 20, 700, 160), 3)
        
        # Draw decorative elements
        pygame.draw.line(self.pygame_window, (180, 140, 100), (50, 20), (750, 20), 3)
        pygame.draw.line(self.pygame_window, (180, 140, 100), (50, 180), (750, 180), 3)
        
        # Draw sound holes
        for i in range(6):
            hole_x = 100 + i * 100
            pygame.draw.circle(self.pygame_window, (60, 40, 20), (hole_x, 100), 15)
            pygame.draw.circle(self.pygame_window, (40, 25, 15), (hole_x, 100), 10)
        
        # Draw bellows pressure indicator
        pygame.draw.rect(self.pygame_window, (100, 70, 40), (300, 10, 200, 15))
        pygame.draw.rect(self.pygame_window, (180, 140, 100), (300, 10, 200 * self.bellows.pressure, 15))
        font = pygame.font.SysFont('Arial', 12)
        text = font.render(f"Volume: {int(self.bellows.pressure * 100)}%", True, (255, 255, 255))
        self.pygame_window.blit(text, (375, 12))
        
        # Draw keys
        for key in self.interactive_keys:
            key.draw(self.pygame_window)
        
        # Draw title
        title_font = pygame.font.SysFont('Arial', 18, bold=True)
        title = title_font.render("Interactive Harmonium Keys", True, (80, 40, 20))
        self.pygame_window.blit(title, (400 - title.get_width() // 2, 5))
        
        pygame.display.flip()

    def update_interactive_keys(self):
        """Update interactive key states."""
        for key in self.interactive_keys:
            key.update()

    def process_hand_landmarks(self, hand_landmarks, frame, hand_label):
        """Process hand landmarks for note playing and bellows control."""
        # Use index fingertip
        index_tip = hand_landmarks.landmark[8]
        tip_x = int(index_tip.x * self.frame_width)
        tip_y = int(index_tip.y * self.frame_height)
        
        # Calculate velocity for intensity
        current_time = time.time()
        velocity = 0.8
        
        if hand_label in self.prev_hand_positions:
            prev_x, prev_y, prev_time = self.prev_hand_positions[hand_label]
            time_diff = current_time - prev_time
            
            if time_diff > 0:
                distance = math.sqrt((tip_x - prev_x)**2 + (tip_y - prev_y)**2)
                speed = distance / time_diff
                velocity = min(1.0, max(0.5, speed / 800))
        
        self.prev_hand_positions[hand_label] = (tip_x, tip_y, current_time)
        
        # Hand color
        indicator_color = (255, 150, 100) if hand_label == "Left" else (100, 200, 255)
        
        # Draw fingertip
        cv2.circle(frame, (tip_x, tip_y), 15, indicator_color, -1)
        cv2.circle(frame, (tip_x, tip_y), 17, (255, 255, 255), 2)
        
        # Assign roles: left hand for bellows, right hand for notes
        if hand_label == "Left":
            # Left hand controls bellows (volume)
            self.bellows.update_pressure(tip_x, tip_y)
        else:
            # Right hand plays notes (map hand position to notes)
            # Map horizontal position to note selection
            relative_x = tip_x / self.frame_width
            note_index = int(relative_x * len(self.keys))
            if 0 <= note_index < len(self.keys):
                self.keys[note_index].press(velocity)

    def run(self):
        """Main application loop."""
        running = True
        while running:
            success, frame = self.cap.read()
            if not success:
                print("Failed to read from camera")
                break
            
            frame = cv2.flip(frame, 1)
            
            # Warm stage background (harmonium ambiance)
            overlay = np.zeros_like(frame)
            for y in range(self.frame_height):
                darkness = int(30 + (y / self.frame_height) * 25)
                overlay[y, :] = [darkness // 2, darkness // 3, darkness // 2]
            
            frame = cv2.addWeighted(frame, 0.4, overlay, 0.6, 0)
            
            # Convert to RGB for MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb_frame)
            
            # Update bellows pressure
            self.bellows.smooth_update()
            
            # Update key volumes based on bellows pressure
            for key in self.keys:
                key.update_volume(self.bellows.pressure)
            
            # Draw harmonium components
            # Draw bellows
            self.bellows.draw(frame)
            
            # Draw harmonium body with realistic wood texture
            body_x1 = int(self.frame_width * 0.25)
            body_y1 = int(self.frame_height * 0.3)
            body_x2 = int(self.frame_width * 0.95)
            body_y2 = int(self.frame_height * 0.8)
            
            # Main body
            cv2.rectangle(frame, (body_x1, body_y1), (body_x2, body_y2), (100, 70, 40), -1)
            cv2.rectangle(frame, (body_x1, body_y1), (body_x2, body_y2), (140, 100, 60), 3)
            
            # Wood grain effect
            for i in range(25):
                wood_shade = 90 + i * 2
                cv2.rectangle(frame, 
                             (int(body_x1 + i * 2), int(body_y1 + i * 1.2)),
                             (int(body_x2 - i * 2), int(body_y2 - i * 1.2)),
                             (wood_shade // 2, wood_shade // 3, wood_shade // 2),
                             1)
            
            # Draw harmonium keys on the body
            key_area_x1 = int(body_x1 + 50)
            key_area_y1 = int(body_y1 + 80)
            key_area_x2 = int(body_x2 - 50)
            key_area_y2 = int(body_y2 - 50)
            
            # Draw key area background
            cv2.rectangle(frame, (key_area_x1, key_area_y1), (key_area_x2, key_area_y2), (180, 160, 140), -1)
            cv2.rectangle(frame, (key_area_x1, key_area_y1), (key_area_x2, key_area_y2), (120, 100, 80), 2)
            
            # Draw individual keys
            num_keys = len(self.keys)
            key_width = int((key_area_x2 - key_area_x1 - 20) / num_keys)
            key_height = int(key_area_y2 - key_area_y1 - 20)
            
            for i in range(num_keys):
                key_x = int(key_area_x1 + 10 + i * key_width)
                key_y = int(key_area_y1 + 10)
                
                # Alternate key colors for visual distinction
                if i % 2 == 0:
                    # Light key
                    cv2.rectangle(frame, (key_x, key_y), (key_x + key_width - 2, key_y + key_height), (240, 230, 220), -1)
                    cv2.rectangle(frame, (key_x, key_y), (key_x + key_width - 2, key_y + key_height), (180, 170, 160), 1)
                else:
                    # Dark key
                    cv2.rectangle(frame, (key_x, key_y), (key_x + key_width - 2, key_y + key_height), (200, 180, 160), -1)
                    cv2.rectangle(frame, (key_x, key_y), (key_x + key_width - 2, key_y + key_height), (140, 120, 100), 1)
                
                # Key label
                if i < len(self.keys):
                    note_name = self.keys[i].note_name
                    text_size = cv2.getTextSize(note_name, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
                    text_x = int(key_x + (key_width - text_size[0]) / 2)
                    text_y = int(key_y + key_height / 2 + text_size[1] / 2)
                    cv2.putText(frame, note_name, (text_x, text_y), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (80, 60, 40), 1)
            
            # Decorative elements
            # Sound holes
            hole_center_x = (body_x1 + body_x2) // 2
            hole_center_y = (body_y1 + body_y2) // 2
            for i in range(12):
                angle = (2 * np.pi * i) / 12
                hole_x = int(hole_center_x + 60 * np.cos(angle))
                hole_y = int(hole_center_y + 40 * np.sin(angle))
                cv2.circle(frame, (hole_x, hole_y), 12, (60, 40, 25), -1)
                cv2.circle(frame, (hole_x, hole_y), 8, (40, 25, 15), -1)
            
            # Brand name
            brand_text = "VIRTUAL HARMONIUM"
            text_size = cv2.getTextSize(brand_text, cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 1.5, 2)[0]
            text_x = int((body_x1 + body_x2 - text_size[0]) // 2)
            text_y = int(body_y1 + 50)
            cv2.putText(frame, brand_text, (text_x, text_y), 
                       cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 1.5, (220, 190, 160), 2)
            
            # Note selector visualization (shows which note region hand is in)
            if results.multi_hand_landmarks:
                for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                    hand_label = "Right"
                    if results.multi_handedness:
                        hand_label = results.multi_handedness[idx].classification[0].label
                    
                    if hand_label == "Right":
                        # Show note selector for right hand
                        index_tip = hand_landmarks.landmark[8]
                        tip_x = int(index_tip.x * self.frame_width)
                        
                        # Map position to note
                        relative_x = tip_x / self.frame_width
                        note_index = int(relative_x * len(self.keys))
                        if 0 <= note_index < len(self.keys):
                            # Visualize note selection
                            selector_width = int((body_x2 - body_x1) / len(self.keys))
                            selector_x = body_x1 + note_index * selector_width
                            cv2.rectangle(frame, (selector_x, body_y1), 
                                        (selector_x + selector_width, body_y1 + 20), 
                                        (255, 255, 200), -1)
                            
                            # Show note name
                            note_names = [key.note_name for key in self.keys]
                            if note_index < len(note_names):
                                cv2.putText(frame, note_names[note_index], 
                                          (selector_x + 10, body_y1 + 15), 
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            
            # Process hands
            if results.multi_hand_landmarks:
                for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                    hand_label = "Right"
                    if results.multi_handedness:
                        hand_label = results.multi_handedness[idx].classification[0].label
                    
                    self.mp_draw.draw_landmarks(
                        frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS,
                        self.mp_draw.DrawingSpec(color=(0, 200, 255), thickness=2, circle_radius=3),
                        self.mp_draw.DrawingSpec(color=(255, 255, 255), thickness=2)
                    )
                    
                    self.process_hand_landmarks(hand_landmarks, frame, hand_label)
            
            # Instructions
            cv2.rectangle(frame, (0, 0), (int(self.frame_width), 100), (0, 0, 0), -1)
            cv2.putText(frame, "AI VIRTUAL HARMONIUM - Indian Classical Instrument", 
                       (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 200, 150), 2)
            cv2.putText(frame, "Left Hand: Bellows Control (Volume) | Right Hand: Note Selection", 
                       (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 255, 200), 2)
            cv2.putText(frame, "Click on keys in pygame window to play with mouse | Press 'Q' to Quit", 
                       (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            cv2.imshow("Virtual Harmonium", frame)
            
            # Handle pygame events for mouse interaction
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:  # Left mouse button
                        self.handle_mouse_input(event.pos, 'press')
                elif event.type == pygame.MOUSEBUTTONUP:
                    if event.button == 1:  # Left mouse button
                        self.handle_mouse_input((0, 0), 'release')  # Position doesn't matter for release
            
            # Update pygame window
            self.draw_interactive_keys()
            
            # Check for 'q' key press
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            
            # Update interactive keys
            self.update_interactive_keys()
            self.clock.tick(60)
        
        self.cleanup()

    def cleanup(self):
        """Release resources."""
        # Stop all sounds
        for key in self.keys:
            if key.is_playing:
                key.sound.stop()
        
        self.cap.release()
        cv2.destroyAllWindows()
        pygame.quit()
        print("Virtual Harmonium closed.")


def main():
    """Entry point."""
    try:
        harmonium = VirtualHarmonium()
        harmonium.run()
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()