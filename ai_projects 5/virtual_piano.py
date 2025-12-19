"""
Virtual AI-Driven Piano
Uses computer vision (MediaPipe) for hand tracking and gesture-based piano control.
No physical piano needed - purely visual interaction.
"""

import cv2
import mediapipe as mp
import pygame
import numpy as np
import time
import math


class PianoKey:
    """Represents a virtual piano key with position, visuals, and sound."""
    
    def __init__(self, note_name, x, y, width, height, frequency, is_black=False):
        """
        Initialize a piano key.
        
        Args:
            note_name: Note name (C, D, E, etc.)
            x, y: Top-left corner coordinates
            width, height: Key dimensions
            frequency: Sound frequency in Hz
            is_black: True for black keys, False for white keys
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
        self.cooldown = 0.15  # Fast response for piano
        
        # Colors
        if is_black:
            self.color = (30, 30, 35)
            self.pressed_color = (80, 80, 120)
        else:
            self.color = (250, 250, 255)
            self.pressed_color = (200, 220, 255)
        
        # Generate piano sound
        self.sound = self._generate_piano_sound(frequency)
        self.sound.set_volume(1.0)
    
    def _generate_piano_sound(self, frequency):
        """Generate realistic piano sound using additive synthesis."""
        sample_rate = 44100
        duration = 2.0  # Piano notes sustain longer
        samples = int(sample_rate * duration)
        wave = np.zeros(samples)
        
        # Piano harmonics with specific amplitudes (based on piano timbre analysis)
        harmonics = [
            (1.0, 1.0),      # Fundamental
            (2.0, 0.7),      # 2nd harmonic
            (3.0, 0.5),      # 3rd harmonic
            (4.0, 0.3),      # 4th harmonic
            (5.0, 0.2),      # 5th harmonic
            (6.0, 0.15),     # 6th harmonic
            (7.0, 0.1),      # 7th harmonic
            (8.0, 0.08),     # 8th harmonic
        ]
        
        for i in range(samples):
            t = i / sample_rate
            
            # Piano has sharp attack and gradual decay
            attack_time = 0.02
            decay_time = 0.3
            sustain_level = 0.6
            release_time = 1.5
            
            if t < attack_time:
                # Sharp attack
                envelope = t / attack_time
            elif t < attack_time + decay_time:
                # Quick decay to sustain
                decay_progress = (t - attack_time) / decay_time
                envelope = 1.0 - (1.0 - sustain_level) * decay_progress
            elif t < duration - release_time:
                # Sustain
                envelope = sustain_level
            else:
                # Release
                release_progress = (t - (duration - release_time)) / release_time
                envelope = sustain_level * (1.0 - release_progress)
            
            # Add harmonics
            for harmonic_num, amplitude in harmonics:
                wave[i] += amplitude * np.sin(2 * np.pi * frequency * harmonic_num * t)
            
            wave[i] *= envelope
        
        # Add slight inharmonicity for realism (piano strings are slightly sharp)
        inharmonic = np.zeros(samples)
        for i in range(samples):
            t = i / sample_rate
            detune = 1.0002  # Slight detuning
            inharmonic[i] = 0.15 * np.sin(2 * np.pi * frequency * detune * t) * np.exp(-2 * t)
        
        wave += inharmonic
        
        # Normalize and soft clip
        wave = np.tanh(wave * 0.8)
        max_val = np.max(np.abs(wave))
        if max_val > 0:
            wave = wave / max_val
        
        # Convert to 16-bit stereo
        wave = np.int16(wave * 32767 * 0.7)
        stereo_wave = np.column_stack((wave, wave))
        
        return pygame.sndarray.make_sound(stereo_wave)
    
    def check_collision(self, x, y):
        """Check if a point intersects with this key."""
        return (self.x <= x <= self.x + self.width and 
                self.y <= y <= self.y + self.height)
    
    def press(self, intensity=1.0):
        """Play sound if cooldown period has elapsed."""
        current_time = time.time()
        if current_time - self.last_press_time >= self.cooldown:
            if self.sound:
                volume = max(0.4, min(1.0, intensity))
                self.sound.set_volume(volume)
                self.sound.play()
            self.is_pressed = True
            self.press_intensity = intensity
            self.last_press_time = current_time
            return True
        return False
    
    def draw(self, frame):
        """Draw realistic piano key with 3D effect."""
        color = self.pressed_color if self.is_pressed else self.color
        
        if self.is_black:
            # Black key - shorter, raised above white keys
            # Shadow
            cv2.rectangle(frame, (self.x + 3, self.y + 3), 
                         (self.x + self.width + 3, self.y + self.height + 3), 
                         (0, 0, 0), -1)
            
            # Main body with gradient
            for i in range(self.height):
                shade_factor = 1.0 - (i / self.height) * 0.3
                grad_color = tuple(int(c * shade_factor) for c in color)
                cv2.line(frame, (self.x, self.y + i), 
                        (self.x + self.width, self.y + i), grad_color, 1)
            
            # Glossy top
            cv2.rectangle(frame, (self.x + 2, self.y + 2), 
                         (self.x + self.width - 2, self.y + 15), 
                         (100, 100, 120), -1)
            
            # Border
            cv2.rectangle(frame, (self.x, self.y), 
                         (self.x + self.width, self.y + self.height), 
                         (0, 0, 0), 2)
        
        else:
            # White key - full height
            # Shadow at bottom
            cv2.rectangle(frame, (self.x + 2, self.y + self.height - 10), 
                         (self.x + self.width - 2, self.y + self.height), 
                         (200, 200, 200), -1)
            
            # Main body
            cv2.rectangle(frame, (self.x, self.y), 
                         (self.x + self.width, self.y + self.height), 
                         color, -1)
            
            # Ivory texture highlight
            cv2.rectangle(frame, (self.x + 3, self.y + 10), 
                         (self.x + self.width - 3, self.y + 30), 
                         (255, 255, 255), -1)
            
            # Side borders for 3D effect
            cv2.line(frame, (self.x, self.y), (self.x, self.y + self.height), 
                    (180, 180, 190), 2)
            cv2.line(frame, (self.x + self.width, self.y), 
                    (self.x + self.width, self.y + self.height), 
                    (180, 180, 190), 2)
            
            # Bottom border
            cv2.line(frame, (self.x, self.y + self.height), 
                    (self.x + self.width, self.y + self.height), 
                    (150, 150, 160), 3)
        
        # Note label at bottom
        if not self.is_black:
            text_size = cv2.getTextSize(self.note_name, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            text_x = self.x + (self.width - text_size[0]) // 2
            text_y = self.y + self.height - 10
            cv2.putText(frame, self.note_name, (text_x, text_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)
        
        # Glow when pressed
        if self.is_pressed:
            glow_overlay = frame.copy()
            cv2.rectangle(glow_overlay, (self.x, self.y), 
                         (self.x + self.width, self.y + self.height), 
                         (255, 255, 200), -1)
            cv2.addWeighted(glow_overlay, 0.3 * self.press_intensity, 
                           frame, 0.7, 0, frame)
        
        # Reset press state
        if self.is_pressed and time.time() - self.last_press_time > 0.1:
            self.is_pressed = False


class VirtualPiano:
    """Main application class for the virtual piano."""
    
    def __init__(self):
        """Initialize the virtual piano with camera, hand tracking, and keys."""
        # Initialize Pygame for audio
        pygame.mixer.pre_init(44100, -16, 2, 512)
        pygame.init()
        
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
        if ret:
            self.frame_height, self.frame_width = frame.shape[:2]
        else:
            self.frame_width, self.frame_height = 1280, 720
        
        # Create piano keys
        self.keys = self._create_piano_keys()
        
        # Track hand positions for velocity
        self.prev_hand_positions = {}
        
        print("Virtual Piano initialized!")
        print("Position your hands over the keys and tap to play.")
        print("Press 'q' to quit.")
    
    def _create_piano_keys(self):
        """Create 2 octaves of piano keys (25 keys: 15 white + 10 black)."""
        keys = []
        
        # Piano layout - 2 octaves starting from C
        white_notes = ['C3', 'D3', 'E3', 'F3', 'G3', 'A3', 'B3', 
                       'C4', 'D4', 'E4', 'F4', 'G4', 'A4', 'B4', 'C5']
        black_notes = ['C#3', 'D#3', 'F#3', 'G#3', 'A#3',
                       'C#4', 'D#4', 'F#4', 'G#4', 'A#4']
        
        # Frequencies (A4 = 440 Hz reference)
        note_frequencies = {
            'C3': 130.81, 'C#3': 138.59, 'D3': 146.83, 'D#3': 155.56,
            'E3': 164.81, 'F3': 174.61, 'F#3': 185.00, 'G3': 196.00,
            'G#3': 207.65, 'A3': 220.00, 'A#3': 233.08, 'B3': 246.94,
            'C4': 261.63, 'C#4': 277.18, 'D4': 293.66, 'D#4': 311.13,
            'E4': 329.63, 'F4': 349.23, 'F#4': 369.99, 'G4': 392.00,
            'G#4': 415.30, 'A4': 440.00, 'A#4': 466.16, 'B4': 493.88,
            'C5': 523.25
        }
        
        # Key dimensions
        white_key_width = self.frame_width // 16
        white_key_height = int(self.frame_height * 0.4)
        black_key_width = int(white_key_width * 0.6)
        black_key_height = int(white_key_height * 0.6)
        
        # Starting position (center bottom)
        start_x = (self.frame_width - (white_key_width * 15)) // 2
        white_key_y = self.frame_height - white_key_height - 50
        black_key_y = white_key_y
        
        # Create white keys first
        white_x = start_x
        for note in white_notes:
            keys.append(PianoKey(
                note, white_x, white_key_y,
                white_key_width, white_key_height,
                note_frequencies[note],
                is_black=False
            ))
            white_x += white_key_width
        
        # Create black keys (positioned between white keys)
        black_positions = [0, 1, 3, 4, 5, 7, 8, 10, 11, 12]  # Positions between white keys
        for i, note in enumerate(black_notes):
            black_x = start_x + (black_positions[i] * white_key_width) + \
                      (white_key_width - black_key_width // 2)
            keys.append(PianoKey(
                note, black_x, black_key_y,
                black_key_width, black_key_height,
                note_frequencies[note],
                is_black=True
            ))
        
        return keys
    
    def process_hand_landmarks(self, hand_landmarks, frame, hand_label):
        """Process hand landmarks and check for key presses."""
        # Use index fingertip for playing
        index_tip = hand_landmarks.landmark[8]
        index_mid = hand_landmarks.landmark[6]
        
        tip_x = int(index_tip.x * self.frame_width)
        tip_y = int(index_tip.y * self.frame_height)
        
        # Calculate velocity
        current_time = time.time()
        velocity = 0.7
        
        if hand_label in self.prev_hand_positions:
            prev_x, prev_y, prev_time = self.prev_hand_positions[hand_label]
            time_diff = current_time - prev_time
            
            if time_diff > 0:
                distance = math.sqrt((tip_x - prev_x)**2 + (tip_y - prev_y)**2)
                speed = distance / time_diff
                velocity = min(1.0, max(0.4, speed / 800))
        
        self.prev_hand_positions[hand_label] = (tip_x, tip_y, current_time)
        
        # Check if pointing down
        is_pointing_down = tip_y > int(index_mid.y * self.frame_height)
        
        # Hand color
        indicator_color = (255, 100, 255) if hand_label == "Right" else (100, 255, 255)
        
        # Draw fingertip
        cv2.circle(frame, (tip_x, tip_y), 15, indicator_color, -1)
        cv2.circle(frame, (tip_x, tip_y), 17, (255, 255, 255), 2)
        
        # Check collision with keys (black keys have priority)
        if is_pointing_down:
            # Check black keys first (they're on top)
            for key in self.keys:
                if key.is_black and key.check_collision(tip_x, tip_y):
                    key.press(velocity)
                    return
            
            # Then check white keys
            for key in self.keys:
                if not key.is_black and key.check_collision(tip_x, tip_y):
                    key.press(velocity)
                    return
    
    def run(self):
        """Main application loop."""
        while True:
            success, frame = self.cap.read()
            if not success:
                print("Failed to read from camera")
                break
            
            frame = cv2.flip(frame, 1)
            
            # Dark concert hall background
            overlay = np.zeros_like(frame)
            for y in range(self.frame_height):
                darkness = int(15 + (y / self.frame_height) * 20)
                overlay[y, :] = [darkness // 4, darkness // 3, darkness // 2]
            
            frame = cv2.addWeighted(frame, 0.3, overlay, 0.7, 0)
            
            # Convert to RGB for MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb_frame)
            
            # Draw piano keys (white keys first, then black keys on top)
            for key in self.keys:
                if not key.is_black:
                    key.draw(frame)
            
            for key in self.keys:
                if key.is_black:
                    key.draw(frame)
            
            # Process hands
            if results.multi_hand_landmarks:
                for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                    hand_label = "Right"
                    if results.multi_handedness:
                        hand_label = results.multi_handedness[idx].classification[0].label
                    
                    self.mp_draw.draw_landmarks(
                        frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS,
                        self.mp_draw.DrawingSpec(color=(0, 255, 150), thickness=2, circle_radius=3),
                        self.mp_draw.DrawingSpec(color=(255, 255, 255), thickness=2)
                    )
                    
                    self.process_hand_landmarks(hand_landmarks, frame, hand_label)
            
            # Instructions
            cv2.rectangle(frame, (0, 0), (self.frame_width, 80), (0, 0, 0), -1)
            cv2.putText(frame, "AI VIRTUAL PIANO - Tap Keys with Your Fingers!", 
                       (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
            cv2.putText(frame, "Press 'Q' to Quit", 
                       (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            cv2.imshow("Virtual Piano", frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
        
        self.cleanup()
    
    def cleanup(self):
        """Release resources."""
        self.cap.release()
        cv2.destroyAllWindows()
        pygame.quit()
        print("Virtual Piano closed.")


def main():
    """Entry point."""
    try:
        piano = VirtualPiano()
        piano.run()
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
