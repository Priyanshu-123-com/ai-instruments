"""
Virtual AI-Driven Harmonium
Indian harmonium simulation with traditional keyboard and bellows control.
Uses computer vision (MediaPipe) for hand tracking and gesture-based harmonium playing.
"""

import cv2
import mediapipe as mp
import pygame
import numpy as np
import time
import math


class HarmoniumKey:
    """Represents a virtual harmonium key."""

    def __init__(self, note, x, y, width, height, frequency, color, is_black=False):
        """
        Initialize a harmonium key.

        Args:
            note: Musical note name (e.g., 'C4', 'D#4')
            x, y: Top-left coordinates
            width, height: Key dimensions
            frequency: Note frequency in Hz
            color: RGB tuple for visual representation
            is_black: Whether this is a black key (sharp/flat)
        """
        self.note = note
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.frequency = frequency
        self.color = color
        self.is_black = is_black
        self.is_pressed = False
        self.last_press_time = 0
        self.press_intensity = 0

        # Generate harmonium sound
        self.sound = self._generate_harmonium_sound(frequency)

    def _generate_harmonium_sound(self, base_freq):
        """Generate harmonium sound using reed organ synthesis."""
        sample_rate = 44100
        duration = 2.0  # Harmonium notes sustain longer
        samples = int(sample_rate * duration)

        # Create time array
        t = np.linspace(0, duration, samples, False)

        # Harmonium sound characteristics:
        # - Rich harmonics from reed vibration
        # - Slight vibrato and amplitude modulation
        # - Natural decay envelope

        # Fundamental and harmonics
        sound = np.zeros(samples)
        for harmonic in range(1, 8):  # 7 harmonics
            freq = base_freq * harmonic
            if freq > sample_rate / 2:  # Nyquist limit
                break

            # Amplitude decreases with harmonic number
            amplitude = 1.0 / harmonic

            # Add slight detuning for richness
            detune = 1 + 0.001 * harmonic * np.sin(2 * np.pi * 0.5 * t)

            # Generate harmonic
            harmonic_wave = amplitude * np.sin(2 * np.pi * freq * detune * t)

            # Add some noise for reed character
            noise = 0.01 * np.random.randn(samples)
            harmonic_wave += noise

            sound += harmonic_wave

        # Apply envelope (slow attack, sustain, slow decay)
        attack_time = 0.1
        decay_time = 0.3
        sustain_level = 0.7

        attack_samples = int(attack_time * sample_rate)
        decay_samples = int(decay_time * sample_rate)

        envelope = np.ones(samples)
        envelope[:attack_samples] = np.linspace(0, 1, attack_samples)
        envelope[-decay_samples:] = np.linspace(sustain_level, 0, decay_samples)
        envelope[attack_samples:-decay_samples] = sustain_level

        sound *= envelope

        # Convert to 16-bit PCM stereo
        sound = np.int16(sound * 32767)
        # Duplicate for stereo
        stereo_sound = np.column_stack((sound, sound))

        # Create pygame sound
        pygame_sound = pygame.sndarray.make_sound(stereo_sound)
        return pygame_sound

    def press(self, intensity=1.0):
        """Press the key and play sound."""
        current_time = time.time()
        if not self.is_pressed:
            volume = max(0.3, min(1.0, intensity))
            self.sound.set_volume(volume)
            self.sound.play(-1)  # Loop the sound
            self.is_pressed = True
            self.press_intensity = intensity
            self.last_press_time = current_time

    def release(self):
        """Release the key and stop sound."""
        if self.is_pressed:
            self.sound.stop()
            self.is_pressed = False

    def draw(self, frame):
        """Draw the harmonium key."""
        # Determine current color based on press state
        if self.is_pressed:
            time_since_press = time.time() - self.last_press_time
            if time_since_press < 0.2:
                # Brighten when pressed
                current_color = tuple(min(c + 80, 255) for c in self.color)
            else:
                current_color = self.color
        else:
            current_color = self.color

        # Draw key rectangle
        cv2.rectangle(frame, (self.x, self.y), (self.x + self.width, self.y + self.height),
                     current_color, -1)

        # Draw key border
        border_color = (255, 255, 255) if not self.is_black else (100, 100, 100)
        cv2.rectangle(frame, (self.x, self.y), (self.x + self.width, self.y + self.height),
                     border_color, 2)

        # Draw note name
        font_scale = 0.4 if len(self.note) <= 2 else 0.3
        text_color = (255, 255, 255) if self.is_black else (0, 0, 0)
        text_x = self.x + self.width // 2 - 10
        text_y = self.y + self.height // 2 + 5
        cv2.putText(frame, self.note, (text_x, text_y),
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, 1)


class HarmoniumBellows:
    """Represents the harmonium bellows for air control."""

    def __init__(self, x, y, width, height):
        """Initialize the bellows."""
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.expansion = 0.5  # 0 to 1 (compressed to expanded)
        self.last_pump_time = 0

    def pump(self, intensity=1.0):
        """Pump the bellows."""
        self.expansion = min(1.0, self.expansion + intensity * 0.1)
        self.last_pump_time = time.time()

    def update(self):
        """Update bellows state (natural compression over time)."""
        if time.time() - self.last_pump_time > 0.1:
            self.expansion = max(0.3, self.expansion - 0.005)

    def draw(self, frame):
        """Draw the bellows."""
        # Draw bellows body
        body_color = (139, 69, 19)  # Brown wood color
        cv2.rectangle(frame, (self.x, self.y), (self.x + self.width, self.y + self.height),
                     body_color, -1)

        # Draw expansion indicator
        expansion_height = int(self.height * self.expansion)
        expansion_color = (160, 82, 45)  # Lighter brown when expanded
        cv2.rectangle(frame, (self.x + 5, self.y + self.height - expansion_height),
                     (self.x + self.width - 5, self.y + self.height),
                     expansion_color, -1)

        # Draw bellows folds
        fold_color = (101, 67, 33)
        for i in range(1, 4):
            fold_y = self.y + (self.height * i // 4)
            cv2.line(frame, (self.x, fold_y), (self.x + self.width, fold_y), fold_color, 2)

        # Draw label
        cv2.putText(frame, "BELLOWS", (self.x + 10, self.y - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)


class VirtualHarmonium:
    """Main application class for the virtual harmonium."""

    def __init__(self):
        """Initialize the virtual harmonium."""
        # Initialize Pygame
        pygame.mixer.pre_init(44100, -16, 2, 512)
        pygame.init()

        # Initialize MediaPipe Hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils

        # Initialize camera
        self.cap = cv2.VideoCapture(0)
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Create harmonium keyboard
        self.keys = self._create_keyboard()
        self.bellows = HarmoniumBellows(50, self.frame_height - 150, 100, 100)

        # Gesture tracking
        self.gesture_cooldown = {}
        self.active_keys = set()

    def _create_keyboard(self):
        """Create the harmonium keyboard layout."""
        keys = []

        # White keys (natural notes)
        white_key_width = 60
        white_key_height = 200
        black_key_width = 40
        black_key_height = 120

        # Starting position
        start_x = 200
        start_y = self.frame_height - 250

        # Note frequencies (Indian harmonium range: C4 to C6)
        note_frequencies = {
            'C4': 261.63, 'D4': 293.66, 'E4': 329.63, 'F4': 349.23,
            'G4': 392.00, 'A4': 440.00, 'B4': 493.88, 'C5': 523.25,
            'D5': 587.33, 'E5': 659.25, 'F5': 698.46, 'G5': 783.99,
            'A5': 880.00, 'B5': 987.77, 'C6': 1046.50
        }

        # White keys positions
        white_positions = []
        for i, note in enumerate(['C4', 'D4', 'E4', 'F4', 'G4', 'A4', 'B4', 'C5', 'D5', 'E5', 'F5', 'G5', 'A5', 'B5', 'C6']):
            x = start_x + i * white_key_width
            white_positions.append((note, x))

        # Create white keys
        for note, x in white_positions:
            key = HarmoniumKey(note, x, start_y, white_key_width, white_key_height,
                              note_frequencies[note], (255, 255, 255), False)
            keys.append(key)

        # Black keys (sharps/flats)
        black_positions = [
            ('C#4', start_x + white_key_width - black_key_width//2),
            ('D#4', start_x + 2*white_key_width - black_key_width//2),
            ('F#4', start_x + 4*white_key_width - black_key_width//2),
            ('G#4', start_x + 5*white_key_width - black_key_width//2),
            ('A#4', start_x + 6*white_key_width - black_key_width//2),
            ('C#5', start_x + 8*white_key_width - black_key_width//2),
            ('D#5', start_x + 9*white_key_width - black_key_width//2),
            ('F#5', start_x + 11*white_key_width - black_key_width//2),
            ('G#5', start_x + 12*white_key_width - black_key_width//2),
            ('A#5', start_x + 13*white_key_width - black_key_width//2),
        ]

        # Create black keys
        for note, x in black_positions:
            if note in note_frequencies:
                key = HarmoniumKey(note, x, start_y, black_key_width, black_key_height,
                                  note_frequencies[note], (50, 50, 50), True)
                keys.append(key)

        return keys

    def draw_harmonium(self, frame):
        """Draw the virtual harmonium interface."""
        # Draw title
        cv2.putText(frame, "VIRTUAL HARMONIUM", (int(self.frame_width * 0.35), 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)

        # Draw keyboard
        for key in self.keys:
            key.draw(frame)

        # Draw bellows
        self.bellows.draw(frame)

        # Draw instructions
        instructions = [
            "Hand Gestures to Play Harmonium:",
            "Point finger at key: Press note",
            "Open palm over bellows: Pump air",
            "Make fist: Release all notes",
            "Press 'Q' to quit"
        ]

        for i, instruction in enumerate(instructions):
            y_pos = self.frame_height - 120 + i * 20
            cv2.putText(frame, instruction, (20, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        # Show active notes
        if self.active_keys:
            active_text = "Playing: " + ", ".join(sorted(self.active_keys))
            cv2.putText(frame, active_text, (int(self.frame_width * 0.6), 80),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    def process_hand_landmarks(self, hand_landmarks, frame, hand_label):
        """Process hand gestures to control harmonium."""
        # Get key landmark positions
        index_finger = hand_landmarks.landmark[8]
        palm_center = hand_landmarks.landmark[9]

        index_x = int(index_finger.x * self.frame_width)
        index_y = int(index_finger.y * self.frame_height)
        palm_x = int(palm_center.x * self.frame_width)
        palm_y = int(palm_center.y * self.frame_height)

        # Get fingertip positions for gesture detection
        fingertips = []
        for tip_idx in [4, 8, 12, 16, 20]:
            tip = hand_landmarks.landmark[tip_idx]
            x = int(tip.x * self.frame_width)
            y = int(tip.y * self.frame_height)
            fingertips.append((x, y))

        # Detect gesture
        gesture = self._detect_gesture(fingertips, palm_x, palm_y)

        # Draw hand landmarks
        hand_color = (0, 255, 0) if hand_label == "Right" else (255, 0, 255)
        for x, y in fingertips:
            cv2.circle(frame, (x, y), 8, hand_color, -1)

        cv2.circle(frame, (palm_x, palm_y), 12, hand_color, -1)
        cv2.circle(frame, (palm_x, palm_y), 15, (255, 255, 255), 2)

        # Process gestures
        self._process_gesture(gesture, index_x, index_y, palm_x, palm_y, hand_label)

    def _detect_gesture(self, fingertips, palm_x, palm_y):
        """Detect hand gesture."""
        # Calculate distances from palm to fingertips
        distances = []
        for tip_x, tip_y in fingertips:
            distance = math.sqrt((tip_x - palm_x)**2 + (tip_y - palm_y)**2)
            distances.append(distance)

        # Index finger pointing (index extended, others curled)
        avg_other_distance = sum(distances[1:]) / len(distances[1:])  # Exclude thumb
        if distances[1] > avg_other_distance * 0.8 and all(d < avg_other_distance * 0.6 for d in distances[2:]):
            return "pointing"

        # Open palm (all fingers extended)
        if all(d > avg_other_distance * 0.7 for d in distances):
            return "open_palm"

        # Closed fist (all fingers curled)
        if all(d < avg_other_distance * 0.5 for d in distances):
            return "closed_fist"

        return "unknown"

    def _process_gesture(self, gesture, index_x, index_y, palm_x, palm_y, hand_label):
        """Process detected gesture."""
        current_time = time.time()

        # Check if pointing at a key
        if gesture == "pointing":
            for key in self.keys:
                if (key.x <= index_x <= key.x + key.width and
                    key.y <= index_y <= key.y + key.height):
                    # Press the key
                    if key.note not in self.active_keys:
                        key.press(0.8)
                        self.active_keys.add(key.note)
                    break

        # Check if open palm over bellows
        elif gesture == "open_palm":
            if (self.bellows.x <= palm_x <= self.bellows.x + self.bellows.width and
                self.bellows.y <= palm_y <= self.bellows.y + self.bellows.height):
                self.bellows.pump(0.8)

        # Release all keys on fist
        elif gesture == "closed_fist":
            gesture_key = f"{hand_label}_fist"
            if gesture_key not in self.gesture_cooldown or current_time - self.gesture_cooldown[gesture_key] > 1.0:
                for key in self.keys:
                    key.release()
                self.active_keys.clear()
                self.gesture_cooldown[gesture_key] = current_time

    def run(self):
        """Main application loop."""
        prev_time = time.time()

        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break

            # Flip frame horizontally for mirror effect
            frame = cv2.flip(frame, 1)

            # Convert to RGB for MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb_frame)

            # Update bellows
            self.bellows.update()

            # Draw harmonium
            self.draw_harmonium(frame)

            # Process hand landmarks
            if results.multi_hand_landmarks:
                for hand_landmarks, handedness in zip(results.multi_hand_landmarks,
                                                     results.multi_handedness):
                    # Draw hand landmarks
                    self.mp_draw.draw_landmarks(frame, hand_landmarks,
                                              self.mp_hands.HAND_CONNECTIONS)

                    # Process for harmonium playing
                    hand_label = handedness.classification[0].label
                    self.process_hand_landmarks(hand_landmarks, frame, hand_label)

            # Calculate and display FPS
            current_time = time.time()
            fps = 1 / (current_time - prev_time)
            prev_time = current_time
            cv2.putText(frame, f"FPS: {int(fps)}", (self.frame_width - 100, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # Show frame
            cv2.imshow("Virtual Harmonium", frame)

            # Check for quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Cleanup
        self.cap.release()
        cv2.destroyAllWindows()
        pygame.quit()


def main():
    """Main function."""
    harmonium = VirtualHarmonium()
    harmonium.run()


if __name__ == "__main__":
    main()