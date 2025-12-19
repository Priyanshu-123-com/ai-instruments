"""
Virtual AI-Driven Tabla
Authentic Indian tabla simulation with traditional playing techniques.
Uses computer vision (MediaPipe) for hand tracking and gesture-based tabla control.
"""

import cv2
import mediapipe as mp
import pygame
import numpy as np
import time
import math


class TablaDrum:
    """Represents a virtual tabla drum (dayan or bayan) with authentic playing techniques."""

    def __init__(self, name, x, y, radius, frequency, color, drum_type='dayan'):
        """
        Initialize a tabla drum.

        Args:
            name: Display name of the drum
            x, y: Center coordinates
            radius: Drum radius
            frequency: Base frequency for sound generation
            color: RGB tuple for visual representation
            drum_type: 'dayan' (right, treble) or 'bayan' (left, bass)
        """
        self.name = name
        self.x = x
        self.y = y
        self.radius = radius
        self.frequency = frequency
        self.color = color
        self.drum_type = drum_type
        self.hit_color = tuple(min(c + 100, 255) for c in color)  # Brighter on hit
        self.is_hit = False
        self.last_hit_time = 0
        self.hit_intensity = 0
        self.cooldown = 0.15  # Fast response for tabla

        # Generate tabla sounds
        self.sounds = self._generate_tabla_sounds(frequency)

    def _generate_tabla_sounds(self, base_freq):
        """Generate authentic tabla sounds using physical modeling."""
        sample_rate = 44100
        duration = 0.8  # Tabla sounds are short
        samples = int(sample_rate * duration)

        sounds = {}

        # Authentic tabla strokes
        if self.drum_type == 'dayan':
            # Dayan strokes - treble, metallic sounds
            strokes = {
                'ta': {'freq': base_freq * 1.8, 'decay': 0.25, 'harmonics': [1, 2.1, 3.2, 4.8, 6.1], 'metallic': 0.3},
                'na': {'freq': base_freq * 2.2, 'decay': 0.2, 'harmonics': [1, 2.8, 4.1, 5.9, 7.2], 'metallic': 0.4},
                'dha': {'freq': base_freq * 1.5, 'decay': 0.35, 'harmonics': [1, 1.9, 2.7, 3.8, 5.2], 'metallic': 0.2},
                'tin': {'freq': base_freq * 2.8, 'decay': 0.15, 'harmonics': [1, 3.1, 4.9, 6.8, 8.3], 'metallic': 0.5},
                'ka': {'freq': base_freq * 1.2, 'decay': 0.4, 'harmonics': [1, 1.7, 2.3, 3.1, 4.2], 'metallic': 0.1}
            }
        else:  # bayan
            # Bayan strokes - bass, resonant sounds
            strokes = {
                'ga': {'freq': base_freq * 0.7, 'decay': 0.6, 'harmonics': [1, 1.4, 2.1, 2.9, 3.8], 'metallic': 0.05},
                'ka': {'freq': base_freq * 0.5, 'decay': 0.8, 'harmonics': [1, 1.2, 1.8, 2.4, 3.1], 'metallic': 0.03},
                'dha': {'freq': base_freq * 0.9, 'decay': 0.5, 'harmonics': [1, 1.6, 2.3, 3.2, 4.1], 'metallic': 0.08},
                'na': {'freq': base_freq * 0.6, 'decay': 0.7, 'harmonics': [1, 1.3, 1.9, 2.6, 3.4], 'metallic': 0.04}
            }

        for stroke_name, params in strokes.items():
            wave = np.zeros(samples)

            for i in range(samples):
                t = i / sample_rate

                # Complex tabla envelope
                if t < 0.02:
                    envelope = t / 0.02  # Sharp attack
                elif t < 0.08:
                    envelope = 1.0 + 0.3 * np.sin(50 * t)  # Initial resonance
                else:
                    envelope = np.exp(-params['decay'] * (t - 0.08))

                # Generate harmonics with tabla characteristics
                base_wave = 0
                for harmonic in params['harmonics']:
                    freq = params['freq'] * harmonic
                    base_wave += np.sin(2 * np.pi * freq * t) / len(params['harmonics'])

                # Add metallic ringing for dayan
                if params['metallic'] > 0:
                    metallic = params['metallic'] * np.sin(2 * np.pi * params['freq'] * 8 * t)
                    metallic *= np.exp(-2 * t)  # Quick decay
                    base_wave += metallic

                wave[i] = base_wave * envelope

            # Add tabla-specific noise (membrane slap and air)
            membrane_noise = np.random.randn(int(sample_rate * 0.03)) * 0.2
            membrane_envelope = np.exp(-np.linspace(0, 4, len(membrane_noise)))
            wave[:len(membrane_noise)] += membrane_noise * membrane_envelope

            # Normalize
            max_val = np.max(np.abs(wave))
            if max_val > 0:
                wave = wave / max_val

            # Convert to audio
            wave = np.int16(wave * 32767 * 0.8)
            stereo_wave = np.column_stack((wave, wave))
            sounds[stroke_name] = pygame.sndarray.make_sound(stereo_wave)

        return sounds

    def check_strike(self, finger_positions, palm_position, hand_label):
        """Check for authentic tabla playing techniques."""
        strikes = []

        if self.drum_type == 'dayan' and hand_label == "Right":
            # Dayan played with right hand fingers
            for i, (finger_x, finger_y) in enumerate(finger_positions):
                distance = math.sqrt((finger_x - self.x)**2 + (finger_y - self.y)**2)
                if distance <= self.radius + 20:  # Allow some tolerance
                    # Determine which finger and stroke
                    if i == 0:  # Index finger
                        strikes.append(('ta', 0.9))
                    elif i == 1:  # Middle finger
                        strikes.append(('na', 0.85))
                    elif i == 2:  # Ring finger
                        strikes.append(('dha', 0.8))
                    elif i == 3:  # Pinky
                        strikes.append(('tin', 0.75))

        elif self.drum_type == 'bayan' and hand_label == "Left":
            # Bayan played with left palm heel
            if palm_position:
                palm_x, palm_y = palm_position
                distance = math.sqrt((palm_x - self.x)**2 + (palm_y - self.y)**2)
                if distance <= self.radius + 30:  # Larger tolerance for palm
                    # Different palm positions create different sounds
                    if palm_y < self.y - 10:  # Upper palm
                        strikes.append(('ga', 1.0))
                    elif palm_x < self.x - 10:  # Left side
                        strikes.append(('ka', 0.95))
                    else:  # Center palm
                        strikes.append(('dha', 0.9))

        return strikes

    def hit(self, stroke_type='ta', intensity=1.0):
        """Play drum sound."""
        current_time = time.time()
        if current_time - self.last_hit_time >= self.cooldown:
            if stroke_type in self.sounds:
                volume = max(0.3, min(1.0, intensity))
                self.sounds[stroke_type].set_volume(volume)
                self.sounds[stroke_type].play()
            self.is_hit = True
            self.hit_intensity = intensity
            self.last_hit_time = current_time
            return True
        return False

    def draw(self, frame):
        """Draw tabla drum with hit animation."""
        # Calculate current color and size based on hit state
        if self.is_hit:
            time_since_hit = time.time() - self.last_hit_time
            if time_since_hit < 0.2:
                # Animate hit effect
                scale = 1 + (0.2 * self.hit_intensity * (1 - time_since_hit / 0.2))
                current_radius = int(self.radius * scale)
                current_color = self.hit_color
            else:
                self.is_hit = False
                current_radius = self.radius
                current_color = self.color
        else:
            current_radius = self.radius
            current_color = self.color

        # Draw drum with authentic tabla appearance
        if self.drum_type == 'dayan':
            # Dayan: smaller, more circular, metallic appearance
            cv2.circle(frame, (self.x, self.y), current_radius, current_color, 4)
            cv2.circle(frame, (self.x, self.y), current_radius - 8, current_color, -1)
            # Add metallic shine
            cv2.circle(frame, (self.x - 8, self.y - 8), 6, (255, 255, 255), -1)
        else:
            # Bayan: larger, more oval, deeper sound
            cv2.ellipse(frame, (self.x, self.y), (current_radius, int(current_radius * 0.85)),
                       0, 0, 360, current_color, 4)
            cv2.ellipse(frame, (self.x, self.y), (current_radius - 8, int(current_radius * 0.85) - 6),
                       0, 0, 360, current_color, -1)

        # Draw drum name
        cv2.putText(frame, self.name, (self.x - 30, self.y + self.radius + 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Show available bols
        bols = "Ta/Na/Dha/Tin" if self.drum_type == 'dayan' else "Ga/Ka/Dha"
        cv2.putText(frame, bols, (self.x - 45, self.y - self.radius - 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)


class VirtualTabla:
    """Main application class for the virtual tabla."""

    def __init__(self):
        """Initialize the virtual tabla."""
        # Initialize Pygame
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

        # Create tabla drums
        self.drums = self._create_tabla_drums()

        # Track hand positions for velocity calculation
        self.prev_hand_positions = {}

        print("Authentic Virtual Tabla initialized!")
        print("Use traditional tabla playing techniques:")
        print("Right hand (Dayan): Index=Ta, Middle=Na, Ring=Dha, Pinky=Tin")
        print("Left hand (Bayan): Palm heel for Ga, Ka, Dha")
        print("Press 'q' to quit.")

    def _create_tabla_drums(self):
        """Create dayan and bayan tabla drums."""
        drums = []

        center_x = self.frame_width // 2
        center_y = self.frame_height // 2
        drum_radius = 80

        # Dayan (right drum, treble)
        dayan = TablaDrum(
            "Dayan", center_x + 120, center_y, drum_radius,
            180, (255, 200, 100), 'dayan'
        )

        # Bayan (left drum, bass)
        bayan = TablaDrum(
            "Bayan", center_x - 120, center_y, drum_radius,
            90, (100, 150, 255), 'bayan'
        )

        drums.extend([dayan, bayan])
        return drums

    def draw_tabla(self, frame):
        """Draw the virtual tabla with authentic appearance and instructions."""
        # Add semi-transparent overlay for better contrast
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0, self.frame_width, self.frame_height), (0, 0, 0), -1)
        cv2.addWeighted(frame, 0.8, overlay, 0.2, 0, frame)

        # Draw title
        cv2.putText(frame, "AUTHENTIC VIRTUAL TABLA", (int(self.frame_width * 0.3), 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)

        # Draw tabla drums
        for drum in self.drums:
            drum.draw(frame)

        # Draw instructions
        instructions = [
            "Right Hand (Dayan): Use fingertips for Ta, Na, Dha, Tin",
            "Left Hand (Bayan): Use palm heel for Ga, Ka, Dha",
            "Position fingers/palm over drums to play authentic bols",
            "Press 'Q' to quit"
        ]

        for i, instruction in enumerate(instructions):
            y_pos = self.frame_height - 100 + i * 20
            cv2.putText(frame, instruction, (20, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

    def process_hand_landmarks(self, hand_landmarks, frame, hand_label):
        """Process hand for authentic tabla playing techniques."""
        # Get fingertip positions (index, middle, ring, pinky)
        finger_tips = [8, 12, 16, 20]  # Landmark indices
        finger_positions = []

        for tip_idx in finger_tips:
            tip = hand_landmarks.landmark[tip_idx]
            x = int(tip.x * self.frame_width)
            y = int(tip.y * self.frame_height)
            finger_positions.append((x, y))

            # Draw fingertips with hand-specific colors
            hand_color = (0, 255, 0) if hand_label == "Right" else (255, 0, 255)
            cv2.circle(frame, (x, y), 8, hand_color, -1)

        # Get palm position for bayan strikes
        palm_center = hand_landmarks.landmark[9]
        palm_x = int(palm_center.x * self.frame_width)
        palm_y = int(palm_center.y * self.frame_height)
        palm_position = (palm_x, palm_y)

        # Draw palm indicator
        hand_color = (0, 255, 0) if hand_label == "Right" else (255, 0, 255)
        cv2.circle(frame, (palm_x, palm_y), 12, hand_color, -1)
        cv2.circle(frame, (palm_x, palm_y), 15, (255, 255, 255), 2)

        # Check strikes on drums using authentic techniques
        for drum in self.drums:
            strikes = drum.check_strike(finger_positions, palm_position, hand_label)
            for stroke_type, intensity in strikes:
                drum.hit(stroke_type, intensity)
                # Show which stroke was played
                cv2.putText(frame, f"{hand_label}: {stroke_type.upper()}", (palm_x - 35, palm_y - 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, hand_color, 2)

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

            # Draw tabla
            self.draw_tabla(frame)

            # Process hand landmarks
            if results.multi_hand_landmarks:
                for hand_landmarks, handedness in zip(results.multi_hand_landmarks,
                                                     results.multi_handedness):
                    # Draw hand landmarks
                    self.mp_draw.draw_landmarks(frame, hand_landmarks,
                                              self.mp_hands.HAND_CONNECTIONS)

                    # Process for tabla playing
                    hand_label = handedness.classification[0].label
                    self.process_hand_landmarks(hand_landmarks, frame, hand_label)

            # Display FPS
            current_time = time.time()
            fps = 1 / (current_time - prev_time)
            prev_time = current_time
            cv2.putText(frame, f"FPS: {int(fps)}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            # Instructions
            cv2.putText(frame, "Virtual Tabla - Strike with hands!", (10, self.frame_height - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # Show frame
            cv2.imshow("Virtual Tabla", frame)

            # Check for quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Cleanup
        self.cap.release()
        cv2.destroyAllWindows()
        pygame.quit()


if __name__ == "__main__":
    tabla = VirtualTabla()
    tabla.run()