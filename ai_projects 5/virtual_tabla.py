"""
Virtual AI-Driven Tabla
Uses computer vision (MediaPipe) for hand tracking and gesture-based tabla control.
Play traditional Indian tabla rhythms with hand gestures.
"""

import cv2
import mediapipe as mp
import pygame
import numpy as np
import time
import math


class TablaDrum:
    """Represents a virtual tabla drum (dayan or bayan) with position, visuals, and sound."""

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

        # Different strokes for dayan and bayan
        if self.drum_type == 'dayan':
            # Dayan strokes: Ta, Na, Tin
            strokes = {
                'ta': {'freq': base_freq * 2, 'decay': 0.3, 'harmonics': [1, 2.3, 3.1]},
                'na': {'freq': base_freq * 1.8, 'decay': 0.4, 'harmonics': [1, 2.1, 3.5]},
                'tin': {'freq': base_freq * 2.5, 'decay': 0.2, 'harmonics': [1, 3.2, 4.8]}
            }
        else:  # bayan
            # Bayan strokes: Ga, Ka, Dha
            strokes = {
                'ga': {'freq': base_freq * 0.8, 'decay': 0.5, 'harmonics': [1, 1.5, 2.2]},
                'ka': {'freq': base_freq * 0.6, 'decay': 0.6, 'harmonics': [1, 1.8, 2.7]},
                'dha': {'freq': base_freq * 0.4, 'decay': 0.7, 'harmonics': [1, 1.3, 2.0]}
            }

        for stroke_name, params in strokes.items():
            wave = np.zeros(samples)

            for i in range(samples):
                t = i / sample_rate

                # Tabla envelope (sharp attack, exponential decay)
                if t < 0.01:
                    envelope = t / 0.01  # Sharp attack
                else:
                    envelope = np.exp(-params['decay'] * t)

                # Add harmonics with tabla characteristics
                for harmonic in params['harmonics']:
                    freq = params['freq'] * harmonic
                    wave[i] += np.sin(2 * np.pi * freq * t) / len(params['harmonics'])

                wave[i] *= envelope

            # Add tabla-specific noise (membrane slap)
            noise_samples = int(sample_rate * 0.05)
            noise = np.random.randn(noise_samples) * 0.3
            noise_envelope = np.linspace(1, 0, noise_samples)
            wave[:noise_samples] += noise * noise_envelope

            # Normalize
            max_val = np.max(np.abs(wave))
            if max_val > 0:
                wave = wave / max_val

            # Convert to audio
            wave = np.int16(wave * 32767 * 0.8)
            stereo_wave = np.column_stack((wave, wave))
            sounds[stroke_name] = pygame.sndarray.make_sound(stereo_wave)

        return sounds

    def check_hit(self, hand_x, hand_y, velocity):
        """Check if hand hits the drum."""
        distance = math.sqrt((hand_x - self.x)**2 + (hand_y - self.y)**2)
        return distance <= self.radius

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

        # Draw drum
        if self.drum_type == 'dayan':
            # Dayan is more circular
            cv2.circle(frame, (self.x, self.y), current_radius, current_color, 3)
            cv2.circle(frame, (self.x, self.y), current_radius - 5, current_color, -1)
        else:
            # Bayan is slightly oval
            cv2.ellipse(frame, (self.x, self.y), (current_radius, int(current_radius * 0.8)),
                       0, 0, 360, current_color, 3)
            cv2.ellipse(frame, (self.x, self.y), (current_radius - 5, int(current_radius * 0.8) - 5),
                       0, 0, 360, current_color, -1)

        # Draw drum name
        cv2.putText(frame, self.name, (self.x - 30, self.y + self.radius + 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Draw stroke indicators
        stroke_info = "Ta/Na/Tin" if self.drum_type == 'dayan' else "Ga/Ka/Dha"
        cv2.putText(frame, stroke_info, (self.x - 40, self.y - self.radius - 10),
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

        print("Virtual Tabla initialized!")
        print("Strike the drums with your hands to play!")
        print("Different hand speeds and positions create different strokes.")
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

    def process_hand_landmarks(self, hand_landmarks, frame, hand_label):
        """Process hand for tabla playing."""
        # Use palm center for hitting
        palm_center = hand_landmarks.landmark[9]  # Middle of palm

        palm_x = int(palm_center.x * self.frame_width)
        palm_y = int(palm_center.y * self.frame_height)

        # Calculate velocity for stroke type
        current_time = time.time()
        velocity = 0.8  # Default medium velocity

        if hand_label in self.prev_hand_positions:
            prev_x, prev_y, prev_time = self.prev_hand_positions[hand_label]
            time_diff = current_time - prev_time

            if time_diff > 0:
                distance = math.sqrt((palm_x - prev_x)**2 + (palm_y - prev_y)**2)
                speed = distance / time_diff
                velocity = min(1.5, max(0.3, speed / 800))  # Normalized velocity

        # Store current position
        self.prev_hand_positions[hand_label] = (palm_x, palm_y, current_time)

        # Determine stroke type based on velocity and hand
        stroke_type = 'ta'  # Default
        if velocity > 1.0:
            stroke_type = 'tin' if hand_label == "Right" else 'dha'
        elif velocity > 0.7:
            stroke_type = 'na' if hand_label == "Right" else 'ka'
        else:
            stroke_type = 'ta' if hand_label == "Right" else 'ga'

        # Check hits on drums
        for drum in self.drums:
            if drum.check_hit(palm_x, palm_y, velocity):
                drum.hit(stroke_type, velocity)

        # Draw hand indicator
        hand_color = (255, 100, 255) if hand_label == "Right" else (100, 255, 255)
        cv2.circle(frame, (palm_x, palm_y), 15, hand_color, -1)
        cv2.circle(frame, (palm_x, palm_y), 18, (255, 255, 255), 2)

        # Show stroke type
        cv2.putText(frame, f"{hand_label}: {stroke_type.upper()}", (palm_x - 30, palm_y - 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, hand_color, 2)

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

            # Draw tabla drums
            for drum in self.drums:
                drum.draw(frame)

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