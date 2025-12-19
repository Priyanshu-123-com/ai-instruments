"""
Virtual AI-Driven Flute
Uses computer vision (MediaPipe) for hand tracking and gesture-based flute control.
Play notes by positioning fingers over virtual holes on the flute.
"""

import cv2
import mediapipe as mp
import pygame
import numpy as np
import time
import math


class FluteNote:
    """Represents a virtual flute note with position and sound."""

    def __init__(self, note_name, frequency, hole_positions, color):
        """
        Initialize a flute note.

        Args:
            note_name: Note name (e.g., 'C4')
            frequency: Frequency in Hz
            hole_positions: List of (x,y) positions for finger holes
            color: RGB color tuple
        """
        self.note_name = note_name
        self.frequency = frequency
        self.hole_positions = hole_positions
        self.color = color
        self.is_playing = False
        self.last_play_time = 0
        self.cooldown = 0.1

        # Generate flute sound
        self.sound = self._generate_flute_sound(frequency)
        self.sound.set_volume(1.0)

    def _generate_flute_sound(self, frequency):
        """Generate realistic flute sound using sine waves with harmonics."""
        sample_rate = 44100
        duration = 2.5  # Longer sustain for flute
        samples = int(sample_rate * duration)

        wave = np.zeros(samples)

        # Flute harmonics (strong even harmonics, more complex than before)
        harmonics = [
            (1.0, 1.0),      # Fundamental
            (2.0, 0.9),      # 2nd (very strong in flute)
            (3.0, 0.4),      # 3rd
            (4.0, 0.7),      # 4th (strong)
            (5.0, 0.3),      # 5th
            (6.0, 0.5),      # 6th
            (7.0, 0.2),      # 7th
            (8.0, 0.3),      # 8th
        ]

        for i in range(samples):
            t = i / sample_rate

            # Flute envelope (smooth attack, slow decay, more realistic)
            if t < 0.08:
                envelope = t / 0.08  # Slower attack
            elif t < 0.2:
                envelope = 1.0  # Sustain
            else:
                envelope = np.exp(-0.3 * (t - 0.2))  # Slower decay

            # Add harmonics with slight phase variation
            for harmonic_num, amplitude in harmonics:
                freq = frequency * harmonic_num
                phase = np.sin(2 * np.pi * freq * t + harmonic_num * 0.1)
                wave[i] += amplitude * phase

            wave[i] *= envelope

        # Add breath noise (more subtle for flute)
        breath_noise = np.random.randn(int(sample_rate * 0.15)) * 0.08
        breath_envelope = np.linspace(0.3, 0, len(breath_noise))
        wave[:len(breath_noise)] += breath_noise * breath_envelope

        # Add slight vibrato
        vibrato_rate = 5.0  # Hz
        vibrato_depth = 0.005
        vibrato = 1 + vibrato_depth * np.sin(2 * np.pi * vibrato_rate * np.arange(samples) / sample_rate)
        wave *= vibrato

        # Normalize
        max_val = np.max(np.abs(wave))
        if max_val > 0:
            wave = wave / max_val

        # Convert to audio
        wave = np.int16(wave * 32767 * 0.8)
        stereo_wave = np.column_stack((wave, wave))

        return pygame.sndarray.make_sound(stereo_wave)

    def check_fingers(self, finger_positions):
        """Check if fingers are positioned correctly over holes."""
        covered_holes = 0
        required_holes = len(self.hole_positions)

        for hole_x, hole_y in self.hole_positions:
            hole_covered = False
            for finger_x, finger_y in finger_positions:
                distance = math.sqrt((finger_x - hole_x)**2 + (finger_y - hole_y)**2)
                if distance < 35:  # Increased radius for bigger holes
                    hole_covered = True
                    break
            if hole_covered:
                covered_holes += 1

        # For this note, all required holes must be covered
        return covered_holes == required_holes

    def play(self):
        """Play the note."""
        current_time = time.time()
        if current_time - self.last_play_time >= self.cooldown:
            if self.sound:
                self.sound.play()
            self.is_playing = True
            self.last_play_time = current_time
            return True
        return False

    def draw(self, frame):
        """Draw flute holes and note label."""
        for i, (hole_x, hole_y) in enumerate(self.hole_positions):
            # Draw bigger hole
            cv2.circle(frame, (hole_x, hole_y), 25, self.color, 3)
            cv2.circle(frame, (hole_x, hole_y), 15, self.color, -1)

            # Hole number
            cv2.putText(frame, str(i+1), (hole_x - 8, hole_y + 8),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Draw note label above the first hole
        if self.hole_positions:
            label_x = self.hole_positions[0][0] - 20
            label_y = self.hole_positions[0][1] - 40
            cv2.putText(frame, self.note_name, (label_x, label_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)


class VirtualFlute:
    """Main application class for the virtual flute."""

    def __init__(self):
        """Initialize the virtual flute."""
        # Initialize Pygame
        pygame.mixer.pre_init(44100, -16, 2, 512)
        pygame.init()

        # Initialize MediaPipe Hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,  # One hand for flute
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

        # Create flute notes
        self.notes = self._create_flute_notes()

        print("Virtual Flute initialized!")
        print("Position your fingers over the holes to play notes!")
        print("Press 'q' to quit.")

    def _create_flute_notes(self):
        """Create flute notes with different hole combinations."""
        notes = []

        # Flute notes with hole positions (simplified)
        # Holes positioned horizontally across the screen
        base_y = int(self.frame_height * 0.6)
        spacing = 80  # Increased spacing for bigger holes
        start_x = int(self.frame_width * 0.2)

        note_configs = [
            ('C4', 261.63, []),  # All holes open
            ('D4', 293.66, [(start_x, base_y)]),  # One hole covered
            ('E4', 329.63, [(start_x, base_y), (start_x + spacing, base_y)]),  # Two holes
            ('F4', 349.23, [(start_x, base_y), (start_x + spacing, base_y), (start_x + 2*spacing, base_y)]),  # Three holes
            ('G4', 392.00, [(start_x, base_y), (start_x + spacing, base_y), (start_x + 2*spacing, base_y), (start_x + 3*spacing, base_y)]),  # Four holes
            ('A4', 440.00, [(start_x, base_y), (start_x + spacing, base_y), (start_x + 2*spacing, base_y), (start_x + 3*spacing, base_y), (start_x + 4*spacing, base_y)]),  # Five holes
        ]

        colors = [
            (100, 200, 255), (150, 220, 255), (200, 240, 255),
            (250, 200, 150), (255, 180, 100), (255, 150, 50)
        ]

        for i, (note, freq, holes) in enumerate(note_configs):
            notes.append(FluteNote(note, freq, holes, colors[i % len(colors)]))

        return notes

    def process_hand_landmarks(self, hand_landmarks, frame):
        """Process hand for finger positions over holes."""
        # Get fingertip positions (index, middle, ring, pinky)
        finger_tips = [8, 12, 16, 20]  # Landmark indices
        finger_positions = []

        for tip_idx in finger_tips:
            tip = hand_landmarks.landmark[tip_idx]
            x = int(tip.x * self.frame_width)
            y = int(tip.y * self.frame_height)
            finger_positions.append((x, y))

            # Draw fingertips
            cv2.circle(frame, (x, y), 8, (0, 255, 0), -1)

        # Check which note to play
        for note in self.notes:
            if note.check_fingers(finger_positions):
                note.play()
                break

    def draw_flute(self, frame):
        """Draw the virtual flute."""
        # Draw flute body (horizontal line)
        flute_y = int(self.frame_height * 0.6)
        start_x = int(self.frame_width * 0.15)
        end_x = int(self.frame_width * 0.85)
        cv2.line(frame, (start_x, flute_y), (end_x, flute_y), (139, 69, 19), 12)

        # Draw all note holes
        for note in self.notes:
            note.draw(frame)

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

            # Draw flute
            self.draw_flute(frame)

            # Process hand landmarks
            if results.multi_hand_landmarks:
                for hand_landmarks, handedness in zip(results.multi_hand_landmarks,
                                                     results.multi_handedness):
                    # Draw hand landmarks
                    self.mp_draw.draw_landmarks(frame, hand_landmarks,
                                              self.mp_hands.HAND_CONNECTIONS)

                    # Process for flute playing
                    self.process_hand_landmarks(hand_landmarks, frame)

                    # Label hand
                    hand_label = handedness.classification[0].label
                    cv2.putText(frame, hand_label, (10, 50),
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            # Display FPS
            current_time = time.time()
            fps = 1 / (current_time - prev_time)
            prev_time = current_time
            cv2.putText(frame, f"FPS: {int(fps)}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            # Show frame
            cv2.imshow("Virtual Flute", frame)

            # Check for quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Cleanup
        self.cap.release()
        cv2.destroyAllWindows()
        pygame.quit()


if __name__ == "__main__":
    flute = VirtualFlute()
    flute.run()