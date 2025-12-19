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
        self.active_color = tuple(min(c + 100, 255) for c in color)  # Brighter when active
        self.is_playing = False
        self.last_play_time = 0
        self.cooldown = 0.1

        # Generate flute sound
        self.sound = self._generate_flute_sound(frequency)
        self.sound.set_volume(1.0)

    def _generate_flute_sound(self, frequency):
        """Generate realistic flute sound using advanced synthesis."""
        sample_rate = 44100
        duration = 3.0  # Longer sustain
        samples = int(sample_rate * duration)

        wave = np.zeros(samples)

        # Advanced flute synthesis with formants
        # Flute formants: around 1kHz, 2.5kHz, 4kHz
        formants = [
            (1.0, 1.0, 0.1),    # Fundamental
            (2.0, 0.8, 0.05),   # 2nd harmonic
            (3.0, 0.4, 0.03),   # 3rd
            (4.0, 0.6, 0.02),   # 4th
            (5.0, 0.3, 0.015),  # 5th
            (6.0, 0.4, 0.01),   # 6th
            (7.0, 0.2, 0.008),  # 7th
            (8.0, 0.25, 0.006), # 8th
        ]

        # Formant frequencies for flute-like timbre
        formant_freqs = [1000, 2500, 4000]
        formant_bws = [100, 200, 300]

        for i in range(samples):
            t = i / sample_rate

            # Advanced envelope with multiple stages
            if t < 0.02:
                envelope = t / 0.02  # Very fast attack
            elif t < 0.1:
                envelope = 1.0 + 0.2 * (t - 0.02) / 0.08  # Slight overshoot
            elif t < 0.3:
                envelope = 1.0  # Sustain
            else:
                envelope = np.exp(-0.8 * (t - 0.3))  # Decay

            # Generate base wave with harmonics
            base_wave = 0
            for harmonic_num, amplitude, phase_offset in formants:
                freq = frequency * harmonic_num
                base_wave += amplitude * np.sin(2 * np.pi * freq * t + phase_offset)

            # Apply formant filtering for realistic timbre
            filtered_wave = base_wave
            for f_freq, f_bw in zip(formant_freqs, formant_bws):
                # Simple formant enhancement
                formant_gain = 1 + 0.5 * np.exp(-((freq - f_freq) / f_bw)**2)
                filtered_wave *= formant_gain

            wave[i] = filtered_wave * envelope

        # Add realistic breath noise and turbulence
        # Initial breath burst
        breath_samples = int(sample_rate * 0.05)
        breath_noise = np.random.randn(breath_samples) * 0.15
        breath_envelope = np.exp(-np.linspace(0, 3, breath_samples))
        wave[:breath_samples] += breath_noise * breath_envelope

        # Subtle continuous breath
        continuous_breath = np.random.randn(samples) * 0.02
        breath_env = np.exp(-t * 0.5)  # Fade out
        wave += continuous_breath * breath_env

        # Add vibrato
        vibrato_rate = 5.5  # Slightly faster than typical
        vibrato_depth = 0.008
        vibrato = 1 + vibrato_depth * np.sin(2 * np.pi * vibrato_rate * np.arange(samples) / sample_rate)
        wave *= vibrato

        # Add subtle pitch variations (microtonal instability)
        pitch_variation = 1 + 0.001 * np.sin(2 * np.pi * 12 * np.arange(samples) / sample_rate)
        # This would require resampling, so skip for simplicity

        # Normalize and add warmth
        wave = np.tanh(wave * 0.8)  # Soft clipping for warmth
        max_val = np.max(np.abs(wave))
        if max_val > 0:
            wave = wave / max_val

        # Convert to audio
        wave = np.int16(wave * 32767 * 0.85)
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

    def draw(self, frame, is_active=False):
        """Draw flute holes and note label."""
        current_color = self.active_color if is_active else self.color

        for i, (hole_x, hole_y) in enumerate(self.hole_positions):
            # Draw bigger hole with glow effect when active
            if is_active:
                # Outer glow
                cv2.circle(frame, (hole_x, hole_y), 30, (255, 255, 0), 2)
                cv2.circle(frame, (hole_x, hole_y), 28, current_color, -1)
                # Inner hole
                cv2.circle(frame, (hole_x, hole_y), 25, (50, 50, 50), -1)
                cv2.circle(frame, (hole_x, hole_y), 20, current_color, 2)
            else:
                cv2.circle(frame, (hole_x, hole_y), 25, current_color, 3)
                cv2.circle(frame, (hole_x, hole_y), 15, current_color, -1)

            # Hole number
            text_color = (255, 255, 255) if not is_active else (0, 0, 0)
            cv2.putText(frame, str(i+1), (hole_x - 8, hole_y + 8),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, text_color, 2)

        # Draw note label with highlight when active
        if self.hole_positions:
            label_x = self.hole_positions[0][0] - 25
            label_y = self.hole_positions[0][1] - 45
            font_scale = 1.0 if is_active else 0.8
            font_thickness = 3 if is_active else 2
            cv2.putText(frame, self.note_name, (label_x, label_y),
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), font_thickness)


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
            max_num_hands=2,  # Both hands for complex fingerings
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6
        )
        self.mp_draw = mp.solutions.drawing_utils

        # Track current playing note
        self.current_note = None
        self.last_note_time = 0

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

    def process_hand_landmarks(self, hand_landmarks, frame, hand_label):
        """Process hand for finger positions over holes."""
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
            cv2.circle(frame, (x, y), 10, hand_color, -1)
            cv2.circle(frame, (x, y), 12, (255, 255, 255), 2)

        return finger_positions

    def combine_hands_and_play(self, all_finger_positions):
        """Combine fingers from both hands and determine which note to play."""
        # Check which note matches the current finger configuration
        for note in self.notes:
            if note.check_fingers(all_finger_positions):
                if note != self.current_note:
                    note.play()
                    self.current_note = note
                    self.last_note_time = time.time()
                return note
        return None

    def draw_flute(self, frame, current_note=None):
        """Draw the virtual flute with enhanced UI."""
        # Add semi-transparent overlay for better contrast
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (self.frame_width, self.frame_height), (0, 0, 0), -1)
        cv2.addWeighted(frame, 0.7, overlay, 0.3, 0, frame)

        # Draw flute body (horizontal line with 3D effect)
        flute_y = int(self.frame_height * 0.6)
        start_x = int(self.frame_width * 0.15)
        end_x = int(self.frame_width * 0.85)

        # Shadow
        cv2.line(frame, (start_x + 3, flute_y + 3), (end_x + 3, flute_y + 3), (0, 0, 0), 16)
        # Main body
        cv2.line(frame, (start_x, flute_y), (end_x, flute_y), (139, 69, 19), 14)
        # Highlight
        cv2.line(frame, (start_x, flute_y - 2), (end_x, flute_y - 2), (200, 120, 80), 2)

        # Draw all note holes
        for note in self.notes:
            is_active = (note == current_note)
            note.draw(frame, is_active)

        # Draw title
        cv2.putText(frame, "VIRTUAL FLUTE", (int(self.frame_width * 0.4), 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
        cv2.putText(frame, "Position fingers over holes to play", (int(self.frame_width * 0.35), 80),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)

        # Show current note being played
        if current_note:
            cv2.putText(frame, f"Now Playing: {current_note.note_name}",
                       (int(self.frame_width * 0.4), self.frame_height - 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 0), 3)

        # Instructions
        instructions = [
            "Use both hands for complex fingerings",
            "Green fingertips: Right hand",
            "Pink fingertips: Left hand",
            "Press 'Q' to quit"
        ]

        for i, instruction in enumerate(instructions):
            y_pos = self.frame_height - 120 + i * 25
            cv2.putText(frame, instruction, (20, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1)

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

            # Collect fingers from all detected hands
            all_finger_positions = []
            if results.multi_hand_landmarks:
                for hand_landmarks, handedness in zip(results.multi_hand_landmarks,
                                                     results.multi_handedness):
                    # Draw hand landmarks
                    self.mp_draw.draw_landmarks(frame, hand_landmarks,
                                              self.mp_hands.HAND_CONNECTIONS)

                    # Process hand and collect finger positions
                    hand_label = handedness.classification[0].label
                    finger_positions = self.process_hand_landmarks(hand_landmarks, frame, hand_label)
                    all_finger_positions.extend(finger_positions)

                    # Label hand
                    cv2.putText(frame, hand_label, (10, 50 + (0 if hand_label == "Right" else 30)),
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            # Determine and play current note
            current_note = self.combine_hands_and_play(all_finger_positions)

            # Clear current note if no fingers detected for a while
            if not all_finger_positions and time.time() - self.last_note_time > 0.5:
                self.current_note = None

            # Draw flute with current note highlight
            self.draw_flute(frame, current_note)

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