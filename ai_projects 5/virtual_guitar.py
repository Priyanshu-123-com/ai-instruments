"""
Virtual AI-Driven Guitar
Uses computer vision (MediaPipe) for hand tracking and gesture-based guitar control.
No physical guitar needed - purely visual interaction with strumming gestures.
"""

import cv2
import mediapipe as mp
import pygame
import numpy as np
import time
import math


class GuitarString:
    """Represents a virtual guitar string with position, visuals, and sound."""
    
    def __init__(self, string_num, note_name, x1, y1, x2, y2, frequency, color):
        """
        Initialize a guitar string.
        
        Args:
            string_num: String number (1-6, 1 is thinnest)
            note_name: Open string note (E, A, D, G, B, E)
            x1, y1: Start position
            x2, y2: End position
            frequency: Base frequency in Hz
            color: RGB color tuple
        """
        self.string_num = string_num
        self.note_name = note_name
        self.x1, self.y1 = x1, y1
        self.x2, self.y2 = x2, y2
        self.frequency = frequency
        self.color = color
        self.is_vibrating = False
        self.last_strum_time = 0
        self.strum_intensity = 0
        self.cooldown = 0.1
        self.vibration_offset = 0
        
        # Generate guitar sound
        self.sound = self._generate_guitar_sound(frequency)
        self.sound.set_volume(1.0)
    
    def _generate_guitar_sound(self, frequency):
        """Generate realistic guitar pluck sound."""
        sample_rate = 44100
        duration = 3.0  # Guitar notes sustain
        samples = int(sample_rate * duration)
        wave = np.zeros(samples)
        
        # Guitar harmonics (strong odd harmonics)
        harmonics = [
            (1.0, 1.0),      # Fundamental
            (2.0, 0.6),      # 2nd
            (3.0, 0.4),      # 3rd (strong in guitar)
            (4.0, 0.25),     # 4th
            (5.0, 0.2),      # 5th
            (6.0, 0.12),     # 6th
            (7.0, 0.08),     # 7th
        ]
        
        for i in range(samples):
            t = i / sample_rate
            
            # Karplus-Strong style envelope (plucked string)
            if t < 0.01:
                # Sharp pluck attack
                envelope = t / 0.01
            else:
                # Exponential decay
                envelope = np.exp(-1.5 * t)
            
            # Add harmonics
            for harmonic_num, amplitude in harmonics:
                phase_shift = np.random.rand() * 0.1  # Slight randomness
                wave[i] += amplitude * np.sin(2 * np.pi * frequency * harmonic_num * t + phase_shift)
            
            wave[i] *= envelope
        
        # Add pluck noise (initial attack)
        pluck_noise = np.random.randn(int(sample_rate * 0.02)) * 0.2
        pluck_envelope = np.linspace(1, 0, len(pluck_noise))
        wave[:len(pluck_noise)] += pluck_noise * pluck_envelope
        
        # Normalize and add warmth
        wave = np.tanh(wave * 0.9)
        max_val = np.max(np.abs(wave))
        if max_val > 0:
            wave = wave / max_val
        
        # Convert to stereo
        wave = np.int16(wave * 32767 * 0.75)
        stereo_wave = np.column_stack((wave, wave))
        
        return pygame.sndarray.make_sound(stereo_wave)
    
    def check_strum(self, x, y, prev_x, prev_y):
        """Check if hand motion crosses the string (strumming)."""
        # Check if the line from (prev_x, prev_y) to (x, y) crosses the string
        # Use line intersection algorithm
        
        # String line: (x1, y1) to (x2, y2)
        # Hand motion: (prev_x, prev_y) to (x, y)
        
        def ccw(A, B, C):
            return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])
        
        def intersect(A, B, C, D):
            return ccw(A,C,D) != ccw(B,C,D) and ccw(A,B,C) != ccw(A,B,D)
        
        A = (self.x1, self.y1)
        B = (self.x2, self.y2)
        C = (prev_x, prev_y)
        D = (x, y)
        
        return intersect(A, B, C, D)
    
    def strum(self, intensity=1.0):
        """Play string sound."""
        current_time = time.time()
        if current_time - self.last_strum_time >= self.cooldown:
            if self.sound:
                volume = max(0.5, min(1.0, intensity))
                self.sound.set_volume(volume)
                self.sound.play()
            self.is_vibrating = True
            self.strum_intensity = intensity
            self.last_strum_time = current_time
            self.vibration_offset = 0
            return True
        return False
    
    def draw(self, frame):
        """Draw guitar string with vibration effect."""
        # Calculate string thickness (thicker = lower strings)
        thickness = 2 + (6 - self.string_num)
        
        # Vibration animation
        if self.is_vibrating:
            time_since_strum = time.time() - self.last_strum_time
            if time_since_strum < 0.5:
                # Vibrate perpendicular to string
                amplitude = 8 * self.strum_intensity * (1 - time_since_strum / 0.5)
                self.vibration_offset = int(amplitude * np.sin(time_since_strum * 30))
            else:
                self.vibration_offset = 0
                if time_since_strum > 0.6:
                    self.is_vibrating = False
        
        # Calculate offset perpendicular to string
        dx = self.x2 - self.x1
        dy = self.y2 - self.y1
        length = math.sqrt(dx*dx + dy*dy)
        if length > 0:
            perp_x = -dy / length * self.vibration_offset
            perp_y = dx / length * self.vibration_offset
        else:
            perp_x, perp_y = 0, 0
        
        # Determine color (brighter when vibrating)
        if self.is_vibrating:
            draw_color = tuple(min(c + 80, 255) for c in self.color)
        else:
            draw_color = self.color
        
        # Draw string with shadow
        shadow_offset = 2
        cv2.line(frame, 
                (int(self.x1 + perp_x + shadow_offset), int(self.y1 + perp_y + shadow_offset)),
                (int(self.x2 + perp_x + shadow_offset), int(self.y2 + perp_y + shadow_offset)),
                (0, 0, 0), thickness + 1)
        
        cv2.line(frame, 
                (int(self.x1 + perp_x), int(self.y1 + perp_y)),
                (int(self.x2 + perp_x), int(self.y2 + perp_y)),
                draw_color, thickness)
        
        # Draw note label at the start
        cv2.putText(frame, self.note_name, 
                   (self.x1 - 30, self.y1 + 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)


class VirtualGuitar:
    """Main application class for the virtual guitar."""
    
    def __init__(self):
        """Initialize the virtual guitar."""
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
        
        # Create guitar strings
        self.strings = self._create_guitar_strings()
        
        # Track hand positions
        self.prev_hand_positions = {}
        
        print("Virtual Guitar initialized!")
        print("Strum across the strings with your hand!")
        print("Press 'q' to quit.")
    
    def _create_guitar_strings(self):
        """Create 6 guitar strings in standard tuning (E A D G B E)."""
        strings = []
        
        # Standard tuning frequencies
        notes = [
            ('E2', 82.41, (200, 100, 100)),   # Low E (thickest)
            ('A2', 110.00, (220, 120, 100)),  # A
            ('D3', 146.83, (230, 150, 100)),  # D
            ('G3', 196.00, (200, 180, 100)),  # G
            ('B3', 246.94, (180, 200, 120)),  # B
            ('E4', 329.63, (150, 220, 150)),  # High E (thinnest)
        ]
        
        # Position strings across middle of screen
        string_spacing = self.frame_height // 8
        start_y = self.frame_height // 3
        
        # Strings span from left to right with slight angle (guitar perspective)
        start_x = int(self.frame_width * 0.25)
        end_x = int(self.frame_width * 0.75)
        
        for i, (note, freq, color) in enumerate(notes):
            y_pos = start_y + i * string_spacing
            # Add slight angle to make it look more natural
            y1 = y_pos - 20
            y2 = y_pos + 20
            
            strings.append(GuitarString(
                i + 1, note, 
                start_x, y1,
                end_x, y2,
                freq, color
            ))
        
        return strings
    
    def process_hand_landmarks(self, hand_landmarks, frame, hand_label):
        """Process hand for strumming motion."""
        # Use index fingertip for strumming
        index_tip = hand_landmarks.landmark[8]
        
        tip_x = int(index_tip.x * self.frame_width)
        tip_y = int(index_tip.y * self.frame_height)
        
        # Calculate velocity
        current_time = time.time()
        velocity = 0.7
        prev_x, prev_y = tip_x, tip_y
        
        if hand_label in self.prev_hand_positions:
            prev_x, prev_y, prev_time = self.prev_hand_positions[hand_label]
            time_diff = current_time - prev_time
            
            if time_diff > 0:
                distance = math.sqrt((tip_x - prev_x)**2 + (tip_y - prev_y)**2)
                speed = distance / time_diff
                velocity = min(1.0, max(0.5, speed / 1000))
        
        # Hand color
        indicator_color = (255, 100, 255) if hand_label == "Right" else (100, 255, 255)
        
        # Draw fingertip trail
        if hand_label in self.prev_hand_positions:
            cv2.line(frame, (prev_x, prev_y), (tip_x, tip_y), indicator_color, 3)
        
        cv2.circle(frame, (tip_x, tip_y), 12, indicator_color, -1)
        cv2.circle(frame, (tip_x, tip_y), 14, (255, 255, 255), 2)
        
        # Check if hand crosses any string
        if hand_label in self.prev_hand_positions:
            for string in self.strings:
                if string.check_strum(tip_x, tip_y, prev_x, prev_y):
                    string.strum(velocity)
        
        # Store position
        self.prev_hand_positions[hand_label] = (tip_x, tip_y, current_time)
    
    def draw_guitar_body(self, frame):
        """Draw simplified guitar body/neck."""
        # Draw fretboard/neck area
        neck_x1 = int(self.frame_width * 0.2)
        neck_y1 = int(self.frame_height * 0.25)
        neck_x2 = int(self.frame_width * 0.8)
        neck_y2 = int(self.frame_height * 0.75)
        
        # Neck background (wood texture simulation)
        for i in range(20):
            wood_shade = 60 + i * 2
            cv2.rectangle(frame, 
                         (neck_x1 + i, neck_y1 + i),
                         (neck_x2 - i, neck_y2 - i),
                         (wood_shade // 3, wood_shade // 2, wood_shade),
                         2)
        
        # Fret markers
        fret_positions = [0.25, 0.4, 0.55, 0.7, 0.85]
        for pos in fret_positions:
            fret_x = int(neck_x1 + (neck_x2 - neck_x1) * pos)
            cv2.line(frame, (fret_x, neck_y1), (fret_x, neck_y2), 
                    (180, 180, 180), 2)
            # Fret dots
            dot_y = (neck_y1 + neck_y2) // 2
            cv2.circle(frame, (fret_x, dot_y), 8, (200, 200, 200), -1)
            cv2.circle(frame, (fret_x, dot_y), 6, (150, 150, 150), -1)
    
    def run(self):
        """Main application loop."""
        while True:
            success, frame = self.cap.read()
            if not success:
                print("Failed to read from camera")
                break
            
            frame = cv2.flip(frame, 1)
            
            # Dark stage background
            overlay = np.zeros_like(frame)
            for y in range(self.frame_height):
                darkness = int(20 + (y / self.frame_height) * 15)
                overlay[y, :] = [darkness // 4, darkness // 3, darkness // 2]
            
            frame = cv2.addWeighted(frame, 0.35, overlay, 0.65, 0)
            
            # Convert for MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb_frame)
            
            # Draw guitar body
            self.draw_guitar_body(frame)
            
            # Draw strings
            for string in self.strings:
                string.draw(frame)
            
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
            cv2.rectangle(frame, (0, 0), (self.frame_width, 90), (0, 0, 0), -1)
            cv2.putText(frame, "AI VIRTUAL GUITAR - Strum Across the Strings!", 
                       (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
            cv2.putText(frame, "Move your finger across strings to play | Press 'Q' to Quit", 
                       (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            cv2.imshow("Virtual Guitar", frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
        
        self.cleanup()
    
    def cleanup(self):
        """Release resources."""
        self.cap.release()
        cv2.destroyAllWindows()
        pygame.quit()
        print("Virtual Guitar closed.")


def main():
    """Entry point."""
    try:
        guitar = VirtualGuitar()
        guitar.run()
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
