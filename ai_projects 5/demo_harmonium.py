"""
Demo Virtual Harmonium
A demonstration version that shows the UI layout and plays sounds without camera input.
"""

import cv2
import pygame
import numpy as np
import time
import math


class DemoHarmoniumKey:
    """Represents a virtual harmonium key for demo purposes."""
    
    def __init__(self, note_name, x, y, width, height, frequency, is_black=False):
        self.note_name = note_name
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.frequency = frequency
        self.is_black = is_black
        self.is_pressed = False
        self.press_intensity = 0.8
        self.sound = self._generate_harmonium_sound(frequency)
        self.sound.set_volume(0.8)
        
        # Colors
        if is_black:
            self.color = (80, 60, 120)  # Dark purple for komal notes
            self.pressed_color = (150, 100, 200)
        else:
            self.color = (220, 200, 240)  # Light purple for shuddha notes
            self.pressed_color = (255, 230, 255)
    
    def _generate_harmonium_sound(self, frequency):
        """Generate realistic harmonium sound."""
        sample_rate = 44100
        duration = 3.0
        samples = int(sample_rate * duration)
        wave = np.zeros(samples)
        
        harmonics = [
            (1.0, 1.0), (2.0, 0.8), (3.0, 0.6), 
            (4.0, 0.4), (5.0, 0.3), (6.0, 0.2)
        ]
        
        for i in range(samples):
            t = i / sample_rate
            tremolo = 1.0 + 0.15 * np.sin(2 * np.pi * 4.5 * t)
            vibrato = 0.5 * np.sin(2 * np.pi * 6.0 * t)
            
            for harmonic_num, amplitude in harmonics:
                mod_freq = frequency * harmonic_num * (1 + 0.001 * vibrato)
                wave[i] += amplitude * np.sin(2 * np.pi * mod_freq * t)
            
            wave[i] *= tremolo
        
        breath_noise = np.random.randn(samples) * 0.1
        for i in range(1, samples):
            breath_noise[i] = 0.9 * breath_noise[i-1] + 0.1 * breath_noise[i]
        
        wave += breath_noise
        wave = np.tanh(wave * 0.7)
        max_val = np.max(np.abs(wave))
        if max_val > 0:
            wave = wave / max_val
        
        wave = np.int16(wave * 32767 * 0.8)
        stereo_wave = np.column_stack((wave, wave))
        
        return pygame.sndarray.make_sound(stereo_wave)
    
    def press(self):
        """Press the key."""
        self.is_pressed = True
        self.sound.play()
    
    def release(self):
        """Release the key."""
        self.is_pressed = False
    
    def draw(self, frame):
        """Draw the key."""
        color = self.pressed_color if self.is_pressed else self.color
        
        # Shadow
        cv2.rectangle(frame, (int(self.x + 4), int(self.y + 4)), 
                     (int(self.x + self.width + 4), int(self.y + self.height + 4)), 
                     (0, 0, 0), -1)
        
        # Main key body
        for i in range(int(self.height)):
            y_pos = int(self.y + i)
            gradient_factor = 1.0 - (i / self.height) * 0.4
            grad_color = tuple(int(c * gradient_factor) for c in color)
            cv2.line(frame, (int(self.x), y_pos), 
                    (int(self.x + self.width), y_pos), grad_color, 1)
        
        # Highlight
        highlight_height = int(self.height * 0.15)
        for i in range(highlight_height):
            alpha = 1.0 - (i / highlight_height)
            highlight_color = tuple(min(255, int(c + 50 * alpha)) for c in color)
            cv2.line(frame, (int(self.x + 2), int(self.y + i)), 
                    (int(self.x + self.width - 2), int(self.y + i)), highlight_color, 1)
        
        # Edges
        cv2.line(frame, (int(self.x), int(self.y)), (int(self.x + self.width), int(self.y)), 
                (150, 130, 180), 2)
        cv2.line(frame, (int(self.x), int(self.y + self.height)), 
                (int(self.x + self.width), int(self.y + self.height)), (80, 60, 120), 2)
        
        # Note label
        text_size = cv2.getTextSize(self.note_name, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
        text_x = int(self.x + (self.width - text_size[0]) // 2)
        text_y = int(self.y + self.height - 15)
        
        cv2.rectangle(frame, 
                     (int(text_x - 8), int(text_y - text_size[1] - 5)),
                     (int(text_x + text_size[0] + 8), int(text_y + 5)),
                     (0, 0, 0), -1)
        
        cv2.putText(frame, self.note_name, (text_x, text_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (240, 220, 255), 2)
        
        # Glow when pressed
        if self.is_pressed:
            glow_overlay = frame.copy()
            cv2.rectangle(glow_overlay, (int(self.x), int(self.y)), 
                         (int(self.x + self.width), int(self.y + self.height)), 
                         (255, 240, 255), -1)
            cv2.addWeighted(glow_overlay, 0.3 * self.press_intensity, 
                           frame, 0.7, 0, frame)


class DemoBellows:
    """Demo bellows control."""
    
    def __init__(self, x, y, width, height):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.pressure = 0.7  # Default pressure
    
    def draw(self, frame):
        """Draw bellows."""
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
            cv2.rectangle(frame, (pleat_x, int(self.y + 5)), 
                        (int(pleat_x + pleat_width), int(self.y + self.height - 5)), 
                        (60, 45, 30), -1)
            cv2.rectangle(frame, (int(pleat_x + 2), int(self.y + 7)), 
                        (int(pleat_x + pleat_width - 2), int(self.y + self.height - 7)), 
                        (40, 30, 20), -1)
        
        # Pressure indicator
        indicator_height = int(self.height * 0.8 * self.pressure)
        if indicator_height > 0:
            indicator_y = int(self.y + self.height * 0.9 - indicator_height)
            cv2.rectangle(frame, (int(self.x + 10), indicator_y), 
                         (int(self.x + self.width - 10), int(self.y + self.height * 0.9)), 
                         (200, 150, 100), -1)
            for i in range(indicator_height):
                y_pos = int(indicator_y + i)
                alpha = 1.0 - (i / indicator_height) * 0.7
                color = (int(200 * alpha), int(150 * alpha), int(100 * alpha))
                cv2.line(frame, (int(self.x + 10), y_pos), 
                        (int(self.x + self.width - 10), y_pos), color, 1)
        
        # Labels
        cv2.putText(frame, "BELLOWS", (int(self.x + 15), int(self.y + 40)), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 240, 200), 2)
        cv2.putText(frame, "Volume Control", (int(self.x + 10), int(self.y + 70)), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 240, 200), 1)


class DemoVirtualHarmonium:
    """Demo version of the virtual harmonium."""
    
    def __init__(self):
        pygame.mixer.pre_init(44100, -16, 2, 512)
        pygame.init()
        
        self.frame_width = 1280
        self.frame_height = 720
        
        # Create keys
        self.keys = self._create_keys()
        
        # Create bellows
        self.bellows = DemoBellows(
            int(self.frame_width * 0.05), 
            int(self.frame_height * 0.2),
            int(self.frame_width * 0.2),
            int(self.frame_height * 0.6)
        )
        
        print("Demo Virtual Harmonium Ready!")
        print("Click on keys to play notes")
        print("Press ESC to quit")
    
    def _create_keys(self):
        """Create demo keys."""
        keys = []
        
        # Indian classical notes
        sa_frequency = 261.63
        shuddha_notes = ['Sa', 'Re', 'Ga', 'Ma', 'Pa', 'Dha', 'Ni', 'SA']
        shuddha_ratios = [1.0, 9/8, 5/4, 4/3, 3/2, 5/3, 15/8, 2.0]
        
        komal_notes = ['rek', 'gak', 'mak', 'dhak', 'nik']
        komal_ratios = [16/15, 6/5, 32/27, 8/5, 9/5]
        
        # Key dimensions
        key_width = int(self.frame_width * 0.06)
        key_height = int(self.frame_height * 0.35)
        start_x = int(self.frame_width * 0.3)
        key_y = int(self.frame_height * 0.4)
        
        # Create shuddha (natural) keys
        for i, (note, ratio) in enumerate(zip(shuddha_notes, shuddha_ratios)):
            frequency = sa_frequency * ratio
            keys.append(DemoHarmoniumKey(
                note, 
                int(start_x + i * key_width * 1.2),
                int(key_y),
                int(key_width),
                int(key_height),
                frequency,
                is_black=False
            ))
        
        # Create komal (flat) keys
        komal_positions = [0.5, 1.5, 3.5, 5.5, 6.5]
        for i, (note, ratio) in enumerate(zip(komal_notes, komal_ratios)):
            frequency = sa_frequency * ratio
            key_x = int(start_x + komal_positions[i] * key_width * 1.2)
            keys.append(DemoHarmoniumKey(
                note,
                int(key_x),
                int(key_y - 20),
                int(key_width * 0.7),
                int(key_height * 0.8),
                frequency,
                is_black=True
            ))
        
        return keys
    
    def run(self):
        """Run demo."""
        # Create window
        cv2.namedWindow("Demo Virtual Harmonium")
        cv2.setMouseCallback("Demo Virtual Harmonium", self._mouse_callback)
        
        # Demo sequence
        demo_sequence = [
            (0, 1.0),  # Sa
            (1, 1.0),  # Re
            (2, 1.0),  # Ga
            (3, 1.0),  # Ma
            (4, 1.0),  # Pa
            (5, 1.0),  # Dha
            (6, 1.0),  # Ni
            (7, 1.0),  # SA
        ]
        
        demo_index = 0
        
        while True:
            # Create frame
            frame = np.zeros((self.frame_height, self.frame_width, 3), dtype=np.uint8)
            
            # Background
            for y in range(self.frame_height):
                darkness = int(30 + (y / self.frame_height) * 25)
                frame[y, :] = [darkness // 2, darkness // 3, darkness // 2]
            
            # Draw harmonium body
            self._draw_harmonium_body(frame)
            
            # Draw bellows
            self.bellows.draw(frame)
            
            # Draw keys
            for key in self.keys:
                key.draw(frame)
            
            # Instructions
            cv2.rectangle(frame, (0, 0), (self.frame_width, 100), (0, 0, 0), -1)
            cv2.putText(frame, "DEMO VIRTUAL HARMONIUM - Indian Classical Instrument", 
                       (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 200, 150), 2)
            cv2.putText(frame, "Click on keys to play notes | Press ESC to Quit", 
                       (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 255, 200), 2)
            
            # Show frame
            cv2.imshow("Demo Virtual Harmonium", frame)
            
            # Play demo sequence
            if demo_index < len(demo_sequence):
                key_idx, intensity = demo_sequence[demo_index]
                if not self.keys[key_idx].is_pressed:
                    self.keys[key_idx].press()
                    print(f"Playing {self.keys[key_idx].note_name}")
                    demo_index += 1
                    time.sleep(1.5)  # Hold note
                    self.keys[key_idx].release()
                    time.sleep(0.5)  # Gap between notes
            
            # Check for ESC key
            key = cv2.waitKey(100) & 0xFF
            if key == 27:  # ESC key
                break
        
        cv2.destroyAllWindows()
        pygame.quit()
        print("Demo Virtual Harmonium closed.")
    
    def _mouse_callback(self, event, x, y, flags, param):
        """Handle mouse events."""
        if event == cv2.EVENT_LBUTTONDOWN:
            for key in self.keys:
                if (key.x <= x <= key.x + key.width and 
                    key.y <= y <= key.y + key.height):
                    key.press()
                    print(f"Playing {key.note_name} with mouse")
                    break
    
    def _draw_harmonium_body(self, frame):
        """Draw harmonium body."""
        body_x1 = int(self.frame_width * 0.25)
        body_y1 = int(self.frame_height * 0.35)
        body_x2 = int(self.frame_width * 0.95)
        body_y2 = int(self.frame_height * 0.8)
        
        # Main body
        cv2.rectangle(frame, (body_x1, body_y1), (body_x2, body_y2), (100, 70, 40), -1)
        
        # Wood grain
        for i in range(20):
            wood_shade = 90 + i * 2
            cv2.rectangle(frame, 
                         (int(body_x1 + i * 1.5), int(body_y1 + i * 0.8)),
                         (int(body_x2 - i * 1.5), int(body_y2 - i * 0.8)),
                         (wood_shade // 2, wood_shade // 3, wood_shade // 2),
                         1)
        
        # Sound holes
        hole_center_x = (body_x1 + body_x2) // 2
        hole_center_y = (body_y1 + body_y2) // 2
        for i in range(8):
            angle = (2 * np.pi * i) / 8
            hole_x = int(hole_center_x + 40 * np.cos(angle))
            hole_y = int(hole_center_y + 30 * np.sin(angle))
            cv2.circle(frame, (hole_x, hole_y), 8, (60, 40, 25), -1)
            cv2.circle(frame, (hole_x, hole_y), 6, (40, 25, 15), -1)
        
        # Brand name
        brand_text = "VIRTUAL HARMONIUM"
        text_size = cv2.getTextSize(brand_text, cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 1.2, 2)[0]
        text_x = int((body_x1 + body_x2 - text_size[0]) // 2)
        text_y = int(body_y1 + 40)
        cv2.putText(frame, brand_text, (text_x, text_y), 
                   cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 1.2, (220, 190, 160), 2)


def main():
    """Entry point."""
    try:
        harmonium = DemoVirtualHarmonium()
        harmonium.run()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
