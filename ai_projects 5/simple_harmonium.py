import pygame
import numpy as np
import time
import math

# Initialize Pygame
pygame.mixer.pre_init(44100, -16, 2, 512)
pygame.mixer.init()
pygame.init()

# Screen dimensions
WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Interactive Harmonium")

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (200, 200, 200)
DARK_GRAY = (150, 150, 150)
LIGHT_GRAY = (230, 230, 230)
RED = (255, 100, 100)
BLUE = (100, 100, 255)
BROWN = (139, 69, 19)
LIGHT_BROWN = (205, 133, 63)
DARK_BROWN = (101, 67, 33)

class HarmoniumKey:
    def __init__(self, x, y, width, height, note_name, frequency, color):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.note_name = note_name
        self.frequency = frequency
        self.color = color
        self.is_pressed = False
        self.sound = self._generate_harmonium_sound(frequency)
        self.sound.set_volume(0.0)  # Start silent, controlled by bellows

    def _generate_harmonium_sound(self, frequency):
        """Generate a harmonium-like sound."""
        sample_rate = 44100
        duration = 2.0  # seconds
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

    def play(self):
        """Play the key sound."""
        if not self.is_playing():
            self.sound.play(loops=-1)  # Loop indefinitely

    def stop(self):
        """Stop the key sound."""
        self.sound.stop()

    def is_playing(self):
        """Check if the key is currently playing."""
        return self.sound.get_num_channels() > 0

    def draw(self, surface):
        """Draw the key on the surface."""
        # Draw key with pressed effect
        if self.is_pressed:
            # Draw pressed key with darker color and shadow
            pygame.draw.rect(surface, DARK_BROWN, (self.x, self.y, self.width, self.height))
            pygame.draw.rect(surface, (80, 40, 10), (self.x, self.y, self.width, self.height), 2)
            # Draw pressed effect
            pygame.draw.rect(surface, (160, 120, 80), (self.x + 2, self.y + 2, self.width - 4, self.height - 4))
        else:
            # Draw normal key with gradient effect
            for i in range(self.height):
                # Create gradient from top to bottom
                gradient_factor = 1.0 - (i / self.height) * 0.3
                r = int(self.color[0] * gradient_factor)
                g = int(self.color[1] * gradient_factor)
                b = int(self.color[2] * gradient_factor)
                pygame.draw.line(surface, (r, g, b), (self.x, self.y + i), (self.x + self.width, self.y + i))
            
            # Draw key outline
            pygame.draw.rect(surface, (100, 70, 40), (self.x, self.y, self.width, self.height), 2)
            
            # Draw key highlight
            pygame.draw.rect(surface, (220, 190, 160), (self.x + 2, self.y + 2, self.width - 4, 10))

        # Draw note name
        font = pygame.font.SysFont('Arial', 14)
        text = font.render(self.note_name, True, WHITE if self.is_pressed else BLACK)
        text_rect = text.get_rect(center=(self.x + self.width // 2, self.y + self.height // 2))
        surface.blit(text, text_rect)

    def check_collision(self, pos):
        """Check if the position collides with this key."""
        x, y = pos
        return (self.x <= x <= self.x + self.width and 
                self.y <= y <= self.y + self.height)

class Harmonium:
    def __init__(self):
        self.keys = []
        self.setup_keys()
        self.bellows_pressure = 0.5  # Default medium pressure
        self.dragging = False
        self.drag_key = None

    def setup_keys(self):
        """Setup the harmonium keys with Indian classical notes."""
        # Frequencies for Indian classical scale (C scale starting from C4)
        notes = [
            ("Sa", 261.63),      # C
            ("Re♭", 277.18),     # C#
            ("Re", 293.66),      # D
            ("Ga♭", 311.13),     # D#
            ("Ga", 329.63),      # E
            ("Ma", 349.23),      # F
            ("Ma#", 369.99),     # F#
            ("Pa", 392.00),      # G
            ("Dha♭", 415.30),    # G#
            ("Dha", 440.00),     # A
            ("Ni♭", 466.16),     # A#
            ("Ni", 493.88),      # B
            ("Sa'", 523.25),     # C (octave)
        ]
        
        # Calculate key dimensions
        total_width = 700
        key_width = total_width // len(notes)
        key_height = 150
        start_x = (WIDTH - total_width) // 2
        start_y = HEIGHT - 200
        
        for i, (note_name, frequency) in enumerate(notes):
            # Alternate colors for visual distinction
            if i % 2 == 0:
                color = LIGHT_BROWN
            else:
                color = (180, 140, 100)  # Slightly different brown
            
            key = HarmoniumKey(
                start_x + i * key_width, start_y,
                key_width, key_height,
                note_name, frequency, color
            )
            self.keys.append(key)

    def handle_mouse_down(self, pos):
        """Handle mouse down event."""
        for key in self.keys:
            if key.check_collision(pos):
                key.is_pressed = True
                key.play()
                self.dragging = True
                self.drag_key = key
                break

    def handle_mouse_up(self):
        """Handle mouse up event."""
        for key in self.keys:
            if key.is_pressed:
                key.is_pressed = False
                key.stop()
        self.dragging = False
        self.drag_key = None

    def handle_mouse_motion(self, pos):
        """Handle mouse motion event."""
        if self.dragging and self.drag_key:
            # Check if we're still over the same key
            if not self.drag_key.check_collision(pos):
                # If we moved off the key, release it
                self.drag_key.is_pressed = False
                self.drag_key.stop()
                self.drag_key = None
                self.dragging = False
        else:
            # Check if we're over any key
            for key in self.keys:
                if key.check_collision(pos):
                    if not key.is_pressed:
                        key.is_pressed = True
                        key.play()
                        self.dragging = True
                        self.drag_key = key
                        break

    def draw(self, surface):
        """Draw the harmonium."""
        # Draw background
        surface.fill((240, 240, 220))  # Light cream background
        
        # Draw harmonium body
        pygame.draw.rect(surface, BROWN, (50, HEIGHT - 250, WIDTH - 100, 200))
        pygame.draw.rect(surface, DARK_BROWN, (50, HEIGHT - 250, WIDTH - 100, 200), 3)
        
        # Draw decorative elements
        pygame.draw.line(surface, (180, 140, 100), (50, HEIGHT - 250), (WIDTH - 50, HEIGHT - 250), 3)
        pygame.draw.line(surface, (180, 140, 100), (50, HEIGHT - 50), (WIDTH - 50, HEIGHT - 50), 3)
        
        # Draw sound holes
        for i in range(6):
            hole_x = 100 + i * 100
            pygame.draw.circle(surface, (60, 40, 20), (hole_x, HEIGHT - 150), 15)
            pygame.draw.circle(surface, (40, 25, 15), (hole_x, HEIGHT - 150), 10)
        
        # Draw bellows pressure indicator
        pygame.draw.rect(surface, (100, 70, 40), (WIDTH // 2 - 100, HEIGHT - 300, 200, 30))
        pygame.draw.rect(surface, (180, 140, 100), (WIDTH // 2 - 100, HEIGHT - 300, 200 * self.bellows_pressure, 30))
        font = pygame.font.SysFont('Arial', 16)
        text = font.render(f"Volume: {int(self.bellows_pressure * 100)}%", True, WHITE)
        surface.blit(text, (WIDTH // 2 - 50, HEIGHT - 295))
        
        # Draw keys
        for key in self.keys:
            key.draw(surface)
        
        # Draw title
        title_font = pygame.font.SysFont('Arial', 24, bold=True)
        title = title_font.render("Interactive Virtual Harmonium", True, (80, 40, 20))
        surface.blit(title, (WIDTH // 2 - title.get_width() // 2, 30))
        
        # Draw instructions
        instr_font = pygame.font.SysFont('Arial', 16)
        instructions = [
            "Click and drag on keys to play notes",
            "Move mouse up/down on left side to control volume",
            "Press Q to quit"
        ]
        
        for i, instruction in enumerate(instructions):
            text = instr_font.render(instruction, True, (100, 100, 100))
            surface.blit(text, (WIDTH // 2 - text.get_width() // 2, 70 + i * 25))

def main():
    clock = pygame.time.Clock()
    harmonium = Harmonium()
    running = True
    
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  # Left mouse button
                    harmonium.handle_mouse_down(event.pos)
            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:  # Left mouse button
                    harmonium.handle_mouse_up()
            elif event.type == pygame.MOUSEMOTION:
                harmonium.handle_mouse_motion(event.pos)
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    running = False
        
        # Update screen
        harmonium.draw(screen)
        pygame.display.flip()
        clock.tick(60)
    
    pygame.quit()

if __name__ == "__main__":
    main()