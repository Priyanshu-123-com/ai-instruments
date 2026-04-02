import cv2
import mediapipe as mp
import pygame
import numpy as np
import time
import math

# --- CONFIGURATION ---
WIDTH = 1280
HEIGHT = 720

# --- AUDIO ENGINE ---
class SoundEngine:
    def __init__(self):
        pygame.mixer.pre_init(44100, -16, 2, 512)
        pygame.mixer.init()
        pygame.mixer.set_num_channels(64) 
        
    def _apply_pan(self, wave, pan):
        left_vol = 1.0 - max(0, pan)
        right_vol = 1.0 + min(0, pan)
        total = left_vol + right_vol
        left_ch = wave * (left_vol/total)
        right_ch = wave * (right_vol/total)
        stereo = np.column_stack((left_ch, right_ch))
        return pygame.sndarray.make_sound((stereo * 32767).astype(np.int16))

    def generate_kick(self, pan=0.0):
        t = np.linspace(0, 0.5, int(44100 * 0.5), False)
        wave = np.sin(2 * np.pi * 150 * np.exp(-15 * t) * t) * np.exp(-5 * t)
        wave[:200] += np.random.uniform(-1, 1, 200) * np.linspace(1, 0, 200)
        return self._apply_pan(np.tanh(wave * 3.0), pan)

    def generate_snare(self, pan=0.0):
        t = np.linspace(0, 0.4, int(44100 * 0.4), False)
        tone = np.sin(2 * np.pi * 180 * t) * np.exp(-15 * t)
        noise = np.random.uniform(-1, 1, len(t)) * np.exp(-12 * t)
        return self._apply_pan(tone * 0.3 + noise * 0.7, pan)
    
    def generate_hihat(self, open=False, pan=0.0):
        duration = 0.3 if open else 0.1
        t = np.linspace(0, duration, int(44100 * duration), False)
        wave = np.random.uniform(-1, 1, len(t)) * np.sin(2 * np.pi * 8000 * t) 
        wave *= np.exp(-15 * t) if open else np.exp(-60 * t)
        return self._apply_pan(wave * 0.4, pan)
        
    def generate_tom(self, freq, pan=0.0):
        t = np.linspace(0, 0.6, int(44100 * 0.6), False)
        wave = np.sin(2 * np.pi * freq * (1 + 0.5 * np.exp(-10 * t)) * t) 
        return self._apply_pan(wave * np.exp(-4 * t), pan)
        
    def generate_cymbal(self, type='crash', pan=0.0):
        duration = 2.0
        t = np.linspace(0, duration, int(44100 * duration), False)
        wave = np.zeros(len(t))
        for f in [300, 500, 800, 1200, 2500, 4000, 8000]:
            wave += np.random.uniform(-1, 1, len(t)) * np.sin(2 * np.pi * f * t)
        return self._apply_pan(wave * np.exp(-3 * t) * 0.15, pan)

# --- 2D DRUM OBJECT ---
class Drum:
    def __init__(self, name, x, y, radius, color, sound):
        self.name = name
        self.pos = np.array([x, y])
        self.radius = radius
        self.base_color = color
        self.sound = sound
        self.hit_time = 0
        self.highlight = 0.0

    def draw(self, surface):
        # Animation Fade
        dt = time.time() - self.hit_time
        if dt < 0.2:
            self.highlight = 1.0 - (dt / 0.2)
        else:
            self.highlight = 0.0
            
        # Draw Base (Transparent)
        s = pygame.Surface((self.radius*2, self.radius*2), pygame.SRCALPHA)
        
        # Color mixing
        r, g, b = self.base_color
        alpha = 100 + int(100 * self.highlight)
        
        # Outer Glow
        pygame.draw.circle(s, (r, g, b, alpha), (self.radius, self.radius), self.radius)
        # Rim
        pygame.draw.circle(s, (255, 255, 255, 200), (self.radius, self.radius), self.radius, 4)
        # Inner Hit Flash
        if self.highlight > 0:
            pygame.draw.circle(s, (255, 255, 255, int(255*self.highlight)), (self.radius, self.radius), int(self.radius * 0.8))
            
        surface.blit(s, (self.pos[0] - self.radius, self.pos[1] - self.radius))
        
        # Label
        font = pygame.font.Font(None, 30)
        txt = font.render(self.name, True, (255, 255, 255))
        surface.blit(txt, (self.pos[0] - txt.get_width()//2, self.pos[1] - txt.get_height()//2))

    def check_hit(self, hand_pos):
        dist = np.linalg.norm(self.pos - hand_pos)
        if dist < self.radius:
            if time.time() - self.hit_time > 0.15: # Debounce
                self.sound.play()
                self.hit_time = time.time()
                return True
        return False

# --- APP ENGINE ---
class VirtualDrum2D:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.DOUBLEBUF | pygame.HWSURFACE)
        pygame.display.set_caption("AETHER 2D - AR DRUMS")
        self.clock = pygame.time.Clock()
        self.sound = SoundEngine()
        
        self.cap = cv2.VideoCapture(0)
        self.cap.set(3, WIDTH)
        self.cap.set(4, HEIGHT)
        self.hands = mp.solutions.hands.Hands(max_num_hands=2, min_detection_confidence=0.7)
        
        self.drums = []
        self._build_layout()
        
    def _build_layout(self):
        # Colors
        RED   = (200, 50, 50)
        BLUE  = (50, 50, 200)
        GOLD  = (200, 180, 50)
        GREEN = (50, 200, 50)
        
        # Layout (Screen Coordinates)
        cx, cy = WIDTH // 2, HEIGHT // 2
        
        # Kick (Center Bottom)
        self.drums.append(Drum("KICK", cx, HEIGHT - 100, 90, RED, self.sound.generate_kick()))
        
        # Snare (Left Center)
        self.drums.append(Drum("SNARE", cx - 250, cy + 100, 70, BLUE, self.sound.generate_snare(-0.3)))
        
        # Floor Tom (Right Center)
        self.drums.append(Drum("FLOOR", cx + 250, cy + 100, 80, RED, self.sound.generate_tom(100, 0.3)))
        
        # Toms (Top Center)
        self.drums.append(Drum("TOM1", cx - 100, cy - 50, 60, RED, self.sound.generate_tom(150, -0.2)))
        self.drums.append(Drum("TOM2", cx + 100, cy - 50, 60, RED, self.sound.generate_tom(130, 0.2)))
        
        # Cymbals (Top Corners)
        self.drums.append(Drum("HH", cx - 350, cy - 100, 55, GOLD, self.sound.generate_hihat(False, -0.8)))
        self.drums.append(Drum("CRASH", cx - 200, cy - 200, 65, GOLD, self.sound.generate_cymbal('crash', -0.5)))
        self.drums.append(Drum("RIDE", cx + 350, cy - 100, 70, GOLD, self.sound.generate_cymbal('ride', 0.8)))

    def run(self):
        while True:
            # 1. Event Handling
            if pygame.event.get(pygame.QUIT): break
            
            # 2. Capture
            ret, frame = self.cap.read()
            if not ret: break
            frame = cv2.flip(frame, 1)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # 3. Vision Processing
            results = self.hands.process(frame_rgb)
            
            # 4. Render Background (Webcam)
            # Convert cv2 frame to pygame surface
            frame_t = frame.transpose(1, 0, 2)
            frame_s = pygame.surfarray.make_surface(frame_t)
            self.screen.blit(frame_s, (0, 0))
            
            # Darken background slightly for contrast
            dark = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
            dark.fill((0, 0, 0, 50))
            self.screen.blit(dark, (0,0))
            
            # 5. Hand Logic & Hit Detection
            if results.multi_hand_landmarks:
                for h in results.multi_hand_landmarks:
                    # Index Finger Tip
                    tip = h.landmark[8]
                    tx, ty = int(tip.x * WIDTH), int(tip.y * HEIGHT)
                    t_pos = np.array([tx, ty])
                    
                    # Draw Stick Tip/Finger
                    pygame.draw.circle(self.screen, (100, 255, 100), (tx, ty), 15)
                    pygame.draw.circle(self.screen, (255, 255, 255), (tx, ty), 8)
                    
                    # Check Drums
                    for d in self.drums:
                        if d.check_hit(t_pos):
                            # Visual Flare on hit
                            pygame.draw.line(self.screen, (255, 255, 255), (tx-20, ty), (tx+20, ty), 2)
                            pygame.draw.line(self.screen, (255, 255, 255), (tx, ty-20), (tx, ty+20), 2)
            
            # 6. Draw Drums (Overlay)
            for d in self.drums:
                d.draw(self.screen)
            
            # FPS
            fps = self.clock.get_fps()
            font = pygame.font.Font(None, 30)
            fps_t = font.render(f"FPS: {int(fps)}", True, (200, 255, 200))
            self.screen.blit(fps_t, (20, 20))
            
            pygame.display.flip()
            self.clock.tick(60)
            
        self.cap.release()
        pygame.quit()

if __name__ == "__main__":
    VirtualDrum2D().run()
