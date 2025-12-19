"""
Virtual AI-Driven Drum Set
Uses computer vision (MediaPipe) for hand tracking and gesture-based drum control.
No physical drums needed - purely visual interaction.
"""

import cv2
import mediapipe as mp
import pygame
import numpy as np
import time
import math


class Drum:
    """Represents a virtual drum element with position, visuals, and sound."""
    
    def __init__(self, name, x, y, radius, frequency, color, drum_type='circle'):
        """
        Initialize a drum element.
        
        Args:
            name: Display name of the drum
            x, y: Center coordinates
            radius: Drum radius
            frequency: Sound frequency for generated tone
            color: RGB tuple for visual representation
            drum_type: 'circle' for cymbals/toms, 'ellipse' for bass
        """
        self.name = name
        self.x = x
        self.y = y
        self.radius = radius
        self.frequency = frequency
        self.color = color
        self.drum_type = drum_type
        self.hit_color = tuple(min(c + 80, 255) for c in color)  # Brighter on hit
        self.is_hit = False
        self.last_hit_time = 0
        self.hit_intensity = 0  # For visual feedback based on hit strength
        self.cooldown = 0.25  # Reduced cooldown for faster response
        
        # Generate sound dynamically
        self.sound = self._generate_drum_sound(frequency)
        self.sound.set_volume(1.0)
    
    def _generate_drum_sound(self, frequency):
        """Generate highly realistic drum sounds using physical modeling."""
        sample_rate = 44100
        samples = int(sample_rate * 1.0)  # 1 second buffer
        wave = np.zeros(samples)
        
        if frequency < 150:  # Kick/Bass drum - deep thump
            for i in range(samples):
                t = i / sample_rate
                if t > 0.6:
                    break
                # Multiple sine waves with pitch bend for realistic kick
                pitch_bend = frequency * 2 * np.exp(-25 * t)
                attack = 5 * np.exp(-80 * t)  # Sharp click
                body = np.sin(2 * np.pi * (frequency + pitch_bend) * t)
                # Add sub-bass
                sub = 0.6 * np.sin(2 * np.pi * (frequency * 0.5) * t)
                # Envelope
                envelope = np.exp(-6 * t) + attack
                # White noise for beater attack
                click = 0.4 * np.random.randn() * np.exp(-100 * t)
                wave[i] = envelope * (0.5 * body + 0.3 * sub) + click
        
        elif frequency < 250:  # Floor tom - resonant
            for i in range(samples):
                t = i / sample_rate
                if t > 0.8:
                    break
                # Pitch bend downward
                pitch = frequency * (1 + 0.5 * np.exp(-30 * t))
                # Fundamental + harmonics
                tone = np.sin(2 * np.pi * pitch * t)
                tone += 0.3 * np.sin(2 * np.pi * pitch * 2.1 * t)
                tone += 0.15 * np.sin(2 * np.pi * pitch * 3.2 * t)
                # Attack and decay
                envelope = (1 + 3 * np.exp(-50 * t)) * np.exp(-5 * t)
                # Skin resonance
                wave[i] = envelope * tone
        
        elif frequency < 450:  # Snare - complex with rattle
            for i in range(samples):
                t = i / sample_rate
                if t > 0.4:
                    break
                # Drum shell resonance (tonal)
                pitch = frequency * (1 + 0.3 * np.exp(-35 * t))
                shell = 0.4 * np.sin(2 * np.pi * pitch * t)
                shell += 0.2 * np.sin(2 * np.pi * pitch * 1.7 * t)
                
                # Snare wire buzz (noise component)
                rattle = np.random.randn()
                # High-pass filtered noise for wires
                if i > 10:
                    rattle = 0.7 * rattle + 0.3 * wave[i-1]
                
                # Sharp attack
                attack_env = 8 * np.exp(-120 * t)
                sustain_env = np.exp(-12 * t)
                
                wave[i] = (attack_env + sustain_env) * shell + sustain_env * 0.8 * rattle
        
        elif frequency < 650:  # Rack toms
            for i in range(samples):
                t = i / sample_rate
                if t > 0.7:
                    break
                pitch = frequency * (1 + 0.4 * np.exp(-28 * t))
                tone = np.sin(2 * np.pi * pitch * t)
                tone += 0.25 * np.sin(2 * np.pi * pitch * 2.3 * t)
                envelope = (1 + 2.5 * np.exp(-45 * t)) * np.exp(-6 * t)
                wave[i] = envelope * tone
        
        elif frequency > 900:  # Hi-hat - metallic, short
            for i in range(samples):
                t = i / sample_rate
                if t > 0.15:
                    break
                # Complex inharmonic partials for metallic sound
                metallic = 0
                for ratio in [1, 1.34, 1.72, 2.19, 2.67, 3.16, 3.89, 4.42]:
                    metallic += np.sin(2 * np.pi * frequency * ratio * t) / (ratio * 2)
                
                # Band-pass filtered noise for sizzle
                noise = np.random.randn() * 0.6
                
                # Very sharp attack, fast decay
                envelope = (1 + 6 * np.exp(-100 * t)) * np.exp(-25 * t)
                wave[i] = envelope * (0.4 * metallic + 0.6 * noise)
        
        else:  # Crash/Ride cymbals - long sustain
            for i in range(samples):
                t = i / sample_rate
                if t > 2.5:
                    break
                # Many inharmonic partials
                metallic = 0
                for idx, ratio in enumerate([1, 1.29, 1.64, 2.13, 2.68, 3.21, 3.79, 4.32, 5.11]):
                    phase_shift = np.random.rand() * 2 * np.pi
                    metallic += np.sin(2 * np.pi * frequency * ratio * t + phase_shift) / (idx + 2)
                
                # Shimmer noise
                noise = np.random.randn() * 0.3
                
                # Attack with long decay
                attack = 4 * np.exp(-80 * t)
                sustain = np.exp(-1.5 * t)
                
                wave[i] = (attack + sustain) * (0.5 * metallic + 0.5 * noise)
        
        # Compression and normalization
        # Soft clipping for warmth
        wave = np.tanh(wave * 1.5)
        
        # Normalize
        max_val = np.max(np.abs(wave))
        if max_val > 0:
            wave = wave / max_val
        
        # Convert to 16-bit
        wave = np.int16(wave * 32767 * 0.8)
        
        # Stereo
        stereo_wave = np.column_stack((wave, wave))
        
        return pygame.sndarray.make_sound(stereo_wave)
    
    def check_collision(self, x, y):
        """Check if a point (hand position) intersects with this drum."""
        # Circular/elliptical collision detection
        if self.drum_type == 'ellipse':
            # Ellipse collision (for bass drum)
            dx = (x - self.x) / self.radius
            dy = (y - self.y) / (self.radius * 0.6)
            return (dx * dx + dy * dy) <= 1
        else:
            # Circle collision
            distance = math.sqrt((x - self.x) ** 2 + (y - self.y) ** 2)
            return distance <= self.radius
    
    def trigger(self, intensity=1.0):
        """Play sound if cooldown period has elapsed, with velocity-based volume."""
        current_time = time.time()
        if current_time - self.last_hit_time >= self.cooldown:
            if self.sound:
                # Adjust volume based on hit intensity (velocity)
                volume = max(0.3, min(1.0, intensity))  # Clamp between 0.3 and 1.0
                self.sound.set_volume(volume)
                self.sound.play()
            self.is_hit = True
            self.hit_intensity = intensity
            self.last_hit_time = current_time
            return True
        return False
    
    def draw(self, frame):
        """Draw photorealistic drum with textures, shadows, and 3D depth."""
        # Intensify color based on hit strength
        if self.is_hit:
            intensity_factor = int(80 * self.hit_intensity)
            color = tuple(min(c + intensity_factor, 255) for c in self.color)
            glow = True
        else:
            color = self.color
            glow = False
        
        if self.drum_type == 'ellipse':
            # BASS DRUM - Large, deep, wooden shell with kick pad
            axes = (self.radius, int(self.radius * 0.6))
            
            # Drop shadow for depth
            shadow_offset = 8
            cv2.ellipse(frame, (self.x + shadow_offset, self.y + shadow_offset), 
                       (axes[0] + 5, axes[1] + 3), 0, 0, 360, (0, 0, 0), -1)
            cv2.GaussianBlur(frame[self.y-axes[1]-10:self.y+axes[1]+20, 
                            self.x-axes[0]-10:self.x+axes[0]+20], (15, 15), 0)
            
            # Wood shell rim (dark brown)
            for i in range(12, 0, -1):
                shade = int(60 - i * 3)
                cv2.ellipse(frame, (self.x, self.y), (axes[0] + i, axes[1] + i), 
                           0, 0, 360, (shade, shade // 2, shade // 3), -1)
            
            # Drum head (leather texture simulation)
            for layer in range(15, 0, -1):
                lightness = int(color[0] * (0.4 + layer * 0.04))
                head_color = (min(lightness, color[0]), min(lightness//1.2, color[1]), min(lightness//1.5, color[2]))
                cv2.ellipse(frame, (self.x, self.y), 
                           (axes[0] - layer, axes[1] - int(layer * 0.6)), 
                           0, 0, 360, head_color, -1)
            
            # Center kick pad (darker, worn area)
            pad_size = (axes[0] // 3, axes[1] // 3)
            for i in range(8, 0, -1):
                pad_color = tuple(int(c * 0.5) for c in color)
                cv2.ellipse(frame, (self.x, self.y), 
                           (pad_size[0] + i, pad_size[1] + i), 
                           0, 0, 360, pad_color, -1)
            
            # Specular highlight (top-left)
            highlight_pos = (self.x - axes[0] // 3, self.y - axes[1] // 3)
            for i in range(5, 0, -1):
                alpha = 0.3 - i * 0.05
                cv2.ellipse(frame, highlight_pos, (20 + i * 4, 12 + i * 2), 
                           25, 0, 180, (255, 255, 255), -1)
            
            # Metal rim with bolts
            cv2.ellipse(frame, (self.x, self.y), axes, 0, 0, 360, (80, 80, 90), 8)
            cv2.ellipse(frame, (self.x, self.y), (axes[0] - 3, axes[1] - 2), 
                       0, 0, 360, (120, 120, 130), 2)
            
            # Tension bolts around rim
            num_bolts = 10
            for i in range(num_bolts):
                angle = (360 / num_bolts) * i
                bolt_x = int(self.x + axes[0] * 0.9 * np.cos(np.radians(angle)))
                bolt_y = int(self.y + axes[1] * 0.9 * np.sin(np.radians(angle)))
                cv2.circle(frame, (bolt_x, bolt_y), 6, (60, 60, 70), -1)
                cv2.circle(frame, (bolt_x, bolt_y), 4, (90, 90, 100), -1)
                cv2.circle(frame, (bolt_x - 1, bolt_y - 1), 2, (140, 140, 150), -1)
        
        else:
            # CYMBALS AND TOMS - Circular drums
            
            # Drop shadow
            shadow_offset = 6
            cv2.circle(frame, (self.x + shadow_offset, self.y + shadow_offset), 
                      self.radius + 4, (0, 0, 0), -1)
            
            # Determine if it's a cymbal (metallic) or drum (skin)
            is_cymbal = self.frequency > 650
            
            if is_cymbal:
                # CYMBAL - Metallic with grooves and hammering
                # Base metal layers with concentric rings
                for ring in range(20, 0, -1):
                    ring_radius = self.radius - ring * 2
                    if ring_radius < 5:
                        break
                    # Simulate hammered brass texture
                    variation = int(15 * np.sin(ring * 0.8))
                    metal_shade = (min(255, color[0] + variation), 
                                  min(255, color[1] + variation), 
                                  min(255, color[2] + variation))
                    cv2.circle(frame, (self.x, self.y), ring_radius, metal_shade, -1)
                
                # Concentric grooves (lathed lines)
                for groove in range(5, self.radius - 10, 8):
                    cv2.circle(frame, (self.x, self.y), groove, (0, 0, 0), 1)
                
                # Bell (center dome)
                bell_radius = self.radius // 4
                for i in range(bell_radius, 0, -1):
                    brightness = int(200 + (bell_radius - i) * 2)
                    cv2.circle(frame, (self.x, self.y), i, 
                              (min(255, brightness), min(255, brightness - 20), 
                               min(255, brightness - 40)), -1)
                
                # Glossy metallic shine (asymmetric)
                shine_x = self.x - self.radius // 3
                shine_y = self.y - self.radius // 3
                for i in range(6, 0, -1):
                    cv2.ellipse(frame, (shine_x, shine_y), 
                               (self.radius // 3 + i * 3, self.radius // 4 + i * 2), 
                               30, 30, 180, (255, 255, 255), -1)
                
                # Edge - thicker for cymbals
                cv2.circle(frame, (self.x, self.y), self.radius, (40, 40, 30), 5)
                cv2.circle(frame, (self.x, self.y), self.radius - 2, (180, 180, 160), 1)
            
            else:
                # TOM/SNARE - Drum head with tension rods
                # Wooden shell sides
                for i in range(18, 0, -1):
                    wood_shade = 30 + i * 2
                    cv2.circle(frame, (self.x, self.y), self.radius + i, 
                              (wood_shade, wood_shade // 2, wood_shade // 4), -1)
                
                # Drum head texture (mylar/plastic)
                for layer in range(12, 0, -1):
                    luminosity = int(color[0] * (0.6 + layer * 0.03))
                    head_color = (min(luminosity, 255), 
                                 min(int(luminosity * 0.95), color[1]), 
                                 min(int(luminosity * 0.9), color[2]))
                    cv2.circle(frame, (self.x, self.y), self.radius - layer * 2, head_color, -1)
                
                # Center dot (resonance control)
                if self.frequency < 450:  # Snare has obvious dot
                    cv2.circle(frame, (self.x, self.y), self.radius // 5, 
                              tuple(int(c * 0.7) for c in color), -1)
                
                # Specular highlight (curved)
                high_x = self.x - self.radius // 4
                high_y = self.y - self.radius // 4
                for i in range(4, 0, -1):
                    cv2.ellipse(frame, (high_x, high_y), 
                               (self.radius // 3 + i * 2, self.radius // 5 + i), 
                               20, 20, 160, (255, 255, 255), -1)
                
                # Metal hoop (tension ring)
                cv2.circle(frame, (self.x, self.y), self.radius, (70, 70, 80), 6)
                cv2.circle(frame, (self.x, self.y), self.radius - 2, (110, 110, 120), 2)
                
                # Tension lugs around the drum
                num_lugs = 6 if self.radius < 80 else 8
                for i in range(num_lugs):
                    angle = (360 / num_lugs) * i
                    lug_x = int(self.x + (self.radius - 5) * np.cos(np.radians(angle)))
                    lug_y = int(self.y + (self.radius - 5) * np.sin(np.radians(angle)))
                    cv2.circle(frame, (lug_x, lug_y), 5, (50, 50, 60), -1)
                    cv2.circle(frame, (lug_x, lug_y), 3, (80, 80, 90), -1)
                    cv2.circle(frame, (lug_x - 1, lug_y - 1), 1, (120, 120, 130), -1)
        
        # Glow effect when hit
        if glow:
            glow_radius = self.radius if self.drum_type == 'circle' else axes[0]
            overlay = frame.copy()
            if self.drum_type == 'ellipse':
                cv2.ellipse(overlay, (self.x, self.y), 
                           (axes[0] + 15, axes[1] + 10), 
                           0, 0, 360, (255, 255, 200), -1)
            else:
                cv2.circle(overlay, (self.x, self.y), glow_radius + 15, (255, 255, 200), -1)
            cv2.addWeighted(overlay, 0.3 * self.hit_intensity, frame, 0.7, 0, frame)
        
        # Label with professional font styling
        text_size = cv2.getTextSize(self.name, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
        text_x = self.x - text_size[0] // 2
        text_y = self.y + text_size[1] // 2
        
        # Text background for readability
        padding = 8
        cv2.rectangle(frame, 
                     (text_x - padding, text_y - text_size[1] - padding),
                     (text_x + text_size[0] + padding, text_y + padding),
                     (0, 0, 0), -1)
        cv2.rectangle(frame, 
                     (text_x - padding, text_y - text_size[1] - padding),
                     (text_x + text_size[0] + padding, text_y + padding),
                     (255, 255, 255), 2)
        
        # Text with shadow
        cv2.putText(frame, self.name, (text_x + 2, text_y + 2), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        cv2.putText(frame, self.name, (text_x, text_y), 

                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Reset hit visual
        if self.is_hit and time.time() - self.last_hit_time > 0.12:
            self.is_hit = False


class VirtualDrumSet:
    """Main application class for the virtual drum set."""
    
    def __init__(self):
        """Initialize the virtual drum set with camera, hand tracking, and drums."""
        # Initialize Pygame for audio
        pygame.mixer.pre_init(44100, -16, 2, 512)  # Low latency audio
        pygame.init()
        
        # Initialize MediaPipe Hands (AI component for hand pose estimation)
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.6,  # Improved sensitivity
            min_tracking_confidence=0.6     # Smoother tracking
        )
        self.mp_draw = mp.solutions.drawing_utils
        
        # Initialize camera
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        # Get actual frame dimensions
        ret, frame = self.cap.read()
        if ret:
            self.frame_height, self.frame_width = frame.shape[:2]
        else:
            self.frame_width, self.frame_height = 1280, 720
        
        # Create virtual drums with screen positions
        self.drums = self._create_drum_layout()
        
        # Track previous hand positions for velocity calculation
        self.prev_hand_positions = {}
        self.prev_time = time.time()
        
        print("Virtual Drum Set initialized!")
        print("Position your hands over the drums and 'tap' to play.")
        print("Press 'q' to quit.")
    
    def _create_drum_layout(self):
        """Create fullscreen drum kit layout with proper spacing."""
        drums = []
        
        # Use full screen dimensions
        w = self.frame_width
        h = self.frame_height
        
        # Professional drum kit layout (spread across full screen)
        
        # CYMBALS - Top corners and sides (larger, more prominent)
        # Crash Cymbal (top-left)
        drums.append(Drum(
            "CRASH",
            int(w * 0.15), int(h * 0.20),
            100,  # larger radius
            750,
            (255, 223, 0),  # Bright gold
            'circle'
        ))
        
        # Ride Cymbal (top-right)
        drums.append(Drum(
            "RIDE",
            int(w * 0.85), int(h * 0.22),
            110,
            680,
            (192, 192, 192),  # Bright silver
            'circle'
        ))
        
        # Hi-Hat (left side, mid-height)
        drums.append(Drum(
            "HI-HAT",
            int(w * 0.20), int(h * 0.45),
            85,
            1100,
            (255, 250, 150),  # Bright brass
            'circle'
        ))
        
        # TOMS - Arc across top-middle
        # High Tom (left of center)
        drums.append(Drum(
            "TOM 1",
            int(w * 0.35), int(h * 0.25),
            75,
            380,
            (70, 130, 220),  # Deep blue
            'circle'
        ))
        
        # Mid Tom (center-top)
        drums.append(Drum(
            "TOM 2",
            int(w * 0.50), int(h * 0.23),
            80,
            320,
            (50, 110, 200),  # Royal blue
            'circle'
        ))
        
        # Floor Tom (right side)
        drums.append(Drum(
            "FLOOR",
            int(w * 0.75), int(h * 0.55),
            95,
            240,
            (40, 90, 180),  # Navy blue
            'circle'
        ))
        
        # SNARE - Front center (most accessible)
        drums.append(Drum(
            "SNARE",
            int(w * 0.42), int(h * 0.60),
            90,
            420,
            (180, 220, 180),  # Chrome/silver-green
            'circle'
        ))
        
        # BASS DRUM - Bottom center (largest element)
        drums.append(Drum(
            "KICK",
            int(w * 0.50), int(h * 0.82),
            140,  # Much larger
            90,
            (200, 50, 50),  # Deep red
            'ellipse'
        ))
        
        return drums
    
    def process_hand_landmarks(self, hand_landmarks, frame, hand_label):
        """
        Process detected hand landmarks and check for drum hits.
        Uses velocity-based detection for more realistic drumming.
        """
        # Get index finger tip (landmark 8) and middle joint (landmark 6)
        index_tip = hand_landmarks.landmark[8]
        index_mid = hand_landmarks.landmark[6]
        
        # Convert normalized coordinates to pixel coordinates
        tip_x = int(index_tip.x * self.frame_width)
        tip_y = int(index_tip.y * self.frame_height)
        mid_x = int(index_mid.x * self.frame_width)
        mid_y = int(index_mid.y * self.frame_height)
        
        # Calculate velocity for dynamic volume
        current_time = time.time()
        velocity = 0.5  # Default medium velocity
        
        if hand_label in self.prev_hand_positions:
            prev_x, prev_y, prev_time = self.prev_hand_positions[hand_label]
            time_diff = current_time - prev_time
            
            if time_diff > 0:
                # Calculate speed of movement
                distance = math.sqrt((tip_x - prev_x)**2 + (tip_y - prev_y)**2)
                speed = distance / time_diff
                
                # Map speed to intensity (0.3 to 1.0)
                # Fast movements = louder hits
                velocity = min(1.0, max(0.3, speed / 1000))
        
        # Store current position
        self.prev_hand_positions[hand_label] = (tip_x, tip_y, current_time)
        
        # Calculate if finger is pointing down (drumming gesture)
        is_pointing_down = tip_y > mid_y
        
        # Determine hand color (left hand = cyan, right hand = magenta)
        if hand_label == "Right":
            indicator_color = (255, 100, 255)  # Magenta for right hand
        else:
            indicator_color = (100, 255, 255)  # Cyan for left hand
        
        # Draw fingertip indicator with larger size for better visibility
        cv2.circle(frame, (tip_x, tip_y), 18, indicator_color, -1)
        cv2.circle(frame, (tip_x, tip_y), 20, (255, 255, 255), 2)
        
        # Only trigger drums when finger is pointing downward (more realistic)
        if is_pointing_down:
            # Check collision with drums
            for drum in self.drums:
                if drum.check_collision(tip_x, tip_y):
                    drum.trigger(velocity)
    
    def run(self):
        """Main application loop."""
        while True:
            success, frame = self.cap.read()
            if not success:
                print("Failed to read from camera")
                break
            
            # Flip frame horizontally for mirror effect
            frame = cv2.flip(frame, 1)
            
            # Create dark stage background with gradient
            overlay = np.zeros_like(frame)
            # Dark gradient from top to bottom (stage lighting effect)
            for y in range(self.frame_height):
                darkness = int(20 + (y / self.frame_height) * 15)
                overlay[y, :] = [darkness // 3, darkness // 2, darkness]
            
            # Blend camera feed with dark overlay
            frame = cv2.addWeighted(frame, 0.4, overlay, 0.6, 0)
            
            # Convert to RGB for MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process hand tracking
            results = self.hands.process(rgb_frame)
            
            # Draw drums first (background layer)
            for drum in self.drums:
                drum.draw(frame)
            
            # Process hand landmarks if detected
            if results.multi_hand_landmarks:
                for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                    # Determine which hand (left or right)
                    hand_label = "Right"
                    if results.multi_handedness:
                        hand_label = results.multi_handedness[idx].classification[0].label
                    
                    # Draw hand skeleton with thicker lines
                    self.mp_draw.draw_landmarks(
                        frame, 
                        hand_landmarks, 
                        self.mp_hands.HAND_CONNECTIONS,
                        self.mp_draw.DrawingSpec(color=(0, 255, 150), thickness=3, circle_radius=4),
                        self.mp_draw.DrawingSpec(color=(255, 255, 255), thickness=3)
                    )
                    
                    # Process for drum hits
                    self.process_hand_landmarks(hand_landmarks, frame, hand_label)
            
            # Add instructions overlay with dark background
            overlay_height = 100
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (self.frame_width, overlay_height), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
            
            cv2.putText(frame, "AI VIRTUAL DRUM KIT - Point and Tap with Your Index Finger!", 
                       (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
            cv2.putText(frame, "Press 'Q' to Quit", 
                       (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Display frame
            cv2.imshow("Virtual Drum Set", frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
        
        # Cleanup
        self.cleanup()
    
    def cleanup(self):
        """Release resources."""
        self.cap.release()
        cv2.destroyAllWindows()
        pygame.quit()
        print("Virtual Drum Set closed.")


def main():
    """Entry point for the application."""
    try:
        drum_set = VirtualDrumSet()
        drum_set.run()
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
