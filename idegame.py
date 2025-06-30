import cv2
import numpy as np
import random
import time
import math
import mediapipe as mp
import pygame
import os

class MathBalloonGame:
    def __init__(self):
        # Game constants - Optimized for hand tracking!
        self.WINDOW_WIDTH = 1280  # 16:9 ratio, good for most webcams
        self.WINDOW_HEIGHT = 720  # HD resolution, optimal for hand detection
        self.MAX_LEVEL = 5
        self.BALLOONS_PER_LEVEL = 8
        self.MAX_HEARTS = 10  # Player starts with 10 hearts/lives
        
        # Game states
        self.GAME_STATE_MENU = 0
        self.GAME_STATE_COUNTDOWN = 1
        self.GAME_STATE_PLAYING = 2
        self.GAME_STATE_LEVEL_COMPLETE = 3
        self.GAME_STATE_GAME_OVER = 4
        
        # 🎨 HOMOGENEOUS COLOR SCHEME
        self.COLOR_THEME = {
            'primary': (120, 80, 40),      # Deep blue
            'secondary': (80, 60, 30),     # Darker blue  
            'accent': (0, 200, 255),       # Orange/yellow
            'success': (100, 200, 100),    # Soft green
            'danger': (100, 100, 200),     # Soft red
            'neutral': (150, 150, 150),    # Gray
            'background': (40, 30, 20),    # Dark blue background
            'text_light': (255, 255, 255), # White
            'text_dark': (50, 50, 50),     # Dark gray
            'ui_overlay': (100, 80, 60)    # Semi-transparent blue
        }
        
        # 📚 LEVEL TARGET NUMBERS - Easy to customize!
        self.LEVEL_TARGETS = {
            1: 5,   # Level 1: Find equations that equal 5
            2: 8,   # Level 2: Find equations that equal 8  
            3: 12,  # Level 3: Find equations that equal 12
            4: 15,  # Level 4: Find equations that equal 15
            5: 20   # Level 5: Find equations that equal 20
        }
        
        # 🧮 EQUATION POOL - Add your own equations here!
        self.EQUATION_POOL = {
            # Equations that equal 5
            5: ['3 + 2', '8 - 3', '1 + 4', '10 - 5', '2 + 3', '9 - 4', '6 - 1', '15 / 3'],
            
            # Equations that equal 8  
            8: ['4 + 4', '10 - 2', '2 * 4', '16 / 2', '5 + 3', '12 - 4', '6 + 2', '9 - 1'],
            
            # Equations that equal 12
            12: ['6 + 6', '3 * 4', '15 - 3', '24 / 2', '8 + 4', '20 - 8', '7 + 5', '14 - 2'],
            
            # Equations that equal 15
            15: ['10 + 5', '3 * 5', '18 - 3', '30 / 2', '9 + 6', '22 - 7', '8 + 7', '20 - 5'],
            
            # Equations that equal 20
            20: ['10 + 10', '4 * 5', '25 - 5', '40 / 2', '12 + 8', '30 - 10', '15 + 5', '22 - 2'],
            
            # Wrong answers (these will be mixed in)
            'wrong': ['2 + 2', '3 * 3', '7 - 2', '6 + 1', '4 * 2', '18 / 3', '11 - 3', '5 + 5',
                     '9 + 2', '13 - 2', '6 * 2', '21 / 3', '8 + 3', '16 - 2', '4 + 5', '12 - 1',
                     '2 * 6', '15 / 5', '7 + 4', '18 - 7', '3 + 4', '14 - 6', '5 * 2', '16 / 4']
        }
        
        # Game state
        self.current_level = 1
        self.current_state = self.GAME_STATE_MENU
        self.correct_catches = 0
        self.level_score = 0
        self.total_score = 0
        self.hearts = self.MAX_HEARTS  # Player's remaining lives
        
        # Level tracking
        self.correct_balloons_in_level = 0  # How many correct balloons exist in current level
        self.correct_balloons_caught = 0    # How many correct balloons user has caught
        
        # Level timing
        self.countdown_timer = 0
        self.countdown_duration = 3000  # 3 seconds
        self.level_complete_timer = 0
        self.level_complete_duration = 3000  # 3 seconds
        
        # Balloons
        self.balloons = []
        self.balloons_spawned = False
        
        # Hand tracking
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Finger trail
        self.finger_trail = []
        self.max_trail_length = 20
        
        # Initialize pygame for sound
        pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=512)
        self.sounds = self.load_sounds()
        self.sprites = self.load_sprites()
        
        # 🖼️ BACKGROUND IMAGE SYSTEM
        self.background_image = self.load_background_image()
        
        # Background music control
        self.background_music_playing = False

    def load_sounds(self):
        """Load sound effects and background music"""
        sounds = {}
        try:
            # 🎵 BACKGROUND MUSIC (replace with your own file!)
            sounds['background'] = pygame.mixer.Sound('background_music.wav')  # Replace with: pygame.mixer.Sound('background_music.wav')
            
            # 🔊 SOUND EFFECTS (replace with your own files!)
            sounds['pop_correct'] = pygame.mixer.Sound('correct_pop.wav')   # Replace with: pygame.mixer.Sound('correct_pop.wav')
            sounds['pop_wrong'] = pygame.mixer.Sound('wrong_pop.wav')     # Replace with: pygame.mixer.Sound('wrong_pop.wav')
            sounds['level_complete'] = None # Replace with: pygame.mixer.Sound('level_complete.wav')
            sounds['game_complete'] = None  # Replace with: pygame.mixer.Sound('game_complete.wav')
            sounds['game_over'] = None      # Replace with: pygame.mixer.Sound('game_over.wav')
            sounds['countdown'] = None      # Replace with: pygame.mixer.Sound('countdown.wav')
            
            print("🎵 Audio files loaded successfully!")
            print("📁 Place your audio files in the same folder as this script:")
            print("   - background_music.wav (looping background music)")
            print("   - correct_pop.wav (correct balloon pop)")
            print("   - wrong_pop.wav (wrong balloon pop)")
            print("   - level_complete.wav (level finish)")
            print("   - game_complete.wav (game finish)")
            print("   - game_over.wav (no hearts left)")
            print("   - countdown.wav (ready, set, go sound)")
            
        except Exception as e:
            print(f"🔇 Audio loading failed: {e}")
            print("Running without sound...")
            sounds = {key: None for key in ['background', 'pop_correct', 'pop_wrong', 
                     'level_complete', 'game_complete', 'game_over', 'countdown']}
        return sounds
    
    def load_background_image(self):
        """Load background image - you can customize this!"""
        try:
            # 🖼️ LOAD YOUR BACKGROUND IMAGE (replace with your own!)
            background = cv2.imread('game_background.png')
            
            if background is not None:
                # Resize background to fit game window
                background = cv2.resize(background, (self.WINDOW_WIDTH, self.WINDOW_HEIGHT))
                print("🖼️ Background image 'game_background.png' loaded successfully!")
                return background
            else:
                print("🖼️ Background image 'game_background.png' not found")
                return None
                
        except Exception as e:
            print(f"🖼️ Failed to load background image: {e}")
            return None
    
    def get_background_frame(self, camera_frame):
        """Get the background - full custom background like a real game!"""
        if self.background_image is not None:
            # 🎮 USE 100% CUSTOM BACKGROUND - like a real professional game!
            return self.background_image.copy()
        else:
            # Fallback: use camera feed with light overlay
            frame = cv2.flip(camera_frame, 1)
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (frame.shape[1], frame.shape[0]), 
                         self.COLOR_THEME['background'], -1)
            cv2.addWeighted(overlay, 0.1, frame, 0.9, 0, frame)
            return frame
    
    def load_sprites(self):
        """Load custom balloon sprites and heart sprites"""
        sprites = {}
        
        # 🎈 LOAD CUSTOM BALLOON SPRITES (your PNG files!)
        try:
            # Load balloon animation frames
            balloon1 = cv2.imread('balloon1.png', cv2.IMREAD_UNCHANGED)  # Normal balloon
            balloon2 = cv2.imread('balloon2.png', cv2.IMREAD_UNCHANGED)  # Pop frame 1
            balloon3 = cv2.imread('balloon3.png', cv2.IMREAD_UNCHANGED)  # Pop frame 2  
            balloon4 = cv2.imread('balloon4.png', cv2.IMREAD_UNCHANGED)  # Pop frame 3
            
            if balloon1 is not None and balloon2 is not None and balloon3 is not None and balloon4 is not None:
                # Resize balloons to standard size (80x80)
                sprites['balloon1'] = cv2.resize(balloon1, (160, 160))
                sprites['balloon2'] = cv2.resize(balloon2, (160, 160))
                sprites['balloon3'] = cv2.resize(balloon3, (160, 160))
                sprites['balloon4'] = cv2.resize(balloon4, (160, 160))
                print("🎈 Custom balloon sprites loaded successfully!")
                
            else:
                print("⚠️ Custom balloon PNGs not found, creating default balloons...")
                # Fallback: create default balloon sprite
                balloon_color = self.COLOR_THEME['accent']
                balloon_sprite = np.zeros((80, 80, 3), dtype=np.uint8)
                cv2.circle(balloon_sprite, (40, 40), 35, balloon_color, -1)
                cv2.circle(balloon_sprite, (40, 40), 35, self.COLOR_THEME['text_light'], 3)
                cv2.circle(balloon_sprite, (30, 30), 10, tuple(int(c * 1.3) for c in balloon_color), -1)
                
                # Use same sprite for all frames
                sprites['balloon1'] = balloon_sprite
                sprites['balloon2'] = balloon_sprite
                sprites['balloon3'] = balloon_sprite
                sprites['balloon4'] = balloon_sprite
                
        except Exception as e:
            print(f"⚠️ Error loading balloon sprites: {e}")
            # Create fallback sprites
            balloon_color = self.COLOR_THEME['accent']
            balloon_sprite = np.zeros((80, 80, 3), dtype=np.uint8)
            cv2.circle(balloon_sprite, (40, 40), 35, balloon_color, -1)
            cv2.circle(balloon_sprite, (40, 40), 35, self.COLOR_THEME['text_light'], 3)
            cv2.circle(balloon_sprite, (30, 30), 10, tuple(int(c * 1.3) for c in balloon_color), -1)
            
            sprites['balloon1'] = balloon_sprite
            sprites['balloon2'] = balloon_sprite
            sprites['balloon3'] = balloon_sprite
            sprites['balloon4'] = balloon_sprite
        
        # 💖 CREATE HEART SPRITES (you can replace these with your own images!)
        # Full heart sprite
        full_heart = np.zeros((30, 30, 3), dtype=np.uint8)
        cv2.circle(full_heart, (10, 12), 8, (0, 0, 255), -1)  # Left heart bump
        cv2.circle(full_heart, (20, 12), 8, (0, 0, 255), -1)  # Right heart bump
        pts = np.array([[15, 25], [5, 15], [25, 15]], np.int32)
        cv2.fillPoly(full_heart, [pts], (0, 0, 255))
        sprites['heart_full'] = full_heart
        
        # Broken heart sprite
        broken_heart = np.zeros((30, 30, 3), dtype=np.uint8)
        cv2.circle(broken_heart, (10, 12), 8, (100, 100, 100), -1)
        cv2.circle(broken_heart, (20, 12), 8, (100, 100, 100), -1)
        pts = np.array([[15, 25], [5, 15], [25, 15]], np.int32)
        cv2.fillPoly(broken_heart, [pts], (100, 100, 100))
        cv2.line(broken_heart, (15, 8), (15, 22), (50, 50, 50), 2)
        sprites['heart_broken'] = broken_heart
        
        return sprites
    
    def play_sound(self, sound_name):
        """Play a sound effect"""
        if sound_name in self.sounds and self.sounds[sound_name]:
            try:
                self.sounds[sound_name].play()
            except:
                pass
    
    def start_background_music(self):
        """Start looping background music"""
        if not self.background_music_playing and self.sounds['background']:
            try:
                pygame.mixer.Sound.play(self.sounds['background'], loops=-1)  # -1 = infinite loop
                pygame.mixer.music.set_volume(0.3)  # Set volume to 30%
                self.background_music_playing = True
                print("🎵 Background music started!")
            except:
                print("🔇 Failed to start background music")
    
    def stop_background_music(self):
        """Stop background music"""
        if self.background_music_playing:
            try:
                pygame.mixer.stop()
                self.background_music_playing = False
                print("🔇 Background music stopped!")
            except:
                pass

    def start_level(self):
        """Start a new level with countdown"""
        self.current_state = self.GAME_STATE_COUNTDOWN
        self.countdown_timer = time.time() * 1000
        self.balloons = []
        self.balloons_spawned = False
        self.correct_catches = 0  # Reset catches for new level
        self.level_score = 0
        self.correct_balloons_in_level = 0  # Reset level counters
        self.correct_balloons_caught = 0    # Reset level counters

    def spawn_level_balloons(self):
        """Spawn exactly 8 balloons with GUARANTEED correct equations"""
        if self.current_level not in self.LEVEL_TARGETS:
            return
            
        target_number = self.LEVEL_TARGETS[self.current_level]
        self.correct_balloons_in_level = 0  # Reset counter
        
        print(f"\n🎯 SPAWNING LEVEL {self.current_level} - TARGET: {target_number}")
        # 🎯 GUARANTEE 3-4 CORRECT EQUATIONS (no more empty levels!)
        correct_count = random.randint(3, 4)  # Increased minimum
        correct_equations = []
        
        # Make sure we have enough correct equations
        if target_number in self.EQUATION_POOL and len(self.EQUATION_POOL[target_number]) >= correct_count:
            correct_equations = random.sample(self.EQUATION_POOL[target_number], correct_count)
            print(f"✅ Selected {len(correct_equations)} correct equations from pool")
        else:
            # Fallback: generate basic correct equations if pool is insufficient
            print(f"⚠️ Generating fallback equations for target {target_number}")
            correct_equations = self.generate_fallback_equations(target_number, correct_count)
        
        # DEBUG: Verify each correct equation
        print("🔍 VERIFYING CORRECT EQUATIONS:")
        verified_correct = []
        for eq in correct_equations:
            result = self.evaluate_equation(eq)
            print(f"   {eq} = {result} (target: {target_number}) -> {'✅ CORRECT' if result == target_number else '❌ WRONG!'}")
            if result == target_number:
                verified_correct.append(eq)
        
        # Fill the rest with wrong equations
        wrong_count = self.BALLOONS_PER_LEVEL - len(verified_correct)
        wrong_equations = random.sample(self.EQUATION_POOL['wrong'], 
                                       min(wrong_count, len(self.EQUATION_POOL['wrong'])))
        
        # DEBUG: Verify wrong equations don't accidentally equal target
        print("🔍 VERIFYING WRONG EQUATIONS:")
        verified_wrong = []
        for eq in wrong_equations:
            result = self.evaluate_equation(eq)
            if result != target_number:
                verified_wrong.append(eq)
                print(f"   {eq} = {result} -> ✅ WRONG (good)")
            else:
                print(f"   {eq} = {result} -> ⚠️ ACCIDENTALLY CORRECT! Skipping...")
        
        # Combine and shuffle
        all_equations = verified_correct + verified_wrong
        
        # If we don't have enough equations, generate more
        while len(all_equations) < self.BALLOONS_PER_LEVEL:
            fallback_eq = self.generate_simple_wrong_equation(target_number)
            all_equations.append(fallback_eq)
            print(f"🔧 Added fallback wrong equation: {fallback_eq}")
        
        # Take only 8 equations
        all_equations = all_equations[:self.BALLOONS_PER_LEVEL]
        random.shuffle(all_equations)
        
        # 🎈 BETTER SPACING SYSTEM - No more collision!
        lane_width = (self.WINDOW_WIDTH - 160) // self.BALLOONS_PER_LEVEL
        
        print("🎈 FINAL BALLOON ASSIGNMENT:")
        for i, equation in enumerate(all_equations):
            # Double-check if equation is correct by evaluating it
            result = self.evaluate_equation(equation)
            is_correct = result == target_number
            
            print(f"   Balloon {i+1}: {equation} = {result} -> {'🟢 CORRECT' if is_correct else '🔴 WRONG'}")
            
            if is_correct:
                self.correct_balloons_in_level += 1  # Count correct balloons
            
            # Assign each balloon to its own lane
            lane_center = 80 + (i * lane_width) + (lane_width // 2)
            x = lane_center + random.randint(-30, 30)
            start_y = -80 - (i * 40)
            
            balloon = Balloon(x, equation, is_correct, self.WINDOW_WIDTH, self.WINDOW_HEIGHT, 
                            self.sprites, start_y)
            self.balloons.append(balloon)
        
        print(f"📊 FINAL RESULT: {self.correct_balloons_in_level} correct balloons out of {len(all_equations)}")
        print("="*50)
        self.balloons_spawned = True

    def generate_simple_wrong_equation(self, target):
        """Generate a simple wrong equation"""
        wrong_result = target + random.randint(1, 5)  # Make sure it's different from target
        a = random.randint(1, 10)
        return f"{a} + {wrong_result - a}"

    def generate_fallback_equations(self, target, count):
        """Generate basic correct equations as fallback"""
        equations = []
        for i in range(count):
            if target >= 2:
                # Simple addition
                a = random.randint(1, target - 1)
                b = target - a
                equations.append(f"{a} + {b}")
            if len(equations) < count and target >= 1:
                # Simple subtraction
                a = target + random.randint(1, 10)
                equations.append(f"{a} - {target}")
        return equations[:count]

    def evaluate_equation(self, equation_str):
        """Safely evaluate a math equation string"""
        try:
            # Replace division symbol and handle properly
            safe_equation = equation_str.replace('/', '/')  # Keep division as is
            safe_equation = safe_equation.replace('*', '*')  # Keep multiplication as is
            result = eval(safe_equation)
            # Handle division results - round to nearest integer for game logic
            return round(result) if isinstance(result, float) else result
        except Exception as e:
            print(f"⚠️ Error evaluating equation '{equation_str}': {e}")
            return 0

    def update_game(self, current_time):
        """Update game state based on current state"""
        if self.current_state == self.GAME_STATE_COUNTDOWN:
            # Handle countdown
            if current_time - self.countdown_timer >= self.countdown_duration:
                self.current_state = self.GAME_STATE_PLAYING
                self.spawn_level_balloons()
        
        elif self.current_state == self.GAME_STATE_PLAYING:
            # Update balloons
            balloons_still_falling = 0
            
            for balloon in self.balloons:
                balloon.update(current_time)
                # Count balloons still on screen (visible area) and not dead
                if balloon.state != "dead" and balloon.y < self.WINDOW_HEIGHT:
                    balloons_still_falling += 1
            
            # 🎯 INSTANT LEVEL COMPLETION - Go to next level immediately when all correct balloons caught!
            if self.correct_balloons_caught >= self.correct_balloons_in_level and self.balloons_spawned:
                print(f"🎉 Level {self.current_level} completed! Caught {self.correct_balloons_caught}/{self.correct_balloons_in_level} correct balloons")
                self.current_state = self.GAME_STATE_LEVEL_COMPLETE
                self.level_complete_timer = current_time
                self.total_score += self.correct_balloons_caught
            # Fallback: Level complete when all balloons have fallen (in case user missed some)
            elif balloons_still_falling == 0 and self.balloons_spawned:
                print(f"⏰ Level {self.current_level} ended - all balloons fell. Caught {self.correct_balloons_caught}/{self.correct_balloons_in_level}")
                self.current_state = self.GAME_STATE_LEVEL_COMPLETE
                self.level_complete_timer = current_time
                self.total_score += self.correct_balloons_caught
        
        elif self.current_state == self.GAME_STATE_LEVEL_COMPLETE:
            # Handle level complete
            if current_time - self.level_complete_timer >= self.level_complete_duration:
                if self.current_level >= self.MAX_LEVEL:
                    self.current_state = self.GAME_STATE_GAME_OVER
                    self.play_sound('game_complete')
                else:
                    self.current_level += 1
                    self.start_level()

class Balloon:
    def __init__(self, x, equation, is_correct, game_width, game_height, sprites, start_y=-50):
        self.x = x
        self.y = start_y  # Start at specified height
        self.equation = equation
        self.is_correct = is_correct
        self.radius = 40
        self.speed = 1.5 + random.uniform(0, 0.5)
        self.game_width = game_width
        self.game_height = game_height
        self.alive = True
        self.sprites = sprites
        
        # 🎈 ANIMATION SYSTEM
        self.state = "falling"  # States: "falling", "popping", "dead"
        self.animation_frame = 1  # Current animation frame (1-4)
        self.animation_timer = 0
        self.animation_frame_duration = 100  # milliseconds per frame
        self.pop_animation_started = False
    
    def update(self, current_time):
        """Update balloon position and animation"""
        if self.state == "falling":
            # Normal falling behavior
            if self.alive:
                self.y += self.speed
                if self.y > self.game_height + 50:
                    self.alive = False
        
        elif self.state == "popping":
            # 🎈 POP ANIMATION - cycle through frames 2, 3, 4 then disappear
            if not self.pop_animation_started:
                self.animation_timer = current_time
                self.animation_frame = 2  # Start with balloon2.png
                self.pop_animation_started = True
            
            # Check if it's time to advance animation frame
            if current_time - self.animation_timer >= self.animation_frame_duration:
                self.animation_frame += 1
                self.animation_timer = current_time
                
                # Animation complete - balloon disappears
                if self.animation_frame > 4:
                    self.state = "dead"
                    self.alive = False
    
    def start_pop_animation(self):
        """Start the popping animation sequence"""
        if self.state == "falling":
            self.state = "popping"
            self.pop_animation_started = False  # Will be set to True in update()
    
    def draw(self, frame, color_theme):
        """Draw the balloon with appropriate sprite based on animation state"""
        if self.state == "dead" or not self.alive:
            return  # Don't draw dead balloons
        
        # Choose sprite based on animation state and frame
        sprite_key = f"balloon{self.animation_frame}"
        
        if sprite_key in self.sprites and self.y > -40:  # Only draw when visible
            sprite = self.sprites[sprite_key]
            sprite_h, sprite_w = sprite.shape[:2]
            
            start_x = int(self.x - sprite_w // 2)
            start_y = int(self.y - sprite_h // 2)
            
            if (start_x >= 0 and start_y >= 0 and 
                start_x + sprite_w < frame.shape[1] and 
                start_y + sprite_h < frame.shape[0]):
                
                # Handle transparency if PNG has alpha channel
                if sprite.shape[2] == 4:  # RGBA
                    # Extract RGB and alpha channels
                    sprite_rgb = sprite[:, :, :3]
                    alpha = sprite[:, :, 3] / 255.0
                    
                    # Blend with background
                    sprite_area = frame[start_y:start_y + sprite_h, start_x:start_x + sprite_w]
                    for c in range(3):
                        sprite_area[:, :, c] = (alpha * sprite_rgb[:, :, c] + 
                                               (1 - alpha) * sprite_area[:, :, c])
                else:
                    # Regular RGB blending
                    sprite_area = frame[start_y:start_y + sprite_h, start_x:start_x + sprite_w]
                    cv2.addWeighted(sprite, 0.9, sprite_area, 0.1, 0, sprite_area)
            
            # Only draw equation text for falling balloons (not during pop animation)
            if self.state == "falling":
                font = cv2.FONT_HERSHEY_SIMPLEX
                text_size = cv2.getTextSize(self.equation, font, 0.6, 2)[0]
                text_x = int(self.x - text_size[0] // 2)
                text_y = int(self.y + text_size[1] // 2)
                
                cv2.rectangle(frame, 
                             (text_x - 5, text_y - text_size[1] - 5),
                             (text_x + text_size[0] + 5, text_y + 5),
                             color_theme['ui_overlay'], -1)
                cv2.putText(frame, self.equation, (text_x, text_y), 
                           font, 0.6, color_theme['text_light'], 2)
    
    def is_touched_by_finger(self, finger_x, finger_y):
        """Check if balloon was touched by finger (only when falling)"""
        if self.state != "falling" or not self.alive or self.y < -40:
            return False
        distance = math.sqrt((finger_x - self.x)**2 + (finger_y - self.y)**2)
        return distance <= self.radius

def main():
    game = MathBalloonGame()
    
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, game.WINDOW_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, game.WINDOW_HEIGHT)
    
    print("🎮 WAVE-BASED MATH BALLOON GAME")
    print("="*40)
    print("📖 HOW IT WORKS:")
    print("   Each level shows a target number")
    print("   8 balloons fall with different equations")
    print("   Pop balloons that equal the target number!")
    print("   After all balloons fall, next level starts")
    print("")
    print("🖼️ CUSTOMIZATION FILES:")
    print("   Place 'game_background.png' for custom background")
    print("   Place balloon1.png, balloon2.png, balloon3.png, balloon4.png for animated balloons!")
    print("   Place 'balloon_pop.wav' for satisfying pop sound effect")
    print("")
    print("🎮 CONTROLS:")
    print("👆 Point your finger at correct balloons!")
    print("⌨️ Press SPACE to start, 'q' to quit, 'r' to restart")
    print("🎵 Press 'M' to toggle background music on/off")
    print("")
    print("💖 LIVES SYSTEM:")
    print("   Start with 10 hearts - lose 1 for each wrong balloon!")
    print("   Game over when all hearts are gone!")
    
    while True:
        ret, camera_frame = cap.read()
        if not ret:
            break
        
        # 🖼️ USE CUSTOM BACKGROUND OR CAMERA FEED
        background_frame = game.get_background_frame(camera_frame)
        frame = background_frame.copy()
        
        current_time = time.time() * 1000
        
        # Detect hands (works on camera feed even with custom background)
        rgb_frame = cv2.cvtColor(cv2.flip(camera_frame, 1), cv2.COLOR_BGR2RGB)
        results = game.hands.process(rgb_frame)
        
        finger_tip = None
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                h, w, _ = frame.shape
                finger_tip = (
                    int(hand_landmarks.landmark[8].x * w),
                    int(hand_landmarks.landmark[8].y * h)
                )
                
                game.finger_trail.append(finger_tip)
                if len(game.finger_trail) > game.max_trail_length:
                    game.finger_trail.pop(0)
                
                game.mp_drawing.draw_landmarks(
                    frame, hand_landmarks, game.mp_hands.HAND_CONNECTIONS)
        
        # Update game
        game.update_game(current_time)
        
        # Handle finger collision during gameplay
        if game.current_state == game.GAME_STATE_PLAYING and finger_tip:
            for balloon in game.balloons[:]:
                if balloon.is_touched_by_finger(finger_tip[0], finger_tip[1]):
                    # 🎈 START POP ANIMATION instead of immediate removal!
                    balloon.start_pop_animation()
                    
                    # Play balloon pop sound
                    game.play_sound('balloon_pop')
                    
                    if balloon.is_correct:
                        game.correct_balloons_caught += 1  # Track correct catches for instant completion
                        game.correct_catches += 1  # Keep old counter for display
                        game.play_sound('pop_correct')
                        print(f"✅ Correct! ({game.correct_balloons_caught}/{game.correct_balloons_in_level})")
                    else:
                        # 💔 LOSE A HEART for wrong balloon!
                        game.hearts -= 1
                        game.play_sound('pop_wrong')
                        print(f"❌ Wrong! Hearts left: {game.hearts}")
                        
                        # Check for game over
                        if game.hearts <= 0:
                            game.current_state = game.GAME_STATE_GAME_OVER
                            game.play_sound('game_over')
                    break
        
        # Draw balloons
        for balloon in game.balloons:
            balloon.draw(frame, game.COLOR_THEME)
        
        # Draw finger trail
        draw_finger_trail(frame, game)
        
        # Draw UI based on game state
        draw_ui(frame, game, current_time)
        
        cv2.imshow('Wave-Based Math Balloon Game', frame)
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' ') and game.current_state == game.GAME_STATE_MENU:
            game.start_level()
            game.start_background_music()  # 🎵 Start music when game begins!
        elif key == ord('r') and game.current_state == game.GAME_STATE_GAME_OVER:
            restart_game(game)
        elif key == ord('m'):  # 🎵 Toggle music on/off with 'M' key
            if game.background_music_playing:
                game.stop_background_music()
            else:
                game.start_background_music()
    
    cap.release()
    cv2.destroyAllWindows()

def draw_finger_trail(frame, game):
    """Draw finger trail"""
    if len(game.finger_trail) > 1:
        for i in range(1, len(game.finger_trail)):
            alpha = i / len(game.finger_trail)
            thickness = int(alpha * 8) + 1
            cv2.line(frame, game.finger_trail[i-1], game.finger_trail[i], 
                    game.COLOR_THEME['accent'], thickness)

def draw_ui(frame, game, current_time):
    """Draw UI based on game state"""
    font = cv2.FONT_HERSHEY_SIMPLEX
    center_x = game.WINDOW_WIDTH // 2
    center_y = game.WINDOW_HEIGHT // 2
    
    if game.current_state == game.GAME_STATE_MENU:
        # Main menu
        cv2.putText(frame, "MATH BALLOON GAME", (center_x - 200, center_y - 100), 
                   font, 1.5, game.COLOR_THEME['accent'], 3)
        cv2.putText(frame, "Press SPACE to start", (center_x - 150, center_y), 
                   font, 1, game.COLOR_THEME['text_light'], 2)
        cv2.putText(frame, "Point finger at balloons that equal the target!", 
                   (center_x - 250, center_y + 50), font, 0.7, game.COLOR_THEME['neutral'], 2)
    
    elif game.current_state == game.GAME_STATE_COUNTDOWN:
        # Countdown screen
        target = game.LEVEL_TARGETS[game.current_level]
        
        cv2.putText(frame, f"LEVEL {game.current_level}", (center_x - 80, center_y - 150), 
                   font, 1.2, game.COLOR_THEME['accent'], 3)
        cv2.putText(frame, f"TARGET NUMBER: {target}", (center_x - 150, center_y - 100), 
                   font, 1, game.COLOR_THEME['success'], 2)
        
        # Countdown timer with sound effects
        elapsed = current_time - game.countdown_timer
        remaining = max(0, game.countdown_duration - elapsed)
        
        if remaining > 2000:
            cv2.putText(frame, "READY?", (center_x - 60, center_y), 
                       font, 1.5, game.COLOR_THEME['text_light'], 3)
        elif remaining > 1000:
            cv2.putText(frame, "SET!", (center_x - 40, center_y), 
                       font, 1.5, game.COLOR_THEME['text_light'], 3)
            # Play countdown sound on first frame of "SET!"
            if elapsed >= 1000 and elapsed < 1100:  # Small window to avoid multiple plays
                game.play_sound('countdown')
        else:
            cv2.putText(frame, "GO!", (center_x - 30, center_y), 
                       font, 1.5, game.COLOR_THEME['accent'], 3)
    
    elif game.current_state == game.GAME_STATE_PLAYING:
        # Game UI with hearts display and progress counter
        target = game.LEVEL_TARGETS[game.current_level]
        
        # UI overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (450, 200), game.COLOR_THEME['ui_overlay'], -1)
        cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)
        
        cv2.putText(frame, f"LEVEL: {game.current_level}", (10, 30), 
                   font, 0.8, game.COLOR_THEME['accent'], 2)
        cv2.putText(frame, f"TARGET: {target}", (10, 60), 
                   font, 0.8, game.COLOR_THEME['success'], 2)
        
        # 🎯 PROGRESS COUNTER - Shows completion status
        progress_color = game.COLOR_THEME['success'] if game.correct_balloons_caught >= game.correct_balloons_in_level else game.COLOR_THEME['text_light']
        cv2.putText(frame, f"PROGRESS: {game.correct_balloons_caught}/{game.correct_balloons_in_level}", (10, 90), 
                   font, 0.8, progress_color, 2)
        
        cv2.putText(frame, f"TOTAL SCORE: {game.total_score}", (10, 120), 
                   font, 0.8, game.COLOR_THEME['accent'], 2)
        
        # 💖 HEARTS DISPLAY - Show remaining lives
        cv2.putText(frame, "LIVES:", (10, 160), font, 0.7, game.COLOR_THEME['text_light'], 2)
        
        # Draw hearts (max 10 hearts in 2 rows)
        for i in range(game.MAX_HEARTS):
            row = i // 5  # 5 hearts per row
            col = i % 5
            heart_x = 80 + col * 35
            heart_y = 140 + row * 35
            
            # Choose heart sprite based on remaining lives
            if i < game.hearts:
                heart_sprite = game.sprites['heart_full']
            else:
                heart_sprite = game.sprites['heart_broken']
            
            # Draw heart sprite
            if heart_sprite is not None:
                h, w = heart_sprite.shape[:2]
                if (heart_y >= 0 and heart_x >= 0 and 
                    heart_y + h < frame.shape[0] and heart_x + w < frame.shape[1]):
                    frame[heart_y:heart_y + h, heart_x:heart_x + w] = heart_sprite
    
    elif game.current_state == game.GAME_STATE_LEVEL_COMPLETE:
        # Level complete screen
        cv2.rectangle(frame, (center_x - 200, center_y - 100), 
                     (center_x + 200, center_y + 100), game.COLOR_THEME['success'], -1)
        cv2.putText(frame, f"LEVEL {game.current_level} COMPLETE!", (center_x - 150, center_y - 30), 
                   font, 1, game.COLOR_THEME['text_light'], 2)
        cv2.putText(frame, f"Caught: {game.correct_balloons_caught}/{game.correct_balloons_in_level} correct", (center_x - 120, center_y + 10), 
                   font, 0.7, game.COLOR_THEME['text_light'], 2)
        
        if game.current_level < game.MAX_LEVEL:
            cv2.putText(frame, "Next level starting...", (center_x - 100, center_y + 50), 
                       font, 0.6, game.COLOR_THEME['text_light'], 2)
        else:
            cv2.putText(frame, "Game completed!", (center_x - 80, center_y + 50), 
                       font, 0.6, game.COLOR_THEME['text_light'], 2)
    
    elif game.current_state == game.GAME_STATE_GAME_OVER:
        # Game over screen (hearts depleted or game complete)
        if game.hearts <= 0:
            # Game over due to no hearts left
            cv2.rectangle(frame, (center_x - 200, center_y - 100), 
                         (center_x + 200, center_y + 100), game.COLOR_THEME['danger'], -1)
            cv2.putText(frame, "GAME OVER!", (center_x - 120, center_y - 30), 
                       font, 1.2, game.COLOR_THEME['text_light'], 3)
            cv2.putText(frame, "No hearts left!", (center_x - 100, center_y + 10), 
                       font, 0.7, game.COLOR_THEME['text_light'], 2)
            cv2.putText(frame, f"Final Score: {game.total_score}", (center_x - 100, center_y + 30), 
                       font, 0.7, game.COLOR_THEME['text_light'], 2)
        else:
            # Game complete (all levels finished)
            cv2.rectangle(frame, (center_x - 200, center_y - 100), 
                         (center_x + 200, center_y + 100), game.COLOR_THEME['accent'], -1)
            cv2.putText(frame, "GAME COMPLETE!", (center_x - 120, center_y - 30), 
                       font, 1.2, game.COLOR_THEME['text_light'], 3)
            cv2.putText(frame, f"Final Score: {game.total_score}", (center_x - 100, center_y + 10), 
                       font, 0.8, game.COLOR_THEME['text_light'], 2)
            cv2.putText(frame, f"Hearts Remaining: {game.hearts}", (center_x - 120, center_y + 30), 
                       font, 0.7, game.COLOR_THEME['text_light'], 2)
        
        cv2.putText(frame, "Press 'R' to restart", (center_x - 100, center_y + 60), 
                   font, 0.7, game.COLOR_THEME['text_light'], 2)

def restart_game(game):
    """Restart the game"""
    game.current_level = 1
    game.current_state = game.GAME_STATE_MENU
    game.correct_catches = 0
    game.total_score = 0
    game.hearts = game.MAX_HEARTS  # Reset hearts to full
    game.balloons = []
    game.finger_trail = []
    game.correct_balloons_in_level = 0  # Reset level counters
    game.correct_balloons_caught = 0    # Reset level counters

if __name__ == "__main__":
    print("Required packages: pip install opencv-python mediapipe pygame numpy")
    main()