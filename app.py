import pygame
import math
import sys
import os
import json
import numpy as np
import random
from collections import deque
from dataclasses import dataclass
from typing import List, Tuple, Optional, Callable, Any

# --- Configuration ---
WIDTH, HEIGHT = 800, 600
FOV = math.pi / 3
HALF_FOV = FOV / 2
NUM_RAYS = 200
MAX_DEPTH = 25
MINIMAP_SIZE = 150
TEX_SIZE = 64

# ==========================================
# 1. EVENT BUS (Observer Pattern)
# ==========================================
class EventBus:
    def __init__(self):
        self.listeners = {}
        
    def subscribe(self, event_name: str, callback: Callable[[Any], None]):
        self.listeners.setdefault(event_name, []).append(callback)
        
    def publish(self, event_name: str, data: Any = None):
        for callback in self.listeners.get(event_name, []):
            callback(data)

# ==========================================
# 2. OBJECT POOL (Particle Pool)
# ==========================================
class ParticlePool:
    def __init__(self, size=500):
        self.pool = []
        for _ in range(size):
            self.pool.append({
                'active': False, 'x': 0, 'y': 0, 'vx': 0, 'vy': 0,
                'life': 0, 'max_life': 0, 'color': (0,0,0), 'size': 0
            })
        self.index = 0
        
    def spawn(self, x, y, vx, vy, life, color, size=3):
        p = self.pool[self.index]
        p['active'] = True
        p['x'], p['y'] = x, y
        p['vx'], p['vy'] = vx, vy
        p['life'], p['max_life'] = life, life
        p['color'], p['size'] = color, size
        self.index = (self.index + 1) % len(self.pool)
        
    def update(self, dt: float):
        for p in self.pool:
            if p['active']:
                p['x'] += p['vx'] * dt
                p['y'] += p['vy'] * dt
                p['life'] -= dt
                if p['life'] <= 0:
                    p['active'] = False

# ==========================================
# 3. COMMAND PATTERN (Input Handling)
# ==========================================
class Command:
    def execute(self, context) -> Optional[str]: pass

class CmdShoot(Command):
    def execute(self, context): context.player.shoot()

class CmdReload(Command):
    def execute(self, context): context.player.reload()

class CmdTogglePause(Command):
    def execute(self, context): return "PAUSE"

class CmdToggleMouseLook(Command):
    def execute(self, context):
        context.mouse_look = not context.mouse_look
        pygame.event.set_grab(context.mouse_look)
        pygame.mouse.set_visible(not context.mouse_look)

# ==========================================
# 4. DATACLASSES (Strict Entities)
# ==========================================
@dataclass
class Player:
    x: float; y: float; angle: float = 0
    speed: float = 3.0; health: int = 100; max_health: int = 100
    shield: int = 0; speed_boost_timer: float = 0
    regen_timer: float = 0; berserk_timer: float = 0
    rapid_fire_timer: float = 0; poison_timer: float = 0
    focus_timer: float = 0; magnet_timer: float = 0; explosive_timer: float = 0
    slow_timer: float = 0
    # New Buffs
    invis_timer: float = 0; iron_skin_timer: float = 0; luck_timer: float = 0
    # New Debuffs
    bleed_timer: float = 0; webbed_timer: float = 0; dizzy_timer: float = 0
    
    shoot_cooldown: float = 0; ammo: int = 12; max_ammo: int = 12
    reload_timer: float = 0; footstep_timer: float = 0
    damage_flash: float = 0; gun_kickback: float = 0
    move_x: int = 0; move_y: int = 0

    def shoot(self):
        if self.shoot_cooldown <= 0 and self.health > 0 and context.state.__class__.__name__ == "PlayingState":
            is_rapid = self.rapid_fire_timer > 0
            cooldown = 0.05 if is_rapid else 0.2
            
            if not is_rapid and self.ammo <= 0:
                context.event_bus.publish("PLAY_SOUND", "empty")
            else:
                self.shoot_cooldown = cooldown
                if not is_rapid:
                    self.ammo -= 1
                self.gun_kickback = 1.0
                context.muzzle_flash_timer = 0.08
                context.event_bus.publish("PLAY_SOUND", "shoot")
                context.shoot_hit_scan()

    def reload(self):
        if self.reload_timer <= 0 and self.ammo < self.max_ammo and self.health > 0:
            self.reload_timer = 1.0
            context.event_bus.publish("PLAY_SOUND", "reload")

@dataclass
class Enemy:
    x: float; y: float; etype: str = 'grunt'
    health: int = 50; speed: float = 1.2; damage: int = 10; scale: float = 1.0
    alive: bool = True; attack_cooldown: float = 0; alert_timer: float = 0
    hit_flash: float = 0; wander_angle: float = 0; wander_timer: float = 0
    applies_poison: bool = False
    # New Enemy Debuffs
    burn_timer: float = 0; choking_timer: float = 0; is_enraged: bool = False

@dataclass
class Pickup:
    x: float; y: float; ptype: str = 'health'
    alive: bool = True; bob: float = 0

@dataclass
class Torch:
    x: float; y: float; flicker: float = 0

@dataclass
class Prop:
    x: float; y: float; emoji_key: str = 'crate'

# ==========================================
# 5. DDA RAYCASTING (Dynamic FOV)
# ==========================================
def cast_rays_dda(player: Player, level, current_fov: float) -> List[Tuple[float, int, int, float]]:
    rays = []
    half_fov = current_fov / 2
    for i in range(NUM_RAYS):
        ray_angle = player.angle - half_fov + (i / NUM_RAYS) * current_fov
        ray_dir_x = math.cos(ray_angle)
        ray_dir_y = math.sin(ray_angle)
        
        map_x = int(player.x)
        map_y = int(player.y)
        
        delta_dist_x = abs(1 / ray_dir_x) if ray_dir_x != 0 else 1e10
        delta_dist_y = abs(1 / ray_dir_y) if ray_dir_y != 0 else 1e10
        
        if ray_dir_x < 0:
            step_x = -1
            side_dist_x = (player.x - map_x) * delta_dist_x
        else:
            step_x = 1
            side_dist_x = (map_x + 1.0 - player.x) * delta_dist_x
            
        if ray_dir_y < 0:
            step_y = -1
            side_dist_y = (player.y - map_y) * delta_dist_y
        else:
            step_y = 1
            side_dist_y = (map_y + 1.0 - player.y) * delta_dist_y
            
        hit_wall = False
        side = 0
        dist = 0
        
        while not hit_wall and dist < MAX_DEPTH:
            if side_dist_x < side_dist_y:
                side_dist_x += delta_dist_x
                map_x += step_x
                side = 1
            else:
                side_dist_y += delta_dist_y
                map_y += step_y
                side = 0
                
            if map_x < 0 or map_x >= level.width or map_y < 0 or map_y >= level.height:
                hit_wall = True
                dist = MAX_DEPTH
            elif level.grid[map_y][map_x] != '.':
                hit_wall = True
                if side == 1: dist = (side_dist_x - delta_dist_x)
                else: dist = (side_dist_y - delta_dist_y)
                    
        corrected_dist = max(0.01, dist * math.cos(ray_angle - player.angle))
        
        if side == 1: hit_x = player.y + dist * ray_dir_y
        else: hit_x = player.x + dist * ray_dir_x
        hit_x = hit_x - math.floor(hit_x)
        
        rays.append((corrected_dist, 1, side, hit_x))
    return rays

def has_line_of_sight(x1, y1, x2, y2, level):
    dist = math.hypot(x2-x1, y2-y1)
    steps = int(dist / 0.1) + 1
    for i in range(1, steps):
        t = i / steps
        cx = x1 + (x2-x1) * t
        cy = y1 + (y2-y1) * t
        if 0 <= int(cy) < level.height and 0 <= int(cx) < level.width:
            if level.grid[int(cy)][int(cx)] != '.': return False
        else: return False
    return True

# ==========================================
# 6. AUDIO SYNTHESIS (8-Bit Engine)
# ==========================================
def generate_sound(freq, duration, wave_type='square', volume=0.5, noise_mix=0.0):
    if not pygame.mixer.get_init(): return None
    sample_rate = 44100
    n_samples = int(sample_rate * duration)
    buf = np.zeros(n_samples, dtype=np.int16)
    
    for i in range(n_samples):
        t = float(i) / sample_rate
        if wave_type == 'square': val = 1.0 if math.sin(2 * math.pi * freq * t) > 0 else -1.0
        elif wave_type == 'sine': val = math.sin(2 * math.pi * freq * t)
        elif wave_type == 'noise': val = np.random.uniform(-1, 1)
        elif wave_type == 'sweep': val = 1.0 if math.sin(2 * math.pi * freq * (1.0 - t / duration) * t) > 0 else -1.0
        elif wave_type == 'sweep_up': val = math.sin(2 * math.pi * freq * (1.0 + (t / duration) * 3) * t)
        else: val = 0
        
        if noise_mix > 0: val = val * (1 - noise_mix) + np.random.uniform(-1, 1) * noise_mix
        env = min(1.0, t / 0.01) * max(0.0, 1.0 - t / duration)
        buf[i] = int(val * volume * env * 32767)
        
    return pygame.sndarray.make_sound(np.column_stack((buf, buf)))

# ==========================================
# 7. LEVEL GENERATOR & PROCEDURAL TEXTURES
# ==========================================
class LevelGenerator:
    def __init__(self, width=24, height=24):
        self.width, self.height = width, height
        self.grid = [['1' for _ in range(width)] for _ in range(height)]
        self.spawn_x, self.spawn_y = 0, 0
        self.enemy_spawns, self.pickup_spawns, self.torch_spawns, self.prop_spawns = [], [], [], []
        self.name, self.wall_color, self.floor_color, self.ceil_color = "", (0,0,0), (0,0,0), (0,0,0)
        self.darkness = 180
        self.wall_texture = None

    def generate(self, level_num):
        c = random.random()
        if c < 0.33:
            self.gen_caves(); self.name="CAVES"; self.wall_color=(110, 80, 60); self.floor_color=(40, 30, 20); self.ceil_color=(15, 10, 5); self.darkness=200
        elif c < 0.66:
            self.gen_rooms(); self.name="DUNGEON"; self.wall_color=(80, 80, 120); self.floor_color=(30, 30, 50); self.ceil_color=(10, 10, 20); self.darkness=220
        else:
            self.gen_maze(); self.name="LABYRINTH"; self.wall_color=(60, 100, 60); self.floor_color=(20, 40, 20); self.ceil_color=(5, 15, 5); self.darkness=240
        
        self.wall_texture = self.generate_texture()
        self.find_spawns(level_num)
        self.gen_torches()
        self.gen_props()

    def generate_texture(self):
        tex = pygame.Surface((TEX_SIZE, TEX_SIZE))
        tex.fill(self.wall_color)
        for _ in range(200):
            x, y = random.randint(0, TEX_SIZE-1), random.randint(0, TEX_SIZE-1)
            c_offset = random.randint(-20, 20)
            col = tuple(max(0, min(255, c + c_offset)) for c in self.wall_color)
            tex.set_at((x, y), col)
        if self.name == "DUNGEON":
            pygame.draw.rect(tex, (0,0,0), (0,0,TEX_SIZE,TEX_SIZE), 2)
            pygame.draw.line(tex, (0,0,0), (0,32), (64,32), 2)
            pygame.draw.line(tex, (0,0,0), (32,0), (32,32), 2)
            pygame.draw.line(tex, (0,0,0), (16,32), (16,64), 2)
            pygame.draw.line(tex, (0,0,0), (48,32), (48,64), 2)
        elif self.name == "LABYRINTH":
            for i in range(0, TEX_SIZE, 8):
                pygame.draw.line(tex, (0,0,0), (i, 0), (i, TEX_SIZE), 1)
                pygame.draw.line(tex, (0,0,0), (0, i), (TEX_SIZE, i), 1)
        return tex

    def gen_caves(self):
        for y in range(self.height):
            for x in range(self.width):
                if x in (0, self.width-1) or y in (0, self.height-1): self.grid[y][x] = '1'
                else: self.grid[y][x] = '.' if random.random() > 0.45 else '1'
        for _ in range(5):
            ng = [row[:] for row in self.grid]
            for y in range(1, self.height-1):
                for x in range(1, self.width-1):
                    walls = sum(1 for dy in [-1,0,1] for dx in [-1,0,1] 
                                if (dx!=0 or dy!=0) and (x+dx<0 or x+dx>=self.width or y+dy<0 or y+dy>=self.height or self.grid[y+dy][x+dx]=='1'))
                    ng[y][x] = '1' if walls >= 5 else '.'
            self.grid = ng

    def gen_rooms(self):
        rooms = []
        for _ in range(100):
            w, h = random.randint(4, 8), random.randint(4, 8)
            x, y = random.randint(1, self.width-w-1), random.randint(1, self.height-h-1)
            if not any(x < rx+rw+1 and x+w+1 > rx and y < ry+rh+1 and y+h+1 > ry for rx,ry,rw,rh in rooms):
                rooms.append((x, y, w, h))
                for ry in range(y, y+h):
                    for rx in range(x, x+w): self.grid[ry][rx] = '.'
        for i in range(len(rooms)-1):
            x1,y1,w1,h1 = rooms[i]; x2,y2,w2,h2 = rooms[i+1]
            cx1, cy1 = x1+w1//2, y1+h1//2; cx2, cy2 = x2+w2//2, y2+h2//2
            if random.random() > 0.5:
                self.carve_h(cx1, cx2, cy1); self.carve_v(cy1, cy2, cx2)
            else:
                self.carve_v(cy1, cy2, cx1); self.carve_h(cx1, cx2, cy2)

    def gen_maze(self):
        for y in range(self.height):
            for x in range(self.width): self.grid[y][x] = '1'
        sx, sy = 1, 1; self.grid[sy][sx] = '.'; stack = [(sx, sy)]
        while stack:
            cx, cy = stack[-1]; neighbors = []
            for dx, dy in [(0,-2),(0,2),(-2,0),(2,0)]:
                nx, ny = cx+dx, cy+dy
                if 0 < nx < self.width-1 and 0 < ny < self.height-1 and self.grid[ny][nx] == '1':
                    neighbors.append((nx, ny, dx, dy))
            if neighbors:
                nx, ny, dx, dy = random.choice(neighbors)
                self.grid[cy+dy//2][cx+dx//2] = '.'; self.grid[ny][nx] = '.'; stack.append((nx, ny))
            else: stack.pop()
            
    def carve_h(self, x1, x2, y):
        for x in range(min(x1, x2), max(x1, x2) + 1): self.grid[y][x] = '.'
    def carve_v(self, y1, y2, x):
        for y in range(min(y1, y2), max(y1, y2) + 1): self.grid[y][x] = '.'

    def gen_torches(self):
        self.torch_spawns = []
        for y in range(1, self.height-1):
            for x in range(1, self.width-1):
                if self.grid[y][x] == '1' and any(self.grid[y+dy][x+dx]=='.' for dx,dy in [(0,-1),(0,1),(-1,0),(1,0)]):
                    if random.random() < 0.1: self.torch_spawns.append((x + 0.5, y + 0.5))

    def gen_props(self):
        self.prop_spawns = []
        prop_choices = ['crate', 'bone', 'tree', 'cobweb', 'rock', 'vase', 'gravestone', 'barrel']
        for _ in range(self.width * self.height // 15):
            rx, ry = random.randint(1, self.width-2), random.randint(1, self.height-2)
            if self.grid[ry][rx] == '.':
                if math.hypot(rx - self.spawn_x, ry - self.spawn_y) > 2:
                    self.prop_spawns.append(Prop(rx + 0.5, ry + 0.5, random.choice(prop_choices)))

    def find_spawns(self, level_num):
        px, py = self.width//2, self.height//2
        found = False
        for r in range(max(self.width, self.height)):
            for dy in range(-r, r+1):
                for dx in range(-r, r+1):
                    nx, ny = px+dx, py+dy
                    if 0<=nx<self.width and 0<=ny<self.height and self.grid[ny][nx]=='.':
                        px, py = nx, ny; found = True; break
                if found: break
            if found: break
        self.grid[py][px] = '.'; self.spawn_x, self.spawn_y = px + 0.5, py + 0.5
        
        self.enemy_spawns = []
        needed = 1 if level_num % 3 == 0 else 4 + level_num
        for _ in range(500):
            if len(self.enemy_spawns) >= needed: break
            rx, ry = random.randint(1, self.width-2), random.randint(1, self.height-2)
            if self.grid[ry][rx] == '.' and math.hypot(rx-px, ry-py) > 5:
                if level_num % 3 == 0:
                    self.enemy_spawns.append((rx+0.5, ry+0.5, 'boss'))
                else:
                    r = random.random()
                    if r < 0.05: etype = 'grunt'
                    elif r < 0.10: etype = 'fast'
                    elif r < 0.14: etype = 'brute'
                    elif r < 0.18: etype = 'ghost'
                    elif r < 0.22: etype = 'zombie'
                    elif r < 0.26: etype = 'robot'
                    elif r < 0.30: etype = 'bat'
                    elif r < 0.34: etype = 'wolf'
                    elif r < 0.38: etype = 'spider'
                    elif r < 0.42: etype = 'snake'
                    elif r < 0.46: etype = 'vampire'
                    elif r < 0.50: etype = 'alien'
                    elif r < 0.54: etype = 'skeleton'
                    elif r < 0.56: etype = 'raccoon'
                    elif r < 0.60: etype = 'octopus'
                    elif r < 0.64: etype = 'snail'
                    elif r < 0.66: etype = 'wyvern'
                    elif r < 0.70: etype = 'bear'
                    elif r < 0.73: etype = 'crocodile'
                    elif r < 0.75: etype = 'trex'
                    elif r < 0.78: etype = 'frog'
                    elif r < 0.81: etype = 'bee'
                    elif r < 0.84: etype = 'crab'
                    elif r < 0.87: etype = 'deer'
                    elif r < 0.89: etype = 'jellyfish'
                    elif r < 0.91: etype = 'mosquito'
                    elif r < 0.94: etype = 'rhino'
                    elif r < 0.96: etype = 'turtle'
                    elif r < 0.97: etype = 'llama'
                    elif r < 0.99: etype = 'tiger'
                    elif r < 0.995: etype = 'hippo'
                    elif r < 0.997: etype = 'otter'
                    elif r < 0.998: etype = 'hedgehog'
                    elif r < 0.999: etype = 'lobster'
                    elif r < 0.9995: etype = 'cricket'
                    else: etype = 'turkey'
                    
                    spawn_count = 1
                    if etype in ['alien', 'virus', 'mouse', 'bee', 'mosquito']: 
                        spawn_count = random.randint(2, 5)
                        
                    for _ in range(spawn_count):
                        self.enemy_spawns.append((rx + 0.5 + random.uniform(-0.2, 0.2), ry + 0.5 + random.uniform(-0.2, 0.2), etype))
                    
        self.pickup_spawns = []
        for _ in range(level_num + 4): # Increased spawns for more effects
            rx, ry = random.randint(1, self.width-2), random.randint(1, self.height-2)
            if self.grid[ry][rx] == '.':
                r = random.random()
                if r < 0.15: ptype = 'health'
                elif r < 0.25: ptype = 'ammo'
                elif r < 0.30: ptype = 'treasure'
                elif r < 0.35: ptype = 'shield'
                elif r < 0.40: ptype = 'speed'
                elif r < 0.45: ptype = 'regen'
                elif r < 0.50: ptype = 'berserk'
                elif r < 0.55: ptype = 'rapid_fire'
                elif r < 0.60: ptype = 'freeze'
                elif r < 0.65: ptype = 'focus'
                elif r < 0.70: ptype = 'magnet'
                elif r < 0.75: ptype = 'explosive'
                elif r < 0.80: ptype = 'invisibility'
                elif r < 0.85: ptype = 'iron_skin'
                elif r < 0.90: ptype = 'luck'
                else: ptype = 'potion'
                self.pickup_spawns.append((rx+0.5, ry+0.5, ptype))

# ==========================================
# 8. FINITE STATE MACHINE (Game States)
# ==========================================
class GameState:
    def handle_input(self, context): pass
    def update(self, context, dt): pass
    def draw(self, screen, context): pass

class PlayingState(GameState):
    def handle_input(self, context):
        p = context.player
        keys = pygame.key.get_pressed()
        
        # Dizzy effect scrambles controls
        dx, dy = 1, 1
        if p.dizzy_timer > 0:
            dx = -1 if random.random() > 0.5 else 1
            dy = -1 if random.random() > 0.5 else 1
            p.angle += random.uniform(-0.05, 0.05)
        
        if keys[pygame.K_w]: p.move_y += dy
        if keys[pygame.K_s]: p.move_y -= dy
        if keys[pygame.K_a]: p.move_x -= dx
        if keys[pygame.K_d]: p.move_x += dx
        
        if keys[pygame.K_LEFT]: p.angle -= 2.5 * context.dt
        if keys[pygame.K_RIGHT]: p.angle += 2.5 * context.dt
        
        if context.mouse_look:
            mx, my = pygame.mouse.get_pos()
            p.angle += (mx - WIDTH//2) * 0.003
            pygame.mouse.set_pos(WIDTH//2, HEIGHT//2)

    def update(self, context, dt):
        p = context.player
        if p.health <= 0: return
        
        # --- Status Effect Logic ---
        if p.webbed_timer > 0:
            p.webbed_timer -= dt
            p.speed = 0.0
            p.move_x, p.move_y = 0, 0 # Hard root
        elif p.slow_timer > 0:
            p.slow_timer -= dt
            p.speed = 1.5
        elif p.speed_boost_timer > 0:
            p.speed_boost_timer -= dt
            p.speed = 5.5
        else:
            p.speed = 3.0
        
        if p.regen_timer > 0:
            p.regen_timer -= dt
            p.health = min(p.max_health, p.health + 10 * dt)
            
        if p.berserk_timer > 0: p.berserk_timer -= dt
        if p.rapid_fire_timer > 0: p.rapid_fire_timer -= dt
        if p.focus_timer > 0: p.focus_timer -= dt
        if p.magnet_timer > 0: p.magnet_timer -= dt
        if p.explosive_timer > 0: p.explosive_timer -= dt
        if p.invis_timer > 0: p.invis_timer -= dt
        if p.iron_skin_timer > 0: p.iron_skin_timer -= dt
        if p.luck_timer > 0: p.luck_timer -= dt
        if p.dizzy_timer > 0: p.dizzy_timer -= dt
        
        if p.poison_timer > 0:
            p.poison_timer -= dt
            p.health -= 3 * dt
            if random.random() < 0.1:
                context.particle_pool.spawn(p.x, p.y, random.uniform(-0.5,0.5), random.uniform(-0.5,0.5), 0.5, (100, 200, 50), 3)
                
        if p.bleed_timer > 0:
            p.bleed_timer -= dt
            p.health -= 6 * dt
            if random.random() < 0.2:
                context.particle_pool.spawn(p.x + random.uniform(-0.2, 0.2), p.y + random.uniform(-0.2, 0.2), random.uniform(-0.5,0.5), random.uniform(-0.5,0.5), 0.5, (200, 0, 0), 4)
                
        if context.freeze_timer > 0: context.freeze_timer -= dt
        
        if p.magnet_timer > 0:
            for pk in context.pickups:
                if pk.alive:
                    dx, dy = p.x - pk.x, p.y - pk.y
                    dist = math.hypot(dx, dy)
                    if dist < 8.0 and dist > 0.1:
                        pk.x += (dx/dist) * 3.0 * dt
                        pk.y += (dy/dist) * 3.0 * dt
        
        mag = math.hypot(p.move_x, p.move_y)
        if mag > 0:
            mx = (p.move_x / mag) * p.speed * dt
            my = (p.move_y / mag) * p.speed * dt
            nx = p.x + mx * math.cos(p.angle) - my * math.sin(p.angle)
            ny = p.y + mx * math.sin(p.angle) + my * math.cos(p.angle)
            if context.level.grid[int(ny)][int(nx + (0.2 if mx > 0 else -0.2))] == '.': p.x = nx
            if context.level.grid[int(ny + (0.2 if my > 0 else -0.2))][int(p.x)] == '.': p.y = ny
            p.footstep_timer -= dt
            if p.footstep_timer <= 0:
                p.footstep_timer = 0.35 if p.speed < 5.0 else 0.2
                context.event_bus.publish("PLAY_SOUND", "footstep")
                
        p.move_x, p.move_y = 0, 0
        
        p.shoot_cooldown = max(0, p.shoot_cooldown - dt)
        p.gun_kickback = max(0, p.gun_kickback - dt * 5)
        if p.reload_timer > 0:
            p.reload_timer -= dt
            if p.reload_timer <= 0: p.ammo = p.max_ammo
            
        context.muzzle_flash_timer = max(0, context.muzzle_flash_timer - dt)
        context.screen_shake = max(0, context.screen_shake - dt)
        context.damage_flash = max(0, context.damage_flash - dt)
        context.hit_marker_timer = max(0, context.hit_marker_timer - dt)
        
        target_fov = math.pi / 5 if p.focus_timer > 0 else FOV
        context.current_fov += (target_fov - context.current_fov) * dt * 4.0
        context.current_half_fov = context.current_fov / 2
        
        for e in context.enemies:
            if not e.alive: continue
            is_frozen = context.freeze_timer > 0
            
            # Enemy DoTs and Debuffs
            if e.burn_timer > 0:
                e.burn_timer -= dt
                e.health -= 5 * dt
                if random.random() < 0.2:
                    context.particle_pool.spawn(e.x, e.y, random.uniform(-0.5,0.5), -random.uniform(0.5, 1.5), 0.5, (255, 100, 0), 3)
                if e.health <= 0:
                    e.alive = False
                    context.score += 50
                    context.event_bus.publish("ENEMY_DIED", e)
                    continue
                    
            if e.choking_timer > 0:
                e.choking_timer -= dt
                e.attack_cooldown = 1.0 # Reset attack cooldown so they can't strike
                
            # Enemy Enrage Logic
            max_hp = context.get_enemy_stats(e.etype)['health']
            if e.health < max_hp * 0.3 and not e.is_enraged and e.etype not in ['deer', 'turkey', 'boss']:
                e.is_enraged = True
                e.speed *= 1.5
                e.damage = int(e.damage * 1.5)
                context.event_bus.publish("PLAY_SOUND", "boss")
                
            dx, dy = p.x - e.x, p.y - e.y
            dist = math.hypot(dx, dy)
            if e.hit_flash > 0: e.hit_flash -= dt
            if e.alert_timer > 0 and e.etype != 'demon': e.alert_timer -= dt
            
            # Invisibility breaks aggro
            if p.invis_timer <= 0:
                if dist < 15 and has_line_of_sight(e.x, e.y, p.x, p.y, context.level): e.alert_timer = 5.0
            else:
                e.alert_timer = max(0, e.alert_timer - dt) # Faster decay if invisible
            
            # Determine actual movement speed for this frame
            actual_speed = e.speed
            if e.etype == 'rhino' and e.alert_timer > 0: actual_speed *= 2.0
            if e.etype == 'hedgehog' and e.alert_timer > 0: actual_speed *= 3.0
            if e.etype == 'tiger' and e.alert_timer > 0:
                e.wander_angle += random.uniform(-0.8, 0.8)
                
            if e.etype == 'hippo' and dist > 5.0: actual_speed = 0.0
                
            if not is_frozen:
                # Enemy-specific AI
                if e.etype in ['deer', 'turkey']:
                    if dist < 10:
                        flee_angle = math.atan2(-dy, -dx)
                        nx = e.x + math.cos(flee_angle)*actual_speed*dt
                        ny = e.y + math.sin(flee_angle)*actual_speed*dt
                        if 0 <= int(ny) < context.level.height and 0 <= int(nx) < context.level.width:
                            if context.level.grid[int(ny)][int(nx)] == '.': e.x, e.y = nx, ny
                        else:
                            e.wander_angle = random.uniform(0, math.pi*2)
                            nx = e.x + math.cos(e.wander_angle)*actual_speed*dt
                            ny = e.y + math.sin(e.wander_angle)*actual_speed*dt
                            if 0 <= int(ny) < context.level.height and 0 <= int(nx) < context.level.width:
                                if context.level.grid[int(ny)][int(nx)] == '.': e.x, e.y = nx, ny
                    else:
                        e.wander_timer -= dt
                        if e.wander_timer <= 0:
                            e.wander_timer = random.uniform(1, 2)
                            e.wander_angle = random.random() * math.pi * 2
                        nx = e.x + math.cos(e.wander_angle)*actual_speed*0.5*dt
                        ny = e.y + math.sin(e.wander_angle)*actual_speed*0.5*dt
                        if context.level.grid[int(ny)][int(nx)] == '.': e.x, e.y = nx, ny

                elif e.etype in ['crab', 'lobster'] and e.alert_timer > 0:
                    if not hasattr(e, 'strafe_dir'): e.strafe_dir = 1 if random.random() > 0.5 else -1
                    if not hasattr(e, 'strafe_timer'): e.strafe_timer = random.uniform(1, 3)
                    e.strafe_timer -= dt
                    if e.strafe_timer <= 0:
                        e.strafe_dir *= -1
                        e.strafe_timer = random.uniform(1, 3)
                        
                    angle = math.atan2(dy, dx) + (math.pi/2 * e.strafe_dir)
                    nx = e.x + math.cos(angle)*actual_speed*dt
                    ny = e.y + math.sin(angle)*actual_speed*dt
                    if 0 <= int(ny) < context.level.height and 0 <= int(nx) < context.level.width:
                        if context.level.grid[int(ny)][int(nx)] == '.': e.x, e.y = nx, ny

                elif e.etype in ['frog', 'cricket']:
                    hop_mult = 2.0 * abs(math.sin(pygame.time.get_ticks() * (0.006 if e.etype=='frog' else 0.012)))
                    if e.etype == 'cricket' and random.random() < 0.05:
                        e.wander_angle = random.uniform(0, math.pi*2)
                        
                    if e.alert_timer > 0 and dist > 0.5:
                        mvx, mvy = (dx/dist)*actual_speed*hop_mult*dt, (dy/dist)*actual_speed*hop_mult*dt
                        nx = e.x + mvx
                        if 0 <= int(e.y) < context.level.height and 0 <= int(nx) < context.level.width:
                            if context.level.grid[int(e.y)][int(nx)] == '.': e.x = nx
                        ny = e.y + mvy
                        if 0 <= int(ny) < context.level.height and 0 <= int(e.x) < context.level.width:
                            if context.level.grid[int(ny)][int(e.x)] == '.': e.y = ny
                    elif dist > 0.5:
                        e.wander_timer -= dt
                        if e.wander_timer <= 0:
                            e.wander_timer = random.uniform(1, 3)
                            e.wander_angle = random.random() * math.pi * 2
                        nx = e.x + math.cos(e.wander_angle)*actual_speed*hop_mult*0.3*dt
                        ny = e.y + math.sin(e.wander_angle)*actual_speed*hop_mult*0.3*dt
                        if context.level.grid[int(ny)][int(nx)] == '.': e.x, e.y = nx, ny
                    else:
                        e.attack_cooldown -= dt
                        if e.attack_cooldown <= 0:
                            e.attack_cooldown = 1.0
                            p.health -= e.damage
                            context.damage_flash = 0.3
                            context.screen_shake = 0.2
                            context.event_bus.publish("PLAY_SOUND", "hurt")

                elif e.alert_timer > 0 and dist > 0.5:
                    if e.etype in ('ghost', 'alien', 'virus', 'bat', 'bee', 'jellyfish', 'mosquito'):
                        e.x += (dx/dist)*actual_speed*dt; e.y += (dy/dist)*actual_speed*dt
                    else:
                        mvx, mvy = (dx/dist)*actual_speed*dt, (dy/dist)*actual_speed*dt
                        nx = e.x + mvx
                        if 0 <= int(e.y) < context.level.height and 0 <= int(nx) < context.level.width:
                            if context.level.grid[int(e.y)][int(nx)] == '.': e.x = nx
                        ny = e.y + mvy
                        if 0 <= int(ny) < context.level.height and 0 <= int(e.x) < context.level.width:
                            if context.level.grid[int(ny)][int(e.x)] == '.': e.y = ny
                elif dist > 0.5:
                    e.wander_timer -= dt
                    if e.wander_timer <= 0:
                        e.wander_timer = random.uniform(1, 3)
                        e.wander_angle = random.random() * math.pi * 2
                    nx = e.x + math.cos(e.wander_angle)*actual_speed*0.3*dt
                    ny = e.y + math.sin(e.wander_angle)*actual_speed*0.3*dt
                    if context.level.grid[int(ny)][int(nx)] == '.': e.x, e.y = nx, ny
                else:
                    if e.etype not in ['crab', 'lobster', 'deer', 'turkey']:
                        e.attack_cooldown -= dt
                        if e.attack_cooldown <= 0:
                            e.attack_cooldown = 1.0
                            dmg = e.damage
                            
                            # Iron Skin blocks all damage and debuffs
                            if p.iron_skin_timer <= 0:
                                if p.shield > 0:
                                    absorb = min(p.shield, dmg)
                                    p.shield -= absorb
                                    dmg -= absorb
                                p.health -= dmg
                                context.damage_flash = 0.3
                                context.screen_shake = 0.2
                                context.event_bus.publish("PLAY_SOUND", "hurt")
                                
                                if e.applies_poison:
                                    p.poison_timer = max(p.poison_timer, 4.0)
                                if e.etype in ('octopus', 'jellyfish'):
                                    p.slow_timer = max(p.slow_timer, 2.0)
                                if e.etype in ('vampire', 'mosquito'):
                                    max_hp = context.get_enemy_stats(e.etype)['health']
                                    e.health = min(max_hp, e.health + 5)
                                # New Debuffs applied by enemies
                                if e.etype == 'spider':
                                    p.webbed_timer = max(p.webbed_timer, 1.5)
                                elif e.etype in ['bat', 'mosquito']:
                                    p.dizzy_timer = max(p.dizzy_timer, 3.0)
                                elif e.etype in ['tiger', 'crocodile', 'wolf']:
                                    p.bleed_timer = max(p.bleed_timer, 4.0)
                                    
            if e.etype == 'raccoon' and e.alert_timer > 0:
                for pk in context.pickups:
                    if pk.alive:
                        ddx, ddy = pk.x - e.x, pk.y - e.y
                        ddist = math.hypot(ddx, ddy)
                        if ddist < 5.0:
                            e.x += (ddx/ddist)*e.speed*dt
                            e.y += (ddy/ddist)*e.speed*dt
                            if ddist < 0.5:
                                pk.alive = False
                                context.event_bus.publish("PLAY_SOUND", "empty")
                                for _ in range(5):
                                    context.particle_pool.spawn(pk.x, pk.y, random.uniform(-1,1), random.uniform(-1,1), 0.5, (150,150,150), 3)
                            break 
                            
        for pk in context.pickups:
            if not pk.alive: continue
            pk.bob += dt * 4
            if math.hypot(pk.x - p.x, pk.y - p.y) < 0.5:
                if pk.ptype == 'health': p.health = min(p.max_health, p.health + 25)
                elif pk.ptype == 'ammo': p.ammo = p.max_ammo
                elif pk.ptype == 'treasure': context.score += 250
                elif pk.ptype == 'shield': p.shield = min(100, p.shield + 50)
                elif pk.ptype == 'speed': p.speed_boost_timer = 5.0
                elif pk.ptype == 'regen': p.regen_timer = 8.0
                elif pk.ptype == 'berserk': p.berserk_timer = 6.0
                elif pk.ptype == 'rapid_fire': p.rapid_fire_timer = 4.0
                elif pk.ptype == 'freeze': context.freeze_timer = 5.0
                elif pk.ptype == 'focus': p.focus_timer = 6.0
                elif pk.ptype == 'magnet': p.magnet_timer = 6.0
                elif pk.ptype == 'explosive': p.explosive_timer = 5.0
                elif pk.ptype == 'invisibility': p.invis_timer = 5.0
                elif pk.ptype == 'iron_skin': p.iron_skin_timer = 4.0
                elif pk.ptype == 'luck': p.luck_timer = 8.0
                elif pk.ptype == 'potion':
                    buffs = ['speed', 'regen', 'berserk', 'rapid_fire', 'shield', 'focus', 'magnet', 'explosive', 'invisibility', 'iron_skin', 'luck']
                    chosen = random.choice(buffs)
                    if chosen == 'speed': p.speed_boost_timer = 5.0
                    elif chosen == 'regen': p.regen_timer = 8.0
                    elif chosen == 'berserk': p.berserk_timer = 6.0
                    elif chosen == 'rapid_fire': p.rapid_fire_timer = 4.0
                    elif chosen == 'shield': p.shield = min(100, p.shield + 50)
                    elif chosen == 'focus': p.focus_timer = 6.0
                    elif chosen == 'magnet': p.magnet_timer = 6.0
                    elif chosen == 'explosive': p.explosive_timer = 5.0
                    elif chosen == 'invisibility': p.invis_timer = 5.0
                    elif chosen == 'iron_skin': p.iron_skin_timer = 4.0
                    elif chosen == 'luck': p.luck_timer = 8.0
                    
                pk.alive = False
                context.event_bus.publish("PLAY_SOUND", "pickup")
                context.event_bus.publish("SPAWN_PICKUP_PARTICLES", pk)
                
        context.particle_pool.update(dt)
        
        if all(not e.alive for e in context.enemies):
            context.state = TransitionState()

    def draw(self, screen, context):
        context.render_world(screen)
        context.render_hud(screen)

class PausedState(GameState):
    def draw(self, screen, context):
        context.render_world(screen)
        context.render_hud(screen)
        overlay = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 180))
        screen.blit(overlay, (0, 0))
        pause_text = context.font_huge.render("PAUSED", True, (255, 255, 255))
        screen.blit(pause_text, (WIDTH//2 - pause_text.get_width()//2, HEIGHT//2 - 100))
        info_y = HEIGHT//2
        for line in ["ESC - Quit", "P - Resume", "M - Toggle Mouse Look", "WASD - Move", "SPACE - Shoot", "R - Reload"]:
            t = context.font_hud.render(line, True, (200, 200, 200))
            screen.blit(t, (WIDTH//2 - t.get_width()//2, info_y))
            info_y += 30

class GameOverState(GameState):
    def __init__(self): self.timer = 0
    def update(self, context, dt):
        self.timer += dt
        if context.score > context.high_score:
            context.high_score = context.score
            try:
                with open('emoji_fps_hs.json', 'w') as f: json.dump({'high_score': context.high_score}, f)
            except: pass
    def draw(self, screen, context):
        context.render_world(screen)
        context.render_hud(screen)
        go_text = context.font_huge.render("GAME OVER", True, (255, 50, 50))
        skull_rect = context.skull_emoji.get_rect(center=(WIDTH//2, HEIGHT//2 - 100))
        screen.blit(context.skull_emoji, skull_rect)
        screen.blit(go_text, (WIDTH//2 - go_text.get_width()//2, HEIGHT//2))
        if self.timer > 1.5:
            restart_text = context.font_hud.render("Press R to Restart or ESC to Quit", True, (255,255,255))
            screen.blit(restart_text, (WIDTH//2 - restart_text.get_width()//2, HEIGHT//2 + 100))

class TransitionState(GameState):
    def __init__(self):
        self.timer = 3.0
        self.text = None
    def update(self, context, dt):
        self.timer -= dt
        if not self.text:
            self.text = context.font_hud.render(f"LEVEL {context.level_num} CLEARED!", True, (255, 255, 0))
            context.event_bus.publish("PLAY_SOUND", "level")
        if self.timer <= 0:
            context.load_next_level()
            context.state = PlayingState()
    def draw(self, screen, context):
        context.render_world(screen)
        context.render_hud(screen)
        fade = 1.0
        if self.timer > 2.5: fade = (3.0 - self.timer) / 0.5
        elif self.timer < 0.5: fade = self.timer / 0.5
        text_surf = self.text.copy()
        text_surf.set_alpha(int(255 * fade))
        screen.blit(text_surf, text_surf.get_rect(center=(WIDTH//2, HEIGHT//2)))

# ==========================================
# 9. GAME CONTEXT (Mediator)
# ==========================================
class GameContext:
    def __init__(self):
        self.event_bus = EventBus()
        self.particle_pool = ParticlePool(500)
        self.level_num = 1
        self.score = 0
        self.high_score = 0
        try:
            if os.path.exists('emoji_fps_hs.json'):
                with open('emoji_fps_hs.json', 'r') as f: self.high_score = json.load(f).get('high_score', 0)
        except: pass
        
        self.mouse_look = False
        self.muzzle_flash_timer = 0
        self.screen_shake = 0
        self.damage_flash = 0
        self.hit_marker_timer = 0
        self.freeze_timer = 0
        self.dt = 0
        
        self.current_fov = FOV
        self.current_half_fov = HALF_FOV
        
        self.crosshair_color = (255, 255, 255)
        self.crosshair_size = 10
        
        self.darkness_surf = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
        self.tint_surf = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
        
        self.load_level()
        self.state = PlayingState()
        
        self.sounds = {}
        self.init_audio()
        self.event_bus.subscribe("PLAY_SOUND", lambda name: self.sounds.get(name, lambda: None).play() if self.sounds.get(name) else None)
        self.event_bus.subscribe("ENEMY_DIED", self.on_enemy_died)
        self.event_bus.subscribe("SPAWN_PICKUP_PARTICLES", self.on_pickup_particles)

    def init_audio(self):
        try:
            pygame.mixer.init()
            self.sounds = {
                'shoot': generate_sound(800, 0.1, 'sweep', 0.3, 0.3),
                'hit': generate_sound(200, 0.15, 'noise', 0.4),
                'death': generate_sound(100, 0.4, 'sweep', 0.5),
                'level': generate_sound(400, 0.4, 'sweep_up', 0.4),
                'reload': generate_sound(300, 0.3, 'square', 0.3),
                'pickup': generate_sound(600, 0.15, 'sweep_up', 0.4),
                'hurt': generate_sound(150, 0.2, 'noise', 0.5),
                'empty': generate_sound(100, 0.05, 'square', 0.2),
                'footstep': generate_sound(80, 0.05, 'noise', 0.15),
                'boss': generate_sound(60, 0.8, 'sweep', 0.5, 0.4),
                'explosion': generate_sound(50, 0.3, 'noise', 0.6, 0.8)
            }
        except: pass

    def load_level(self):
        self.level = LevelGenerator()
        self.level.generate(self.level_num)
        self.player = Player(self.level.spawn_x, self.level.spawn_y)
        self.enemies = [Enemy(x, y, t, **self.get_enemy_stats(t)) for x, y, t in self.level.enemy_spawns]
        self.pickups = [Pickup(x, y, t) for x, y, t in self.level.pickup_spawns]
        self.torches = [Torch(x, y) for x, y in self.level.torch_spawns]
        self.props = self.level.prop_spawns
        self.explored = [[False]*self.level.width for _ in range(self.level.height)]
        
        self.sky_surf = pygame.Surface((WIDTH, HEIGHT//2))
        self.floor_surf = pygame.Surface((WIDTH, HEIGHT//2))
        for y in range(HEIGHT//2):
            c1 = tuple(min(255, c + y//10) for c in self.level.ceil_color)
            c2 = tuple(min(255, c + (HEIGHT//2 - y)//5) for c in self.level.floor_color)
            pygame.draw.line(self.sky_surf, c1, (0, y), (WIDTH, y))
            pygame.draw.line(self.floor_surf, c2, (0, y), (WIDTH, y))
            
        if self.level_num % 3 == 0: self.event_bus.publish("PLAY_SOUND", "boss")

    def load_next_level(self):
        self.level_num += 1
        self.load_level()
        self.player.health = min(100, self.player.health + 20)
        self.player.ammo = self.player.max_ammo

    def get_enemy_stats(self, etype):
        stats = {
            'grunt':     {'health':50,  'speed':1.2, 'damage':10, 'scale':1.0, 'applies_poison':False},
            'fast':      {'health':30,  'speed':2.5, 'damage':8,  'scale':0.8, 'applies_poison':False},
            'brute':     {'health':120, 'speed':0.7, 'damage':20, 'scale':1.4, 'applies_poison':False},
            'ghost':     {'health':40,  'speed':1.5, 'damage':15, 'scale':0.9, 'applies_poison':False},
            'zombie':    {'health':80,  'speed':0.8, 'damage':12, 'scale':1.1, 'applies_poison':True},
            'robot':     {'health':60,  'speed':1.8, 'damage':12, 'scale':1.0, 'applies_poison':False},
            'bat':       {'health':20,  'speed':3.5, 'damage':6,  'scale':0.7, 'applies_poison':False},
            'wolf':      {'health':65,  'speed':2.2, 'damage':15, 'scale':1.1, 'applies_poison':True},
            'spider':    {'health':35,  'speed':1.6, 'damage':8,  'scale':0.8, 'applies_poison':True},
            'snake':     {'health':40,  'speed':1.8, 'damage':10, 'scale':0.8, 'applies_poison':True},
            'vampire':   {'health':70,  'speed':1.4, 'damage':15, 'scale':1.1, 'applies_poison':False},
            'alien':     {'health':15,  'speed':3.0, 'damage':4,  'scale':0.6, 'applies_poison':False},
            'skeleton':  {'health':25,  'speed':1.8, 'damage':18, 'scale':0.9, 'applies_poison':False},
            'demon':     {'health':100, 'speed':1.5, 'damage':25, 'scale':1.2, 'applies_poison':False},
            'virus':     {'health':10,  'speed':4.0, 'damage':2,  'scale':0.5, 'applies_poison':True},
            'raccoon':   {'health':40,  'speed':2.8, 'damage':5,  'scale':0.8, 'applies_poison':False},
            'octopus':   {'health':60,  'speed':1.1, 'damage':10, 'scale':1.0, 'applies_poison':False},
            'snail':     {'health':200, 'speed':0.3, 'damage':30, 'scale':1.1, 'applies_poison':False},
            'wyvern':    {'health':150, 'speed':2.5, 'damage':20, 'scale':1.3, 'applies_poison':False},
            'boss':      {'health':400, 'speed':1.0, 'damage':25, 'scale':2.5, 'applies_poison':False},
            'bear':      {'health':250, 'speed':1.0, 'damage':35, 'scale':1.5, 'applies_poison':False},
            'mouse':     {'health':10,  'speed':3.8, 'damage':3,  'scale':0.4, 'applies_poison':False},
            'crocodile': {'health':90,  'speed':1.1, 'damage':22, 'scale':1.2, 'applies_poison':True},
            'trex':      {'health':300, 'speed':2.0, 'damage':40, 'scale':1.6, 'applies_poison':False},
            'frog':      {'health':30,  'speed':2.0, 'damage':8,  'scale':0.8, 'applies_poison':False},
            'bee':       {'health':15,  'speed':3.0, 'damage':5,  'scale':0.6, 'applies_poison':True},
            'crab':      {'health':60,  'speed':1.5, 'damage':10, 'scale':0.9, 'applies_poison':False},
            'deer':      {'health':50,  'speed':3.5, 'damage':0,  'scale':1.0, 'applies_poison':False},
            'jellyfish': {'health':25,  'speed':1.0, 'damage':5,  'scale':0.8, 'applies_poison':True},
            'mosquito':  {'health':10,  'speed':3.8, 'damage':3,  'scale':0.5, 'applies_poison':True},
            'rhino':     {'health':180, 'speed':1.2, 'damage':30, 'scale':1.4, 'applies_poison':False},
            'turtle':    {'health':150, 'speed':0.6, 'damage':15, 'scale':1.0, 'applies_poison':False},
            'llama':     {'health':60,  'speed':1.8, 'damage':5,  'scale':1.0, 'applies_poison':True},
            'tiger':     {'health':100, 'speed':2.8, 'damage':25, 'scale':1.2, 'applies_poison':False},
            'hippo':     {'health':220, 'speed':2.0, 'damage':35, 'scale':1.5, 'applies_poison':False},
            'otter':     {'health':40,  'speed':2.5, 'damage':8,  'scale':0.8, 'applies_poison':False},
            'hedgehog':  {'health':30,  'speed':1.5, 'damage':12, 'scale':0.7, 'applies_poison':False},
            'lobster':   {'health':120, 'speed':1.4, 'damage':18, 'scale':1.1, 'applies_poison':False},
            'cricket':   {'health':15,  'speed':2.5, 'damage':4,  'scale':0.5, 'applies_poison':False},
            'turkey':    {'health':40,  'speed':3.0, 'damage':0,  'scale':0.9, 'applies_poison':False}
        }
        return stats.get(etype, stats['grunt'])

    def shoot_hit_scan(self):
        best_dist, hit_enemy = MAX_DEPTH, None
        for e in self.enemies:
            if not e.alive: continue
            dx, dy = e.x - self.player.x, e.y - self.player.y
            dist = math.hypot(dx, dy)
            if dist < best_dist:
                angle_to = math.atan2(dy, dx)
                diff = (angle_to - self.player.angle + math.pi) % (2*math.pi) - math.pi
                if abs(diff) < 0.12: best_dist, hit_enemy = dist, e
                    
        if hit_enemy:
            self.crosshair_color = (255, 0, 0)
            self.hit_marker_timer = 0.1
            
            if hit_enemy.etype == 'otter' and random.random() < 0.33:
                self.crosshair_color = (200, 200, 200)
                self.event_bus.publish("PLAY_SOUND", "empty")
                return
                
            damage = 25 if self.player.berserk_timer <= 0 else 50
            
            # Luck Crit
            if self.player.luck_timer > 0 and random.random() < 0.25:
                damage *= 2
                self.hit_marker_timer = 0.2
                
            if hit_enemy.etype == 'turtle': damage *= 0.5
            
            hit_enemy.health -= damage
            hit_enemy.hit_flash = 0.2
            hit_enemy.alert_timer = 5.0
            self.event_bus.publish("PLAY_SOUND", "hit")
            
            # Apply Enemy Debuffs based on Player Buffs
            if self.player.explosive_timer > 0:
                hit_enemy.burn_timer = max(hit_enemy.burn_timer, 3.0)
            if self.player.focus_timer > 0:
                hit_enemy.choking_timer = max(hit_enemy.choking_timer, 2.0)
            
            for _ in range(8):
                ang, spd = random.random()*math.pi*2, random.uniform(1,3)
                self.particle_pool.spawn(hit_enemy.x, hit_enemy.y, math.cos(ang)*spd, math.sin(ang)*spd, 0.5, (200,50,50), random.randint(2,5))
            
            if self.player.explosive_timer > 0:
                self.event_bus.publish("PLAY_SOUND", "explosion")
                self.screen_shake = max(self.screen_shake, 0.3)
                for e2 in self.enemies:
                    if e2.alive and e2 != hit_enemy:
                        dist_aoe = math.hypot(e2.x - hit_enemy.x, e2.y - hit_enemy.y)
                        if dist_aoe < 2.5:
                            e2.health -= 15
                            e2.hit_flash = 0.2
                            e2.alert_timer = 5.0
                            e2.burn_timer = max(e2.burn_timer, 3.0) # Spread burning
                            if e2.health <= 0:
                                e2.alive = False
                                self.score += 50
                                self.event_bus.publish("ENEMY_DIED", e2)
                for _ in range(40):
                    ang, spd = random.random()*math.pi*2, random.uniform(1,6)
                    self.particle_pool.spawn(hit_enemy.x, hit_enemy.y, math.cos(ang)*spd, math.sin(ang)*spd, 0.8, (255, 150, 0), random.randint(4,7))
                
            if hit_enemy.health <= 0:
                hit_enemy.alive = False
                score_mult = 2 if self.player.luck_timer > 0 else 1
                self.score += {
                    'grunt':100, 'fast':150, 'brute':200, 'ghost':250, 'zombie':150, 'robot':200, 
                    'bat':120, 'wolf':220, 'spider':180, 'snake':160, 'vampire':250, 'boss':1000,
                    'alien':80, 'skeleton':150, 'demon':400, 'virus':60, 'raccoon':200, 'octopus':250, 'snail':300, 'wyvern':500,
                    'bear':600, 'mouse':50, 'crocodile':350, 'trex':800, 'frog':120, 'bee':80, 'crab':200, 'deer':400, 'jellyfish':150, 'mosquito':80,
                    'rhino':450, 'turtle':300, 'llama':150, 'tiger':500, 'hippo':700, 'otter':250, 'hedgehog':180, 'lobster':350, 'cricket':90, 'turkey':300
                }[hit_enemy.etype] * score_mult
                self.event_bus.publish("ENEMY_DIED", hit_enemy)
        else:
            self.crosshair_color = (255, 255, 255)

    def on_enemy_died(self, enemy):
        self.event_bus.publish("PLAY_SOUND", "death")
        for _ in range(20):
            ang, spd = random.random()*math.pi*2, random.uniform(2,5)
            self.particle_pool.spawn(enemy.x, enemy.y, math.cos(ang)*spd, math.sin(ang)*spd, 0.8, (150,30,30), random.randint(3,6))
            
        drops = 3 if enemy.etype == 'deer' else 2 if enemy.etype == 'turkey' else 1
        
        # Luck increases drop rate
        drop_chance = 0.7 if self.player.luck_timer > 0 else 0.5
        if enemy.etype == 'turkey': drop_chance = 1.0 # Turkeys always drop
        
        for _ in range(drops):
            if random.random() < drop_chance or enemy.etype == 'turkey':
                ptype = 'health' if enemy.etype == 'turkey' else ('health' if random.random() > 0.5 else 'ammo')
                self.pickups.append(Pickup(enemy.x + random.uniform(-0.5, 0.5), enemy.y + random.uniform(-0.5, 0.5), ptype))

    def on_pickup_particles(self, pk):
        col_map = {
            'health': (100,255,100), 'ammo': (255,200,0), 'treasure': (0,255,255), 'shield': (0,150,255),
            'speed': (255,255,0), 'regen': (50,255,50), 'berserk': (255,0,0), 'rapid_fire': (255,150,0), 
            'freeze': (0,255,255), 'focus': (200,200,255), 'magnet': (255,100,200), 'explosive': (255,80,0),
            'invisibility': (220,220,220), 'iron_skin': (200,200,200), 'luck': (0,255,100), 'potion': (200, 100, 255)
        }
        col = col_map.get(pk.ptype, (255,255,255))
        for _ in range(15):
            ang, spd = random.random()*math.pi*2, random.uniform(1,4)
            self.particle_pool.spawn(pk.x, pk.y, math.cos(ang)*spd, math.sin(ang)*spd, 0.4, col, 4)

    def render_world(self, screen):
        sx = random.randint(-int(self.screen_shake*10), int(self.screen_shake*10)) if self.screen_shake > 0 else 0
        sy = random.randint(-int(self.screen_shake*10), int(self.screen_shake*10)) if self.screen_shake > 0 else 0
        
        screen.blit(self.sky_surf, (0, sy))
        screen.blit(self.floor_surf, (0, HEIGHT//2 + sy))
        
        rays = cast_rays_dda(self.player, self.level, self.current_fov)
        slice_w = WIDTH / NUM_RAYS
        
        for i, (dist, _, side, hit_x) in enumerate(rays):
            wh = min(HEIGHT, (HEIGHT / dist) * 0.8)
            shade = max(0, 1.0 - (dist / MAX_DEPTH)) * (0.7 if side == 1 else 1.0)
            tex_x = int(hit_x * TEX_SIZE)
            tex_slice = self.level.wall_texture.subsurface((tex_x, 0, 1, TEX_SIZE))
            scaled_slice = pygame.transform.scale(tex_slice, (int(slice_w) + 1, int(wh)))
            shade_val = int(255 * shade)
            scaled_slice.fill((shade_val, shade_val, shade_val), special_flags=pygame.BLEND_MULT)
            screen.blit(scaled_slice, (i * slice_w + sx, (HEIGHT - wh) / 2 + sy))
            
        sprites = []
        sprites.extend([('prop', p, p.x, p.y) for p in self.props])
        sprites.extend([('torch', t, t.x, t.y) for t in self.torches])
        sprites.extend([('pickup', pk, pk.x, pk.y) for pk in self.pickups if pk.alive])
        sprites.extend([('enemy', e, e.x, e.y) for e in self.enemies if e.alive])
        sprites.sort(key=lambda s: -math.hypot(s[2]-self.player.x, s[3]-self.player.y))
        
        visible_torches = []
        for stype, obj, ox, oy in sprites:
            dx, dy = ox - self.player.x, oy - self.player.y
            dist = math.hypot(dx, dy)
            if dist < MAX_DEPTH:
                ang_to = math.atan2(dy, dx)
                ang_diff = (ang_to - self.player.angle + math.pi) % (2*math.pi) - math.pi
                if abs(ang_diff) < self.current_half_fov + 0.1:
                    sx_sc = (0.5 + (ang_diff / self.current_fov)) * WIDTH
                    scale = 500 / dist
                    
                    if stype == 'torch':
                        flicker = 0.9 + math.sin(obj.flicker) * 0.1
                        obj.flicker += self.dt * 15
                        sw, sh = max(1, int(self.torch_emoji.get_width() * scale / 10 * flicker)), max(1, int(self.torch_emoji.get_height() * scale / 10 * flicker))
                        img = pygame.transform.scale(self.torch_emoji, (sw, sh))
                        screen.blit(img, img.get_rect(center=(sx_sc, HEIGHT//2)))
                        visible_torches.append((sx_sc, HEIGHT//2, scale, flicker))
                        
                    elif stype == 'prop':
                        img = self.prop_emojis[obj.emoji_key]
                        sw, sh = max(1, int(img.get_width()*scale/15)), max(1, int(img.get_height()*scale/15))
                        scaled = pygame.transform.scale(img, (sw, sh))
                        screen.blit(scaled, scaled.get_rect(center=(sx_sc, HEIGHT//2)))
                        
                    elif stype == 'pickup':
                        scale = 300 / dist
                        bob = math.sin(obj.bob) * 5 * scale / 30
                        img = self.pickup_emojis[obj.ptype]
                        sw, sh = max(1, int(img.get_width()*scale/15)), max(1, int(img.get_height()*scale/15))
                        scaled = pygame.transform.scale(img, (sw, sh))
                        screen.blit(scaled, scaled.get_rect(center=(sx_sc, HEIGHT//2+bob)))
                        
                    elif stype == 'enemy':
                        sz = obj.scale * 10
                        img = self.enemy_emojis[obj.etype]
                        sw, sh = max(1, int(img.get_width()*scale/sz)), max(1, int(img.get_height()*scale/sz))
                        scaled = pygame.transform.scale(img, (sw, sh))
                        
                        if obj.etype in ['ghost', 'bee', 'jellyfish', 'mosquito']:
                            y_off = math.sin(pygame.time.get_ticks()*0.005)*10
                        elif obj.etype in ['frog', 'cricket']:
                            y_off = -abs(math.sin(pygame.time.get_ticks() * (0.006 if obj.etype=='frog' else 0.012))) * 15
                        else:
                            y_off = 0
                            
                        if obj.hit_flash > 0:
                            tint = pygame.Surface((sw, sh), pygame.SRCALPHA); tint.fill((255,100,100,150))
                            scaled = scaled.copy(); scaled.blit(tint, (0,0), special_flags=pygame.BLEND_RGBA_MULT)
                        
                        if self.freeze_timer > 0:
                            tint = pygame.Surface((sw, sh), pygame.SRCALPHA); tint.fill((100,150,255,100))
                            scaled = scaled.copy(); scaled.blit(tint, (0,0), special_flags=pygame.BLEND_RGBA_MULT)
                            
                        # Enraged tint
                        if obj.is_enraged:
                            tint = pygame.Surface((sw, sh), pygame.SRCALPHA); tint.fill((255,0,0,100))
                            scaled = scaled.copy(); scaled.blit(tint, (0,0), special_flags=pygame.BLEND_RGBA_MULT)
                            
                        screen.blit(scaled, scaled.get_rect(center=(sx_sc, HEIGHT//2+y_off)))
                        
                        if obj.alert_timer > 4.0 and obj.etype not in ['deer', 'turkey']:
                            alert_img = self.alert_emoji
                            asw, ash = max(1, int(alert_img.get_width()*scale/20)), max(1, int(alert_img.get_height()*scale/20))
                            alert_scaled = pygame.transform.scale(alert_img, (asw, ash))
                            screen.blit(alert_scaled, alert_scaled.get_rect(center=(sx_sc, HEIGHT//2+y_off-sh//2-20)))
                            
                        max_hp = self.get_enemy_stats(obj.etype)['health']
                        if obj.health < max_hp:
                            pygame.draw.rect(screen, (50,0,0), (sx_sc-sw//2, HEIGHT//2+y_off-sh//2-10, sw, 5))
                            pygame.draw.rect(screen, (255,0,0), (sx_sc-sw//2, HEIGHT//2+y_off-sh//2-10, sw*(obj.health/max_hp), 5))
        
        for p in self.particle_pool.pool:
            if p['active']:
                dx, dy = p['x'] - self.player.x, p['y'] - self.player.y
                dist = math.hypot(dx, dy)
                if dist < MAX_DEPTH:
                    ang_to = math.atan2(dy, dx)
                    ang_diff = (ang_to - self.player.angle + math.pi) % (2*math.pi) - math.pi
                    if abs(ang_diff) < self.current_half_fov:
                        sx_p = (0.5 + (ang_diff / self.current_fov)) * WIDTH
                        scale = 300 / dist
                        alpha = p['life'] / p['max_life']
                        sz = max(1, int(p['size'] * scale))
                        col = tuple(int(c * alpha) for c in p['color'])
                        pygame.draw.circle(screen, col, (int(sx_p), HEIGHT//2), sz)
        
        self.darkness_surf.fill((0, 0, 0, self.level.darkness))
        self.darkness_surf.blit(self.mask_player, (WIDTH//2-250+sx, HEIGHT//2-250+sy), special_flags=pygame.BLEND_RGBA_SUB)
        if self.muzzle_flash_timer > 0:
            self.darkness_surf.blit(self.mask_flash, (WIDTH//2-400+sx, HEIGHT//2-400+sy), special_flags=pygame.BLEND_RGBA_SUB)
        for tx, ty, tscale, flicker in visible_torches:
            r = int(100 * tscale / 10 * flicker)
            if r > 10:
                scaled_tmask = pygame.transform.scale(self.mask_torch, (r*2, r*2))
                self.darkness_surf.blit(scaled_tmask, (tx-r+sx, ty-r+sy), special_flags=pygame.BLEND_RGBA_SUB)
        screen.blit(self.darkness_surf, (0, 0))
        
        self.tint_surf.fill((0, 0, 0, 0))
        for tx, ty, tscale, flicker in visible_torches:
            r = int(60 * tscale / 10 * flicker)
            if r > 10:
                for rad in range(r, 0, -5):
                    a = int(80 * (1 - rad/r) * flicker)
                    pygame.draw.circle(self.tint_surf, (255, 150, 50, a), (int(tx), int(ty)), rad)
        if self.muzzle_flash_timer > 0:
            for rad in range(300, 0, -20):
                a = int(100 * (1 - rad/300))
                pygame.draw.circle(self.tint_surf, (255, 230, 150, a), (WIDTH//2, HEIGHT//2), rad)
        screen.blit(self.tint_surf, (0, 0), special_flags=pygame.BLEND_ADD)
        
        # Screen Overlays for Status Effects
        if self.player.berserk_timer > 0:
            pulse = (math.sin(pygame.time.get_ticks() * 0.01) + 1) / 2
            overlay = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
            overlay.fill((255, 0, 0, int(40 * pulse)))
            screen.blit(overlay, (0, 0))
        if self.freeze_timer > 0:
            overlay = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 150, 255, 40))
            screen.blit(overlay, (0, 0))
        if self.player.focus_timer > 0:
            overlay = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
            pygame.draw.rect(overlay, (0, 0, 0, 150), (0, 0, WIDTH, HEIGHT//4))
            pygame.draw.rect(overlay, (0, 0, 0, 150), (0, HEIGHT - HEIGHT//4, WIDTH, HEIGHT//4))
            pygame.draw.rect(overlay, (0, 0, 0, 150), (0, 0, WIDTH//6, HEIGHT))
            pygame.draw.rect(overlay, (0, 0, 0, 150), (WIDTH - WIDTH//6, 0, WIDTH//6, HEIGHT))
            screen.blit(overlay, (0, 0))
        if self.player.iron_skin_timer > 0:
            overlay = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
            overlay.fill((200, 200, 200, 30))
            screen.blit(overlay, (0, 0))
        if self.player.invis_timer > 0:
            overlay = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
            overlay.fill((200, 200, 255, 20))
            screen.blit(overlay, (0, 0))
        if self.player.bleed_timer > 0:
            pulse = (math.sin(pygame.time.get_ticks() * 0.02) + 1) / 2
            alpha = int(100 + 50 * pulse)
            vignette = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
            pygame.draw.circle(vignette, (255,0,0,0), (WIDTH//2, HEIGHT//2), HEIGHT//2)
            vignette.fill((150,0,0,alpha), special_flags=pygame.BLEND_RGBA_SUB)
            screen.blit(vignette, (0,0))
        if self.player.dizzy_timer > 0:
            overlay = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
            overlay.fill((255, 255, 0, 20))
            screen.blit(overlay, (0, 0))
        if self.player.webbed_timer > 0:
            overlay = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 50, 0, 60))
            screen.blit(overlay, (0, 0))

    def render_hud(self, screen):
        bob = math.sin(pygame.time.get_ticks() * 0.01) * 5 if self.player.move_x != 0 or self.player.move_y != 0 else 0
        kick = self.player.gun_kickback * 20
        screen.blit(self.gun_emoji, self.gun_emoji.get_rect(center=(WIDTH//2, HEIGHT - 80 + bob + kick)))
        if self.muzzle_flash_timer > 0:
            screen.blit(self.flash_emoji, self.flash_emoji.get_rect(center=(WIDTH//2, HEIGHT - 120 + bob + kick)))
            
        face_idx = min(4, int(self.player.health / 25))
        screen.blit(self.face_emojis[face_idx], (20, 15))
        
        status_x = 75
        status_y = 25
        active_effects = []
        if self.player.shield > 0: active_effects.append(('🛡️', self.player.shield))
        if self.player.speed_boost_timer > 0: active_effects.append(('⚡', self.player.speed_boost_timer))
        if self.player.regen_timer > 0: active_effects.append(('🍃', self.player.regen_timer))
        if self.player.berserk_timer > 0: active_effects.append(('🩸', self.player.berserk_timer))
        if self.player.rapid_fire_timer > 0: active_effects.append(('🚀', self.player.rapid_fire_timer))
        if self.player.poison_timer > 0: active_effects.append(('🤢', self.player.poison_timer))
        if self.freeze_timer > 0: active_effects.append(('❄️', self.freeze_timer))
        if self.player.focus_timer > 0: active_effects.append(('🦅', self.player.focus_timer))
        if self.player.magnet_timer > 0: active_effects.append(('🧲', self.player.magnet_timer))
        if self.player.explosive_timer > 0: active_effects.append(('💣', self.player.explosive_timer))
        if self.player.slow_timer > 0: active_effects.append(('🐌', self.player.slow_timer))
        if self.player.invis_timer > 0: active_effects.append(('🕶️', self.player.invis_timer))
        if self.player.iron_skin_timer > 0: active_effects.append(('✊', self.player.iron_skin_timer))
        if self.player.luck_timer > 0: active_effects.append(('🍀', self.player.luck_timer))
        if self.player.bleed_timer > 0: active_effects.append(('🩸', self.player.bleed_timer))
        if self.player.webbed_timer > 0: active_effects.append(('🕸️', self.player.webbed_timer))
        if self.player.dizzy_timer > 0: active_effects.append(('😵', self.player.dizzy_timer))
        
        for emoji_str, timer in active_effects:
            img = self.font_large.render(emoji_str, True, (255, 255, 255))
            screen.blit(img, (status_x, status_y))
            if isinstance(timer, float):
                txt = self.font_small.render(f"{timer:.1f}s", True, (255,255,255))
                screen.blit(txt, (status_x + 30, status_y + 10))
            status_x += 70
            
        screen.blit(self.font_hud.render("AMMO", True, (255,255,255)), (20, HEIGHT-40))
        for i in range(self.player.ammo): pygame.draw.rect(screen, (255,200,0), (90+i*12, HEIGHT-32, 8, 16))
        if self.player.reload_timer > 0:
            screen.blit(self.font_small.render("RELOADING...", True, (255,200,0)), (90, HEIGHT-50))
            
        screen.blit(self.font_hud.render(f"SCORE: {self.score}", True, (255,255,255)), (WIDTH-180, 30))
        screen.blit(self.font_small.render(f"HI: {self.high_score}", True, (200,200,0)), (WIDTH-180, 60))
        screen.blit(self.font_hud.render(f"LV:{self.level_num} - {self.level.name}", True, (200,200,200)), (WIDTH-250, 85))
        
        if self.level_num % 3 == 0 and any(e.etype == 'boss' and e.alive for e in self.enemies):
            boss_text = self.font_large.render("⚠️ BOSS ⚠️", True, (255,50,50))
            screen.blit(boss_text, (WIDTH//2 - boss_text.get_width()//2, 20))
            
        self.crosshair_size = 10 + int(self.player.shoot_cooldown * 30)
        cx, cy = WIDTH//2, HEIGHT//2
        pygame.draw.line(screen, self.crosshair_color, (cx-self.crosshair_size, cy), (cx-5, cy), 2)
        pygame.draw.line(screen, self.crosshair_color, (cx+5, cy), (cx+self.crosshair_size, cy), 2)
        pygame.draw.line(screen, self.crosshair_color, (cx, cy-self.crosshair_size), (cx, cy-5), 2)
        pygame.draw.line(screen, self.crosshair_color, (cx, cy+5), (cx, cy+self.crosshair_size), 2)
        
        if self.hit_marker_timer > 0:
            screen.blit(self.hit_marker, self.hit_marker.get_rect(center=(cx, cy)))
        self.crosshair_color = (255, 255, 255)
        
        mm_x, mm_y = WIDTH-MINIMAP_SIZE-10, HEIGHT-MINIMAP_SIZE-10
        pygame.draw.rect(screen, (0,0,0), (mm_x-2, mm_y-2, MINIMAP_SIZE+4, MINIMAP_SIZE+4))
        ts = MINIMAP_SIZE / max(self.level.width, self.level.height)
        for y in range(self.level.height):
            for x in range(self.level.width):
                if self.explored[y][x]:
                    col = (40,40,40) if self.level.grid[y][x]=='.' else (120,120,120)
                    pygame.draw.rect(screen, col, (mm_x+x*ts, mm_y+y*ts, ts+1, ts+1))
        px_mm, py_mm = mm_x+self.player.x*ts, mm_y+self.player.y*ts
        pygame.draw.circle(screen, (0,255,0), (int(px_mm), int(py_mm)), 3)
        pygame.draw.line(screen, (0,255,0), (px_mm, py_mm), (px_mm+math.cos(self.player.angle)*6, py_mm+math.sin(self.player.angle)*6), 2)
        for e in self.enemies:
            if e.alive and self.explored[int(e.y)][int(e.x)]:
                pygame.draw.circle(screen, (255,0,0) if e.etype!='boss' else (255,100,255), (int(mm_x+e.x*ts), int(mm_y+e.y*ts)), 2)
        for pk in self.pickups:
            if pk.alive and self.explored[int(pk.y)][int(pk.x)]:
                col = (0,255,0) if pk.ptype=='health' else (255,200,0) if pk.ptype=='ammo' else (0,255,255)
                pygame.draw.circle(screen, col, (int(mm_x+pk.x*ts), int(mm_y+pk.y*ts)), 2)
                
        if self.damage_flash > 0:
            overlay = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
            overlay.fill((255, 0, 0, int(150 * (self.damage_flash / 0.3))))
            screen.blit(overlay, (0, 0))
        if self.player.health < 30 and self.player.health > 0:
            pulse = (math.sin(pygame.time.get_ticks() * 0.005) + 1) / 2
            alpha = int(80 * pulse * (1 - self.player.health / 30))
            vignette = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
            pygame.draw.circle(vignette, (255,0,0,0), (WIDTH//2, HEIGHT//2), HEIGHT//3)
            vignette.fill((200,0,0,alpha), special_flags=pygame.BLEND_RGBA_SUB)
            screen.blit(vignette, (0,0))

# ==========================================
# 10. MAIN ENTRY POINT
# ==========================================
def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Emoji FPS - Ultimate Status Effects Edition")
    clock = pygame.time.Clock()
    
    global context
    context = GameContext()
    
    emoji_font = "segoeuiemoji,applecoloremoji,notocoloremoji,arial"
    context.font_large = pygame.font.SysFont(emoji_font, 64)
    context.font_huge = pygame.font.SysFont(emoji_font, 150)
    context.font_hud = pygame.font.SysFont("monospace", 24, bold=True)
    context.font_small = pygame.font.SysFont("monospace", 16)
    
    context.enemy_emojis = {
        'grunt': context.font_huge.render("👽", True, (255, 255, 255)),
        'fast': context.font_huge.render("🦂", True, (255, 255, 255)),
        'brute': context.font_huge.render("👹", True, (255, 255, 255)),
        'ghost': context.font_huge.render("👻", True, (255, 255, 255)),
        'zombie': context.font_huge.render("🧟", True, (255, 255, 255)),
        'robot': context.font_huge.render("🤖", True, (255, 255, 255)),
        'bat': context.font_huge.render("🦇", True, (255, 255, 255)),
        'wolf': context.font_huge.render("🐺", True, (255, 255, 255)),
        'spider': context.font_huge.render("🕷️", True, (255, 255, 255)),
        'snake': context.font_huge.render("🐍", True, (255, 255, 255)),
        'vampire': context.font_huge.render("🧛", True, (255, 255, 255)),
        'alien': context.font_huge.render("👾", True, (255, 255, 255)),
        'skeleton': context.font_huge.render("☠️", True, (255, 255, 255)),
        'demon': context.font_huge.render("😈", True, (255, 255, 255)),
        'virus': context.font_huge.render("🦠", True, (255, 255, 255)),
        'raccoon': context.font_huge.render("🦝", True, (255, 255, 255)),
        'octopus': context.font_huge.render("🐙", True, (255, 255, 255)),
        'snail': context.font_huge.render("🐌", True, (255, 255, 255)),
        'wyvern': context.font_huge.render("🐲", True, (255, 255, 255)),
        'boss': context.font_huge.render("🐉", True, (255, 255, 255)),
        'bear': context.font_huge.render("🐻", True, (255, 255, 255)),
        'mouse': context.font_huge.render("🐭", True, (255, 255, 255)),
        'crocodile': context.font_huge.render("🐊", True, (255, 255, 255)),
        'trex': context.font_huge.render("🦖", True, (255, 255, 255)),
        'frog': context.font_huge.render("🐸", True, (255, 255, 255)),
        'bee': context.font_huge.render("🐝", True, (255, 255, 255)),
        'crab': context.font_huge.render("🦀", True, (255, 255, 255)),
        'deer': context.font_huge.render("🦌", True, (255, 255, 255)),
        'jellyfish': context.font_huge.render("🪼", True, (255, 255, 255)),
        'mosquito': context.font_huge.render("🦟", True, (255, 255, 255)),
        'rhino': context.font_huge.render("🦏", True, (255, 255, 255)),
        'turtle': context.font_huge.render("🐢", True, (255, 255, 255)),
        'llama': context.font_huge.render("🦙", True, (255, 255, 255)),
        'tiger': context.font_huge.render("🐯", True, (255, 255, 255)),
        'hippo': context.font_huge.render("🦛", True, (255, 255, 255)),
        'otter': context.font_huge.render("🦦", True, (255, 255, 255)),
        'hedgehog': context.font_huge.render("🦔", True, (255, 255, 255)),
        'lobster': context.font_huge.render("🦞", True, (255, 255, 255)),
        'cricket': context.font_huge.render("🦗", True, (255, 255, 255)),
        'turkey': context.font_huge.render("🦃", True, (255, 255, 255))
    }
    context.prop_emojis = {
        'crate': context.font_large.render("📦", True, (255, 255, 255)),
        'bone': context.font_large.render("🦴", True, (255, 255, 255)),
        'tree': context.font_large.render("🌲", True, (255, 255, 255)),
        'cobweb': context.font_large.render("🕸️", True, (255, 255, 255)),
        'rock': context.font_large.render("🪨", True, (255, 255, 255)),
        'vase': context.font_large.render("🏺", True, (255, 255, 255)),
        'gravestone': context.font_large.render("🪦", True, (255, 255, 255)),
        'barrel': context.font_large.render("🛢️", True, (255, 255, 255))
    }
    context.pickup_emojis = {
        'health': context.font_large.render("🍎", True, (255, 255, 255)),
        'ammo': context.font_large.render("💥", True, (255, 255, 255)),
        'treasure': context.font_large.render("💎", True, (255, 255, 255)),
        'shield': context.font_large.render("🛡️", True, (255, 255, 255)),
        'speed': context.font_large.render("⚡", True, (255, 255, 255)),
        'regen': context.font_large.render("🍃", True, (255, 255, 255)),
        'berserk': context.font_large.render("🩸", True, (255, 255, 255)),
        'rapid_fire': context.font_large.render("🚀", True, (255, 255, 255)),
        'freeze': context.font_large.render("❄️", True, (255, 255, 255)),
        'focus': context.font_large.render("🦅", True, (255, 255, 255)),
        'magnet': context.font_large.render("🧲", True, (255, 255, 255)),
        'explosive': context.font_large.render("💣", True, (255, 255, 255)),
        'invisibility': context.font_large.render("🕶️", True, (255, 255, 255)),
        'iron_skin': context.font_large.render("✊", True, (255, 255, 255)),
        'luck': context.font_large.render("🍀", True, (255, 255, 255)),
        'potion': context.font_large.render("🧪", True, (255, 255, 255))
    }
    context.gun_emoji = context.font_huge.render("🔫", True, (255, 255, 255))
    context.flash_emoji = context.font_huge.render("💥", True, (255, 255, 255))
    context.skull_emoji = context.font_huge.render("💀", True, (255, 255, 255))
    context.torch_emoji = context.font_large.render("🔥", True, (255, 255, 255))
    context.alert_emoji = context.font_large.render("💢", True, (255, 255, 255))
    context.hit_marker = context.font_large.render("🎯", True, (255, 255, 255))
    context.face_emojis = [
        context.font_large.render("💀", True, (255, 255, 255)),
        context.font_large.render("🤕", True, (255, 255, 255)),
        context.font_large.render("😵", True, (255, 255, 255)),
        context.font_large.render("😐", True, (255, 255, 255)),
        context.font_large.render("😎", True, (255, 255, 255))
    ]
    
    def create_radial_mask(radius, max_alpha=255):
        surf = pygame.Surface((radius*2, radius*2), pygame.SRCALPHA)
        for r in range(radius, 0, -2):
            alpha = int(max_alpha * (1 - r/radius)**1.5)
            pygame.draw.circle(surf, (255, 255, 255, alpha), (radius, radius), r)
        return surf
    context.mask_player = create_radial_mask(250, 220)
    context.mask_torch = create_radial_mask(100, 200)
    context.mask_flash = create_radial_mask(400, 255)
    
    key_map = {
        pygame.K_SPACE: CmdShoot(), pygame.K_r: CmdReload(),
        pygame.K_p: CmdTogglePause(), pygame.K_m: CmdToggleMouseLook()
    }
    
    running = True
    while running:
        context.dt = clock.tick(60) / 1000.0
        if context.dt > 0.1: context.dt = 0.1
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT: running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    if isinstance(context.state, PausedState) or isinstance(context.state, GameOverState): running = False
                    else: context.state = PausedState()
                elif event.key == pygame.K_r and isinstance(context.state, GameOverState):
                    context.level_num = 1; context.score = 0; context.load_level(); context.state = PlayingState()
                else:
                    cmd = key_map.get(event.key)
                    if cmd:
                        res = cmd.execute(context)
                        if res == "PAUSE":
                            context.state = PausedState() if not isinstance(context.state, PausedState) else PlayingState()
        
        if isinstance(context.state, GameOverState) or isinstance(context.state, PlayingState):
            context.state.handle_input(context)
        
        context.state.update(context, context.dt)
        
        screen.fill((0, 0, 0))
        context.state.draw(screen, context)
        
        if context.player.health <= 0 and not isinstance(context.state, GameOverState):
            context.state = GameOverState()
            
        pygame.display.flip()

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()