import pygame
import numpy as np
import random
import time
import pickle
import os
import matplotlib.pyplot as plt
from collections import defaultdict, deque

# ==========================================
# 1. KONFIGURASI (SETTINGS)
# ==========================================
class Config:
    # Dimensi
    GRID_W = 10
    GRID_H = 10
    CELL_SIZE = 48
    MARGIN = 2
    PANEL_WIDTH = 400
    
    WINDOW_W = GRID_W * (CELL_SIZE + MARGIN) + PANEL_WIDTH
    WINDOW_H = GRID_H * (CELL_SIZE + MARGIN)

    # Warna (R, G, B)
    COLOR_BG = (30, 30, 30)
    COLOR_GRID = (220, 220, 220)
    COLOR_CLEAN = (255, 255, 255)
    COLOR_ROBOT = (0, 120, 255)
    COLOR_CHARGER = (255, 215, 0)
    COLOR_TEXT = (240, 240, 240)
    COLOR_BTN_IDLE = (60, 60, 60)
    COLOR_OVERLAY = (0, 0, 0, 210)

    # Robot & Baterai
    BATTERY_MAX = 100.0
    BATTERY_COST = 0.6
    CHARGE_SPEED = 12.0
    LIMIT_LOW = 30.0  # Batas bawah untuk mulai ngecas
    LIMIT_HIGH = 95.0 # Batas atas untuk berhenti ngecas

    # Posisi
    START_POS = (0, 0)
    CHARGERS = [(8, 6)]

    # Q-Learning Hyperparameters
    ALPHA = 0.2         # Learning Rate
    GAMMA = 0.95        # Discount Factor
    EPS_START = 1.0     # Explorasi awal
    EPS_END = 0.01      # Explorasi akhir
    EPS_DECAY = 0.9995  # Penurunan epsilon
    N_EPISODES = 20000 
    
    # Rewards
    REW_CLEAN = 25.0
    REW_MOVE_CLEAN = -0.5
    REW_WALL = -5.0
    REW_WIN = 1000.0
    REW_DEAD = -100.0
    REW_FIRST_CHARGE = 100.0
    REW_CHARGING_PROCESS = 2.0
    REW_BACKTRACK = -2.5

    ACTIONS = ['up', 'down', 'left', 'right', 'charge']
    QTABLE_FILE = "robot_brain_smart.pkl"


# ==========================================
# 2. ENVIRONMENT
# ==========================================
class SweepEnv:
    def __init__(self):
        self.grid = None
        self.robot_pos = None
        self.battery = 0
        self.steps = 0
        self.reset()

    def reset(self):
        self.grid = np.ones((Config.GRID_H, Config.GRID_W), dtype=np.uint8)
        self.robot_pos = list(Config.START_POS)
        self.prev_pos = None
        self.battery = Config.BATTERY_MAX
        self.has_charged = False
        self.charging_phase = False
        
        # Bersihkan posisi awal
        sx, sy = Config.START_POS
        self.grid[sy, sx] = 0
        
        self.steps = 0
        return self.get_state()

    def _get_dist_charger(self):
        """Menghitung jarak Manhattan ke charger terdekat"""
        x, y = self.robot_pos
        cx, cy = Config.CHARGERS[0]
        return abs(x - cx) + abs(y - cy)

    def get_surrounding_info(self):
        """Sensor sederhana 4 arah (0-15)"""
        x, y = self.robot_pos
        sensor = 0
        # Cek Atas, Bawah, Kiri, Kanan
        if y > 0 and self.grid[y-1][x] == 1: sensor += 1
        if y < Config.GRID_H-1 and self.grid[y+1][x] == 1: sensor += 2
        if x > 0 and self.grid[y][x-1] == 1: sensor += 4
        if x < Config.GRID_W-1 and self.grid[y][x+1] == 1: sensor += 8
        return sensor

    def get_state(self):
        """Mengembalikan tuple state untuk Q-Table"""
        x, y = self.robot_pos
        
        # Discretize Battery Level
        if self.battery < Config.LIMIT_LOW: batt_state = 0
        elif self.battery < 70: batt_state = 1
        else: batt_state = 2
        
        sensor = self.get_surrounding_info()
        return (x, y, batt_state, self.charging_phase, self.has_charged, sensor)

    def step(self, action_idx):
        action = Config.ACTIONS[action_idx]
        x, y = self.robot_pos
        reward = -0.1
        done = False
        reason = ""

        dist_before = self._get_dist_charger()

        # Logika Trigger Charging Phase
        if (x, y) in Config.CHARGERS and self.battery < Config.LIMIT_LOW:
            self.charging_phase = True
        
        if self.battery >= Config.LIMIT_HIGH:
            self.charging_phase = False

        # --- EKSEKUSI AKSI ---
        if self.charging_phase:
            if action == 'charge':
                if self.battery < Config.BATTERY_MAX:
                    self.battery = min(Config.BATTERY_MAX, self.battery + Config.CHARGE_SPEED)
                    reward = Config.REW_CHARGING_PROCESS
                    if not self.has_charged:
                        reward += Config.REW_FIRST_CHARGE
                        self.has_charged = True
                    self.prev_pos = None 
                else:
                    self.charging_phase = False
            else:
                reward = -20.0 # Hukuman jika kabur saat harus charge
        else:
            if action == 'charge':
                reward = -5.0 # Hukuman charge sembarangan
            else:
                # Pergerakan
                nx, ny = x, y
                if action == 'up': ny -= 1
                elif action == 'down': ny += 1
                elif action == 'left': nx -= 1
                elif action == 'right': nx += 1
                
                # Cek Tembok
                if nx < 0 or nx >= Config.GRID_W or ny < 0 or ny >= Config.GRID_H:
                    reward = Config.REW_WALL
                else:
                    is_backtracking = (self.prev_pos is not None and (nx, ny) == tuple(self.prev_pos))
                    
                    self.prev_pos = list(self.robot_pos)
                    self.robot_pos = [nx, ny]
                    self.battery -= Config.BATTERY_COST
                    
                    is_dirty = (self.grid[ny][nx] == 1)
                    dist_now = self._get_dist_charger()

                    # Reward Shaping
                    if self.battery < Config.LIMIT_LOW:
                        # Prioritas: Ke Charger
                        if dist_now < dist_before: reward = 5.0
                        else: reward = -5.0
                        if is_dirty:
                            self.grid[ny][nx] = 0
                            reward += 2.0
                    else:
                        # Prioritas: Bersih-bersih
                        if is_dirty:
                            self.grid[ny][nx] = 0
                            reward = Config.REW_CLEAN
                        else:
                            reward = Config.REW_MOVE_CLEAN
                            if is_backtracking:
                                reward += Config.REW_BACKTRACK

        # Cek Kondisi Terminal
        dirty_cnt = np.sum(self.grid)
        
        if self.battery <= 0:
            done = True
            reward = Config.REW_DEAD
            reason = "BATTERY DEAD"
        elif dirty_cnt == 0:
            if self.has_charged:
                done = True
                reward = Config.REW_WIN
                reason = "MISSION COMPLETE"
            else:
                reward = -2.0 # Penalti kecil jika bersih tapi belum pernah charge (opsional)

        self.steps += 1
        if self.steps > 2500:
            done = True
            reward = -50
            reason = "TIMEOUT / STUCK"

        return self.get_state(), reward, done, dirty_cnt, reason


# ==========================================
# 3. AGENT (BRAIN)
# ==========================================
class QAgent:
    def __init__(self):
        self.n_actions = len(Config.ACTIONS)
        self.q_table = defaultdict(lambda: np.zeros(self.n_actions))
        self.epsilon = Config.EPS_START

    def get_action(self, state, force_greedy=False):
        if not force_greedy and random.random() < self.epsilon:
            return random.randint(0, self.n_actions - 1)
        return np.argmax(self.q_table[state])

    def update(self, s, a, r, s_next, done):
        old_value = self.q_table[s][a]
        next_max = np.max(self.q_table[s_next])
        
        # Rumus Bellman Equation
        target = r + Config.GAMMA * next_max * (not done)
        self.q_table[s][a] = old_value + Config.ALPHA * (target - old_value)

    def save(self):
        with open(Config.QTABLE_FILE, 'wb') as f:
            pickle.dump(dict(self.q_table), f)
        print("Brain Saved.")

    def load(self):
        if os.path.exists(Config.QTABLE_FILE):
            with open(Config.QTABLE_FILE, 'rb') as f:
                self.q_table = defaultdict(lambda: np.zeros(self.n_actions), pickle.load(f))
            self.epsilon = Config.EPS_END
            print("Brain Loaded.")
            return True
        return False


# ==========================================
# 4. VISUALIZER (UI)
# ==========================================
class Visualizer:
    def __init__(self, screen):
        self.screen = screen
        self.font_main = pygame.font.SysFont("Consolas", 20, bold=True)
        self.font_title = pygame.font.SysFont("Consolas", 24, bold=True)
        
        # UI Elements
        x_panel = Config.WINDOW_W - Config.PANEL_WIDTH + 20
        self.btn_train = pygame.Rect(x_panel, 350, 150, 40)
        self.btn_sim = pygame.Rect(x_panel + 180, 350, 150, 40)
        self.btn_graph = pygame.Rect(x_panel, 410, 330, 40)

    def draw_text(self, text, x, y, size=20, color=Config.COLOR_TEXT, center=False):
        font = pygame.font.SysFont("Consolas", size, bold=True)
        img = font.render(text, True, color)
        if center:
            rect = img.get_rect(center=(x, y))
            self.screen.blit(img, rect)
        else:
            self.screen.blit(img, (x, y))

    def render_env(self, env):
        # Draw Grid
        for y in range(Config.GRID_H):
            for x in range(Config.GRID_W):
                rect = (x*(Config.CELL_SIZE+Config.MARGIN), 
                        y*(Config.CELL_SIZE+Config.MARGIN), 
                        Config.CELL_SIZE, Config.CELL_SIZE)
                
                # Warna Lantai (Kotor vs Bersih)
                color = Config.COLOR_GRID if env.grid[y][x] else Config.COLOR_CLEAN
                pygame.draw.rect(self.screen, color, rect)
                
                # Draw Charger
                if (x, y) in Config.CHARGERS:
                    cx, cy = rect[0] + 24, rect[1] + 24
                    pygame.draw.circle(self.screen, Config.COLOR_CHARGER, (cx, cy), 12)
                
                # Draw Robot
                if (x, y) == tuple(env.robot_pos):
                    rx, ry = rect[0] + 24, rect[1] + 24
                    pygame.draw.circle(self.screen, Config.COLOR_ROBOT, (rx, ry), 20)
                    
                    # Indikator status di atas kepala robot
                    if env.battery < Config.LIMIT_LOW: ind_col = (255, 0, 0)
                    elif env.charging_phase: ind_col = (0, 0, 255)
                    else: ind_col = (0, 255, 0)
                    pygame.draw.circle(self.screen, ind_col, (rx, ry), 6)

    def render_panel(self, env, agent, mode, episode):
        px = Config.WINDOW_W - Config.PANEL_WIDTH + 20
        
        self.draw_text("CONTROL PANEL", px, 20, 24, (0, 200, 255))
        self.draw_text(f"Status  : {mode}", px, 60)
        self.draw_text(f"Episode : {episode}", px, 90)
        self.draw_text(f"Epsilon : {agent.epsilon:.3f}", px, 120)

        py = 180
        pygame.draw.line(self.screen, (100,100,100), (px, py), (px+350, py), 2)
        
        # Info Baterai
        batt_color = (255, 50, 50) if env.battery < Config.LIMIT_LOW else (255, 255, 0)
        self.draw_text(f"BATERAI: {env.battery:.1f}%", px, py+20, 26, batt_color)
        
        # Status Text
        if env.charging_phase:
            msg, col = "CHARGING...", (100, 100, 255)
        elif env.battery < Config.LIMIT_LOW:
            msg, col = "LOW BATTERY!", (255, 50, 50)
        else:
            msg, col = "CLEANING", (50, 255, 50)
        self.draw_text(msg, px, py+55, 18, col)
        
        # Persentase Kebersihan
        dirt_pct = 100 - np.sum(env.grid)
        self.draw_text(f"BERSIH: {dirt_pct}%", px, py+100, 22)

        # Draw Buttons
        c_train = (0, 150, 0) if mode == "FAST_TRAIN" else Config.COLOR_BTN_IDLE
        pygame.draw.rect(self.screen, c_train, self.btn_train, border_radius=5)
        txt_train = "RESUME" if episode > 0 else "START TRAIN"
        self.draw_text(txt_train, self.btn_train.x + 20, self.btn_train.y + 10)

        c_sim = (0, 100, 150) if mode == "WATCH_LEARNING" else Config.COLOR_BTN_IDLE
        pygame.draw.rect(self.screen, c_sim, self.btn_sim, border_radius=5)
        self.draw_text("WATCH 1 EP", self.btn_sim.x + 25, self.btn_sim.y + 10)

        pygame.draw.rect(self.screen, (150, 100, 0), self.btn_graph, border_radius=5)
        self.draw_text("Lihat Grafik", self.btn_graph.x + 100, self.btn_graph.y + 10)

    def show_popup(self, episode, reason, dirt_pct, steps):
        overlay = pygame.Surface((Config.WINDOW_W, Config.WINDOW_H), pygame.SRCALPHA)
        overlay.fill(Config.COLOR_OVERLAY)
        self.screen.blit(overlay, (0,0))
        
        # Kotak Pop-up
        w, h = 420, 320
        bx, by = (Config.WINDOW_W - w) // 2, (Config.WINDOW_H - h) // 2
        
        pygame.draw.rect(self.screen, (40, 40, 50), (bx, by, w, h), border_radius=12)
        pygame.draw.rect(self.screen, (100, 200, 255), (bx, by, w, h), 2, border_radius=12)
        
        self.draw_text("EPISODE SELESAI", bx + w//2, by + 25, 28, (255, 200, 0), center=True)
        pygame.draw.line(self.screen, (100,100,100), (bx+20, by+60), (bx+w-20, by+60), 1)

        # Isi Data
        col_res = (0, 255, 0) if "COMPLETE" in reason else (255, 50, 50)
        self.draw_text(f"Episode     : #{episode}", bx + 30, by + 80, 20)
        self.draw_text(f"Status      : {reason}", bx + 30, by + 110, 20, col_res)
        self.draw_text(f"Kebersihan  : {dirt_pct}%", bx + 30, by + 140, 20)
        self.draw_text(f"Total Steps : {steps}", bx + 30, by + 170, 20, (255, 255, 100))
        
        # Footer
        pygame.draw.line(self.screen, (100,100,100), (bx+20, by+240), (bx+w-20, by+240), 1)
        self.draw_text("Klik [RESUME] atau [WATCH] di panel", bx + w//2, by + 270, 16, (200, 200, 200), center=True)


# ==========================================
# 5. FUNGSI UTILITAS
# ==========================================
def plot_learning_curve(rewards):
    if not rewards:
        print("Belum ada data history reward.")
        return
    
    plt.figure(figsize=(10, 5))
    plt.plot(rewards, label='Total Reward', color='#0078FF', alpha=0.6)
    
    if len(rewards) >= 50:
        win_size = 50
        moving_avg = np.convolve(rewards, np.ones(win_size)/win_size, mode='valid')
        plt.plot(range(win_size-1, len(rewards)), moving_avg, label='Avg (50 eps)', color='red', linewidth=2)
    
    plt.title("Grafik Pembelajaran Robot")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    plt.show()


# ==========================================
# 6. MAIN LOOP
# ==========================================
def main():
    pygame.init()
    screen = pygame.display.set_mode((Config.WINDOW_W, Config.WINDOW_H))
    pygame.display.set_caption("Smart Robot Cleaner - Refactored")
    clock = pygame.time.Clock()
    
    # Inisialisasi Objek
    env = SweepEnv()
    agent = QAgent()
    viz = Visualizer(screen)
    
    # State Variables
    mode = "IDLE"  # Options: IDLE, FAST_TRAIN, WATCH_LEARNING, SHOW_STATS
    episode = 0
    history_rewards = []
    current_ep_reward = 0
    
    # Cache untuk pop-up stats
    last_stats = {"reason": "", "dirt": 0, "steps": 0}
    
    running = True
    while running:
        screen.fill(Config.COLOR_BG)
        
        # --- EVENT HANDLING ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            
            if event.type == pygame.MOUSEBUTTONDOWN:
                mx, my = pygame.mouse.get_pos()
                
                # Cek Klik Tombol
                if viz.btn_train.collidepoint(mx, my):
                    mode = "FAST_TRAIN"
                    # Reset env jika start dari kondisi mati/selesai
                    if np.sum(env.grid) == 0 or env.battery <= 0:
                        env.reset()
                        current_ep_reward = 0
                    print(f"Resuming training at Episode {episode}...")

                elif viz.btn_sim.collidepoint(mx, my):
                    mode = "WATCH_LEARNING"
                    env.reset()
                    current_ep_reward = 0
                    print("Watching 1 episode...")

                elif viz.btn_graph.collidepoint(mx, my):
                    plot_learning_curve(history_rewards)

        # --- LOGIC UPDATE ---
        if mode == "FAST_TRAIN":
            # Train 500 langkah per frame agar cepat
            for _ in range(500):
                state = env.get_state()
                action = agent.get_action(state)
                next_state, reward, done, _, reason = env.step(action)
                
                agent.update(state, action, reward, next_state, done)
                current_ep_reward += reward
                
                if done:
                    episode += 1
                    history_rewards.append(current_ep_reward)
                    current_ep_reward = 0
                    
                    if agent.epsilon > Config.EPS_END:
                        agent.epsilon *= Config.EPS_DECAY
                    
                    env.reset()
                    
                    if episode >= Config.N_EPISODES:
                        mode = "IDLE"
                        agent.save()
                        print("Target Episode Tercapai.")
                        break

        elif mode == "WATCH_LEARNING":
            state = env.get_state()
            action = agent.get_action(state)
            next_state, reward, done, dirt, reason = env.step(action)
            
            agent.update(state, action, reward, next_state, done)
            current_ep_reward += reward
            
            time.sleep(0.05) # Delay visual
            
            if done:
                episode += 1
                history_rewards.append(current_ep_reward)
                current_ep_reward = 0
                if agent.epsilon > Config.EPS_END:
                    agent.epsilon *= Config.EPS_DECAY
                
                # Simpan stats untuk ditampilkan
                last_stats = {
                    "reason": reason,
                    "dirt": 100 - dirt,
                    "steps": env.steps
                }
                mode = "SHOW_STATS"

        # --- DRAWING ---
        viz.render_env(env)
        viz.render_panel(env, agent, mode, episode)
        
        if mode == "SHOW_STATS":
            viz.show_popup(episode, last_stats["reason"], last_stats["dirt"], last_stats["steps"])

        pygame.display.flip()
        
        # Cap FPS saat mode visual
        if mode != "FAST_TRAIN":
            clock.tick(60)

    pygame.quit()

if __name__ == "__main__":
    main()
