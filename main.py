import pygame
import random
import sys
import cv2
import numpy as np
import json
import os

# --- Initialization ---
pygame.init()

# --- Screen Dimensions ---
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 800
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("AI Traffic Management Simulation")

# --- Colors ---
GRAY = (100, 100, 100)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
YELLOW = (255, 255, 0)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
BLUE = (0, 0, 255)

# --- Traffic Light Properties ---
TRAFFIC_LIGHT_STATES = {
    0: {'ns': 'green', 'ew': 'red'}, # State 0: NS Green, EW Red
    1: {'ns': 'red', 'ew': 'green'}   # State 1: NS Red, EW Green
}
current_traffic_light_state_index = 0
YELLOW_LIGHT_DURATION = 50

# --- Road and Lane Properties ---
ROAD_WIDTH = 100
LANE_WIDTH = ROAD_WIDTH // 2

# --- Vehicle Properties ---
VEHICLE_WIDTH = 30
VEHICLE_HEIGHT = 50
VEHICLE_SPEED = 3

class Vehicle(pygame.sprite.Sprite):
    """Represents a single vehicle in the simulation."""
    def __init__(self, x, y, direction, color):
        super().__init__()
        self.original_image = pygame.Surface([VEHICLE_WIDTH, VEHICLE_HEIGHT])
        self.original_image.fill(color)
        self.image = self.original_image
        self.rect = self.image.get_rect(topleft=(x, y))
        self.direction = direction
        self.speed = VEHICLE_SPEED
        self.stopped = False
        self.wait_time = 0

        if self.direction == 'south':
            self.image = pygame.transform.rotate(self.original_image, 180)
        elif self.direction == 'east':
            self.image = pygame.transform.rotate(self.original_image, -90)
            self.rect.width, self.rect.height = self.rect.height, self.rect.width
        elif self.direction == 'west':
            self.image = pygame.transform.rotate(self.original_image, 90)
            self.rect.width, self.rect.height = self.rect.height, self.rect.width

    def update(self, traffic_light_state, vehicles):
        """Updates the vehicle's position and state."""
        is_at_stop_line = self.is_near_intersection()

        if is_at_stop_line:
            self.handle_intersection(traffic_light_state)
        
        self.handle_collisions(vehicles)

        if self.stopped:
            self.wait_time += 1
        else:
            self.wait_time = 0 # Reset wait time if moving
            if self.direction == 'north': self.rect.y -= self.speed
            elif self.direction == 'south': self.rect.y += self.speed
            elif self.direction == 'east': self.rect.x += self.speed
            elif self.direction == 'west': self.rect.x -= self.speed
    
    def is_near_intersection(self):
        """Checks if the vehicle is approaching the intersection stop line."""
        stop_line_margin = 5
        if self.direction == 'north' and (SCREEN_HEIGHT / 2 + ROAD_WIDTH - stop_line_margin) <= self.rect.bottom <= (SCREEN_HEIGHT / 2 + ROAD_WIDTH): return True
        if self.direction == 'south' and (SCREEN_HEIGHT / 2 - ROAD_WIDTH) <= self.rect.top <= (SCREEN_HEIGHT / 2 - ROAD_WIDTH + stop_line_margin): return True
        if self.direction == 'east' and (SCREEN_WIDTH / 2 - ROAD_WIDTH) <= self.rect.left <= (SCREEN_WIDTH / 2 - ROAD_WIDTH + stop_line_margin): return True
        if self.direction == 'west' and (SCREEN_WIDTH / 2 + ROAD_WIDTH - stop_line_margin) <= self.rect.right <= (SCREEN_WIDTH / 2 + ROAD_WIDTH): return True
        return False

    def handle_intersection(self, traffic_light_state):
        """Decides whether the vehicle should stop based on the traffic light."""
        if (self.direction in ['north', 'south'] and traffic_light_state['ns'] != 'green') or \
           (self.direction in ['east', 'west'] and traffic_light_state['ew'] != 'green'):
            self.stopped = True
        else:
            self.stopped = False

    def handle_collisions(self, vehicles):
        """Prevents vehicles from overlapping."""
        other_vehicles = vehicles.copy()
        other_vehicles.remove(self)
        future_rect = self.rect.copy()
        
        # Look ahead based on direction
        if self.direction == 'north': future_rect.y -= self.speed
        elif self.direction == 'south': future_rect.y += self.speed
        elif self.direction == 'east': future_rect.x += self.speed
        elif self.direction == 'west': future_rect.x -= self.speed

        for other in other_vehicles:
            if self.direction == other.direction and future_rect.colliderect(other.rect):
                self.stopped = True
                return
        
        if not self.is_near_intersection():
            self.stopped = False

class TrafficAI:
    """AI agent using Q-learning to manage traffic lights."""
    def __init__(self, learning_rate=0.1, discount_factor=0.9, exploration_rate=0.1, q_table_file='q_table.json'):
        self.q_table = {}
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = exploration_rate
        self.q_table_file = q_table_file
        self.load_q_table()

    def get_state_representation(self, traffic_counts):
        """Converts traffic counts into a simplified, discrete state representation."""
        # Example: (num_ns, num_ew) where each value is low(0), medium(1), or high(2)
        ns_traffic = traffic_counts.get('north', 0) + traffic_counts.get('south', 0)
        ew_traffic = traffic_counts.get('east', 0) + traffic_counts.get('west', 0)
        
        def discretize(value):
            if value < 3: return 0 # Low
            if value < 6: return 1 # Medium
            return 2 # High
            
        return (discretize(ns_traffic), discretize(ew_traffic))

    def choose_action(self, state):
        """Chooses an action using an epsilon-greedy policy."""
        # Actions: 0 = keep current light state, 1 = switch light state
        if random.random() < self.epsilon:
            return random.choice([0, 1])  # Explore
        else:
            # Exploit: choose the best known action
            q_values = self.q_table.get(str(state), {})
            return max(q_values, key=q_values.get, default=random.choice([0, 1]))

    def update_q_table(self, state, action, reward, next_state):
        """Updates the Q-value for a state-action pair using the Bellman equation."""
        state_str = str(state)
        next_state_str = str(next_state)
        
        old_value = self.q_table.get(state_str, {}).get(str(action), 0)
        
        next_max = max(self.q_table.get(next_state_str, {}).values(), default=0)
        
        new_value = old_value + self.lr * (reward + self.gamma * next_max - old_value)
        
        if state_str not in self.q_table:
            self.q_table[state_str] = {}
        self.q_table[state_str][str(action)] = new_value

    def save_q_table(self):
        """Saves the Q-table to a file."""
        with open(self.q_table_file, 'w') as f:
            json.dump(self.q_table, f)

    def load_q_table(self):
        """Loads the Q-table from a file if it exists."""
        if os.path.exists(self.q_table_file):
            with open(self.q_table_file, 'r') as f:
                self.q_table = json.load(f)

def draw_environment():
    """Draws the road layout and lane lines."""
    screen.fill(BLACK)
    pygame.draw.rect(screen, GRAY, (SCREEN_WIDTH/2 - ROAD_WIDTH, 0, ROAD_WIDTH * 2, SCREEN_HEIGHT))
    pygame.draw.rect(screen, GRAY, (0, SCREEN_HEIGHT/2 - ROAD_WIDTH, SCREEN_WIDTH, ROAD_WIDTH * 2))
    for y in range(0, SCREEN_HEIGHT, 20):
        if not (SCREEN_HEIGHT/2 - ROAD_WIDTH < y < SCREEN_HEIGHT/2 + ROAD_WIDTH):
            pygame.draw.line(screen, WHITE, (SCREEN_WIDTH/2, y), (SCREEN_WIDTH/2, y + 10), 2)
    for x in range(0, SCREEN_WIDTH, 20):
        if not (SCREEN_WIDTH/2 - ROAD_WIDTH < x < SCREEN_WIDTH/2 + ROAD_WIDTH):
            pygame.draw.line(screen, WHITE, (x, SCREEN_HEIGHT/2), (x + 10, SCREEN_HEIGHT/2), 2)

def draw_traffic_lights(state):
    """Draws the traffic lights."""
    ns_color = GREEN if state['ns'] == 'green' else RED if state['ns'] == 'red' else YELLOW
    ew_color = GREEN if state['ew'] == 'green' else RED if state['ew'] == 'red' else YELLOW
    pygame.draw.circle(screen, ns_color, (SCREEN_WIDTH/2 - ROAD_WIDTH - 20, SCREEN_HEIGHT/2 - ROAD_WIDTH - 20), 15)
    pygame.draw.circle(screen, ns_color, (SCREEN_WIDTH/2 + ROAD_WIDTH + 20, SCREEN_HEIGHT/2 + ROAD_WIDTH + 20), 15)
    pygame.draw.circle(screen, ew_color, (SCREEN_WIDTH/2 + ROAD_WIDTH + 20, SCREEN_HEIGHT/2 - ROAD_WIDTH - 20), 15)
    pygame.draw.circle(screen, ew_color, (SCREEN_WIDTH/2 - ROAD_WIDTH - 20, SCREEN_HEIGHT/2 + ROAD_WIDTH + 20), 15)

def detect_traffic(frame, rois):
    """Detects vehicles in ROIs using OpenCV."""
    traffic_counts = {}
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    for direction, (x, y, w, h) in rois.items():
        roi = gray_frame[y:y+h, x:x+w]
        _, dark_mask = cv2.threshold(roi, 95, 255, cv2.THRESH_BINARY_INV)
        _, light_mask = cv2.threshold(roi, 105, 255, cv2.THRESH_BINARY)
        binary_mask = cv2.bitwise_or(dark_mask, light_mask)
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        traffic_counts[direction] = len(contours)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)
        cv2.putText(frame, f"{direction.capitalize()}: {len(contours)}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    return traffic_counts, frame

def calculate_reward(vehicles):
    """Calculates the reward based on total wait time. Lower wait time is better."""
    total_wait_time = sum(v.wait_time for v in vehicles)
    # Negative reward: we want to minimize this value.
    return -total_wait_time

def main():
    """Main simulation loop."""
    global current_traffic_light_state_index

    clock = pygame.time.Clock()
    all_sprites = pygame.sprite.Group()
    vehicles = pygame.sprite.Group()
    ai_agent = TrafficAI()

    vehicle_generation_timer = 0
    VEHICLE_GENERATION_INTERVAL = 100
    
    ai_decision_timer = 0
    AI_DECISION_INTERVAL = 180 # AI makes a decision every 3 seconds (60fps * 3)

    roi_length = 200
    rois = {
        'north': (SCREEN_WIDTH//2 - LANE_WIDTH, SCREEN_HEIGHT//2 + ROAD_WIDTH, LANE_WIDTH, roi_length),
        'south': (SCREEN_WIDTH//2, SCREEN_HEIGHT//2 - ROAD_WIDTH - roi_length, LANE_WIDTH, roi_length),
        'east': (SCREEN_WIDTH//2 - ROAD_WIDTH - roi_length, SCREEN_HEIGHT//2, roi_length, LANE_WIDTH),
        'west': (SCREEN_WIDTH//2 + ROAD_WIDTH, SCREEN_HEIGHT//2 - LANE_WIDTH, roi_length, LANE_WIDTH)
    }

    running = True
    is_yellow_light_phase = False
    yellow_light_timer = 0
    
    last_state = None
    last_action = None
    last_reward = 0

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                ai_agent.save_q_table()
                running = False

        # --- Vehicle Generation ---
        vehicle_generation_timer += 1
        if vehicle_generation_timer > VEHICLE_GENERATION_INTERVAL:
            direction = random.choice(['north', 'south', 'east', 'west'])
            color = random.choice([BLUE, (200,0,200), (0,200,200)])
            if direction == 'north': x, y = SCREEN_WIDTH/2 - LANE_WIDTH + (LANE_WIDTH - VEHICLE_WIDTH)/2, SCREEN_HEIGHT
            elif direction == 'south': x, y = SCREEN_WIDTH/2 + (LANE_WIDTH - VEHICLE_WIDTH)/2, -VEHICLE_HEIGHT
            elif direction == 'east': x, y = -VEHICLE_HEIGHT, SCREEN_HEIGHT/2 - LANE_WIDTH + (LANE_WIDTH - VEHICLE_WIDTH)/2
            elif direction == 'west': x, y = SCREEN_WIDTH, SCREEN_HEIGHT/2 + (LANE_WIDTH - VEHICLE_WIDTH)/2
            all_sprites.add(Vehicle(x, y, direction, color))
            vehicles.add(all_sprites.sprites()[-1])
            vehicle_generation_timer = 0

        # --- AI and Traffic Light Logic ---
        frame = pygame.surfarray.array3d(screen)
        frame = frame.transpose([1, 0, 2])
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        traffic_counts, _ = detect_traffic(frame_bgr.copy(), rois)
        
        current_state = ai_agent.get_state_representation(traffic_counts)
        
        ai_decision_timer += 1
        if ai_decision_timer > AI_DECISION_INTERVAL:
            ai_decision_timer = 0
            # Update Q-table with the result of the last action
            if last_state is not None:
                ai_agent.update_q_table(last_state, last_action, last_reward, current_state)

            # AI chooses a new action
            action = ai_agent.choose_action(current_state)
            if action == 1 and not is_yellow_light_phase: # Action to switch lights
                is_yellow_light_phase = True
                yellow_light_timer = 0
            
            last_state = current_state
            last_action = action

        if is_yellow_light_phase:
            yellow_light_timer += 1
            if TRAFFIC_LIGHT_STATES[current_traffic_light_state_index]['ns'] == 'green':
                active_state = {'ns': 'yellow', 'ew': 'red'}
            else:
                active_state = {'ns': 'red', 'ew': 'yellow'}
            if yellow_light_timer > YELLOW_LIGHT_DURATION:
                is_yellow_light_phase = False
                current_traffic_light_state_index = 1 - current_traffic_light_state_index
        else:
            active_state = TRAFFIC_LIGHT_STATES[current_traffic_light_state_index]

        # --- Update Sprites ---
        for vehicle in vehicles:
            vehicle.update(active_state, vehicles)
        
        last_reward = calculate_reward(vehicles) # Calculate reward for the current state

        # Remove off-screen vehicles
        for vehicle in list(vehicles):
            if not screen.get_rect().colliderect(vehicle.rect):
                vehicle.kill()

        # --- Drawing ---
        draw_environment()
        draw_traffic_lights(active_state)
        all_sprites.draw(screen)
        
        # --- OpenCV Display ---
        _, processed_frame = detect_traffic(frame_bgr, rois)
        cv2.imshow('AI Traffic Analysis', processed_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            ai_agent.save_q_table()
            running = False

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()
    cv2.destroyAllWindows()
    sys.exit()

if __name__ == '__main__':
    main()
