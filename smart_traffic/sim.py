# LAG
# NO. OF VEHICLES IN SIGNAL CLASS
# stops not used
# DISTRIBUTION
# BUS TOUCHING ON TURNS
# Distribution using python class

# *** IMAGE XY COOD IS TOP LEFT
import random
import math
import time
import threading
# from vehicle_detection import detection
import pygame
import sys
import os

# options={
#    'model':'./cfg/yolo.cfg',     #specifying the path of model
#    'load':'./bin/yolov2.weights',   #weights
#    'threshold':0.3     #minimum confidence factor to create a box, greater than 0.3 good
# }

# tfnet=TFNet(options)    #READ ABOUT TFNET

# Default values of signal times
defaultRed = 150
defaultYellow = 5
defaultGreen = 20
defaultMinimum = 10
defaultMaximum = 60

signals = []
noOfSignals = 4
simTime = 300       # change this to change time of simulation
timeElapsed = 0

currentGreen = 0   # Indicates which signal is green
nextGreen = (currentGreen+1)%noOfSignals
currentYellow = 0   # Indicates whether yellow signal is on or off 

# Average times for vehicles to pass the intersection
carTime = 2
bikeTime = 1
rickshawTime = 2.25 
busTime = 2.5
truckTime = 2.5

# Count of cars at a traffic signal
noOfCars = 0
noOfBikes = 0
noOfBuses =0
noOfTrucks = 0
noOfRickshaws = 0
noOfLanes = 2

# Red signal time at which cars will be detected at a signal
detectionTime = 5

speeds = {'car':2.25, 'bus':1.8, 'truck':1.8, 'rickshaw':2, 'bike':2.5, 'ambulance':2.4, 'firetruck':2.1, 'police':2.3}  # tuned to medium emergency speeds

# Coordinates of start
x = {'right':[0,0,0], 'down':[755,727,697], 'left':[1400,1400,1400], 'up':[602,627,657]}    
y = {'right':[348,370,398], 'down':[0,0,0], 'left':[498,466,436], 'up':[800,800,800]}

vehicles = {'right': {0:[], 1:[], 2:[], 'crossed':0}, 'down': {0:[], 1:[], 2:[], 'crossed':0}, 'left': {0:[], 1:[], 2:[], 'crossed':0}, 'up': {0:[], 1:[], 2:[], 'crossed':0}}
vehicleTypes = {0:'car', 1:'bus', 2:'truck', 3:'rickshaw', 4:'bike', 5:'ambulance', 6:'firetruck', 7:'police'}
emergencyVehicleTypes = {'ambulance','firetruck','police'}
emergencyCounts = {'right':0,'down':0,'left':0,'up':0}

# Emergency handling state
emergencyActive = 0
emergencyTarget = None  # direction number (0-3)
emergencyStage = 'idle'  # idle | prepare | serving
emergencyPopupStart = 0
emergencyHoldUntil = 0
emergencyPopupDuration = 4.0
emergencyPopupText = "Emergency Vehicle Detected!"

# Progress watchdog to avoid stalls
crossedProgress = {
    'right': {'last_count': 0, 'last_time': time.time()},
    'down':  {'last_count': 0, 'last_time': time.time()},
    'left':  {'last_count': 0, 'last_time': time.time()},
    'up':    {'last_count': 0, 'last_time': time.time()},
}
 
# Freeze window to stop vehicles at stop lines (during all-red)
freezeUntil = 0.0

# Probability to spawn an emergency vehicle per spawn event (0.0 - 1.0)
emergencySpawnProb = 0.02
# Minimum time gap between emergency spawns (seconds) to avoid clusters
emergencyMinGapSeconds = 8.0
lastEmergencySpawnAt = 0.3
directionNumbers = {0:'right', 1:'down', 2:'left', 3:'up'}

# ===================== Multi-Agent RL (one agent per signal) =====================
# We use a simple independent Q-learning per direction. Each agent evaluates the
# value of giving green now, based on a compact state: (queue_bin, max_wait_bin).
# Reward after serving: vehicles_passed - wait_penalty * max_wait_all_directions.
class RLAgent:
    def __init__(self, direction_index, alpha=0.3, gamma=0.85, epsilon=0.1):
        self.direction_index = direction_index
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_values = {}  # state_key -> value of action "serve_now"
        self.last_state = None
        self.active = False

    def state_key(self):
        qlen = count_queue_in_direction(self.direction_index)
        maxw = get_max_wait_time(self.direction_index)
        # discretize
        if qlen == 0:
            qbin = 0
        elif qlen <= 3:
            qbin = 1
        elif qlen <= 7:
            qbin = 2
        else:
            qbin = 3
        if maxw < 5:
            wbin = 0
        elif maxw < 10:
            wbin = 1
        elif maxw < 20:
            wbin = 2
        else:
            wbin = 3
        return (qbin, wbin)

    def value(self, state_key=None):
        if state_key is None:
            state_key = self.state_key()
        return self.q_values.get(state_key, 0.0)

    def start_serving(self):
        self.last_state = self.state_key()
        self.active = True

    def finish_serving(self, reward, next_state_key=None):
        if not self.active or self.last_state is None:
            return
        if next_state_key is None:
            next_state_key = self.state_key()
        old = self.q_values.get(self.last_state, 0.0)
        target = reward + self.gamma * self.value(next_state_key)
        self.q_values[self.last_state] = old + self.alpha * (target - old)
        self.active = False
        self.last_state = None

# Global agent storage and episode context
agents = {}
agentContext = {  # per-direction context during a green episode
    0: {'crossed_start': 0, 'start_time': 0.0},
    1: {'crossed_start': 0, 'start_time': 0.0},
    2: {'crossed_start': 0, 'start_time': 0.0},
    3: {'crossed_start': 0, 'start_time': 0.0},
}
wait_penalty = 0.25  # weight to penalize maximum waiting time across directions
long_wait_threshold = 12.0  # seconds; override rule to open lanes with long waits (even if few vehicles)

# ===================== Downstream ETA/Info Display =====================
nextJunctionDistanceMeters = 540.0
baseTravelSpeedMps = 12.0           # ~43.2 km/h base speed
avgEmergencySpeedMps = 18.0        # ~64.8 km/h
emergencyETAByDir = {'right': None, 'down': None, 'left': 0.0, 'up': None}
emergencyMessageUntil = {'right': 0.0, 'down': 0.0, 'left': 0.0, 'up': 0.0}

# Dynamic ETA calculation based on vehicle density
def calculate_dynamic_eta(direction_name, vehicles_passed_this_green):
    """
    Calculate ETA based on vehicle density:
    - 0-5 vehicles: very low density, fastest speed
    - 6-10 vehicles: low density, faster speed
    - 11-20 vehicles: medium density, normal speed
    - 21+ vehicles: high density, slower speed
    """
    base_distance = nextJunctionDistanceMeters
    
    # Density-based speed adjustment with multiple levels
    if vehicles_passed_this_green <= 5:
        # Very low density: vehicles can move fastest
        speed_multiplier = 1.3  # 30% faster
        density_factor = 0.7    # ETA reduced by 30%
    elif vehicles_passed_this_green <= 10:
        # Low density: vehicles can move faster
        speed_multiplier = 1.1  # 10% faster
        density_factor = 0.9    # ETA reduced by 10%
    elif vehicles_passed_this_green <= 20:
        # Medium density: normal speed
        speed_multiplier = 1.0  # normal speed
        density_factor = 1.0    # normal ETA
    else:
        # High density: vehicles move slower due to congestion
        speed_multiplier = 0.6  # 40% slower
        density_factor = 1.6    # ETA increased by 60%
    
    # Calculate adjusted speed
    adjusted_speed = baseTravelSpeedMps * speed_multiplier
    
    # Calculate ETA with density factor
    eta_seconds = (base_distance / max(1e-6, adjusted_speed)) * density_factor
    
    return eta_seconds

def format_eta(seconds):
    seconds = max(0, int(round(seconds)))
    m = seconds // 60
    s = seconds % 60
    if m > 0:
        return f"{m}:{s:02d} min"
    return f"{s} s"

# Coordinates of signal image, timer, and vehicle count
signalCoods = [(530,230),(810,230),(810,570),(530,570)]
signalTimerCoods = [(530,210),(810,210),(810,550),(530,550)]
vehicleCountCoods = [(480,210),(880,210),(880,550),(480,550)]
vehicleCountTexts = ["0", "0", "0", "0"]

# Coordinates of stop lines
stopLines = {'right': 590, 'down': 330, 'left': 800, 'up': 535}
defaultStop = {'right': 580, 'down': 320, 'left': 810, 'up': 545}
stops = {'right': [580,580,580], 'down': [320,320,320], 'left': [810,810,810], 'up': [545,545,545]}

mid = {'right': {'x':705, 'y':445}, 'down': {'x':695, 'y':450}, 'left': {'x':695, 'y':425}, 'up': {'x':695, 'y':400}}
rotationAngle = 3

# Gap between vehicles
gap = 15    # stopping gap
gap2 = 15   # moving gap

pygame.init()
simulation = pygame.sprite.Group()

class TrafficSignal:
    def __init__(self, red, yellow, green, minimum, maximum):
        self.red = red
        self.yellow = yellow
        self.green = green
        self.minimum = minimum
        self.maximum = maximum
        self.signalText = "30"
        self.totalGreenTime = 0

def print_results_and_exit():
    totalVehicles = 0
    print('Lane-wise Vehicle Counts')
    for i in range(noOfSignals):
        print('Lane',i+1,':',vehicles[directionNumbers[i]]['crossed'])
        totalVehicles += vehicles[directionNumbers[i]]['crossed']
    print('Total vehicles passed: ',totalVehicles)
    print('Total time passed: ',timeElapsed)
    if timeElapsed>0:
        print('No. of vehicles passed per unit time: ',(float(totalVehicles)/float(timeElapsed)))
    print('Emergency Vehicles Passed (by lane):')
    for d in ['right','down','left','up']:
        print(d.capitalize()+':', emergencyCounts[d])
    pygame.quit()
    sys.exit()
        
class Vehicle(pygame.sprite.Sprite):
    def __init__(self, lane, vehicleClass, direction_number, direction, will_turn):
        pygame.sprite.Sprite.__init__(self)
        self.lane = lane
        self.vehicleClass = vehicleClass
        self.speed = speeds[vehicleClass]
        self.direction_number = direction_number
        self.direction = direction
        self.x = x[direction][lane]
        self.y = y[direction][lane]
        self.crossed = 0
        self.willTurn = will_turn
        self.turned = 0
        self.rotateAngle = 0
        self.spawn_time = time.time()
        vehicles[direction][lane].append(self)
        # self.stop = stops[direction][lane]
        self.index = len(vehicles[direction][lane]) - 1
        path = "images/" + direction + "/" + vehicleClass + ".png"
        self.originalImage = pygame.image.load(path)
        self.currentImage = pygame.image.load(path)

    
        if(direction=='right'):
            if(len(vehicles[direction][lane])>1 and vehicles[direction][lane][self.index-1].crossed==0):    # if more than 1 vehicle in the lane of vehicle before it has crossed stop line
                self.stop = vehicles[direction][lane][self.index-1].stop - vehicles[direction][lane][self.index-1].currentImage.get_rect().width - gap         # setting stop coordinate as: stop coordinate of next vehicle - width of next vehicle - gap
            else:
                self.stop = defaultStop[direction]
            # Set new starting and stopping coordinate
            temp = self.currentImage.get_rect().width + gap    
            x[direction][lane] -= temp
            stops[direction][lane] -= temp
        elif(direction=='left'):
            if(len(vehicles[direction][lane])>1 and vehicles[direction][lane][self.index-1].crossed==0):
                self.stop = vehicles[direction][lane][self.index-1].stop + vehicles[direction][lane][self.index-1].currentImage.get_rect().width + gap
            else:
                self.stop = defaultStop[direction]
            temp = self.currentImage.get_rect().width + gap
            x[direction][lane] += temp
            stops[direction][lane] += temp
        elif(direction=='down'):
            if(len(vehicles[direction][lane])>1 and vehicles[direction][lane][self.index-1].crossed==0):
                self.stop = vehicles[direction][lane][self.index-1].stop - vehicles[direction][lane][self.index-1].currentImage.get_rect().height - gap
            else:
                self.stop = defaultStop[direction]
            temp = self.currentImage.get_rect().height + gap
            y[direction][lane] -= temp
            stops[direction][lane] -= temp
        elif(direction=='up'):
            if(len(vehicles[direction][lane])>1 and vehicles[direction][lane][self.index-1].crossed==0):
                self.stop = vehicles[direction][lane][self.index-1].stop + vehicles[direction][lane][self.index-1].currentImage.get_rect().height + gap
            else:
                self.stop = defaultStop[direction]
            temp = self.currentImage.get_rect().height + gap
            y[direction][lane] += temp
            stops[direction][lane] += temp
        simulation.add(self)

    def render(self, screen):
        screen.blit(self.currentImage, (self.x, self.y))

    def move(self):
        # During freeze window, keep vehicles before stop line stationary
        if time.time() < freezeUntil:
            # allow vehicles that already crossed to continue; others hold
            if self.crossed == 0:
                return
        if(self.direction=='right'):
            if(self.crossed==0 and self.x+self.currentImage.get_rect().width>stopLines[self.direction]):   # if the image has crossed stop line now
                self.crossed = 1
                vehicles[self.direction]['crossed'] += 1
                if self.vehicleClass in emergencyVehicleTypes:
                    emergencyCounts[self.direction] += 1
                    # emergency passed: compute ETA and show special message for a short time
                    eta = nextJunctionDistanceMeters / max(1e-6, avgEmergencySpeedMps)
                    emergencyETAByDir[self.direction] = eta
                    emergencyMessageUntil[self.direction] = time.time() + 5.0
            if(self.willTurn==1):
                if(self.crossed==0 or self.x+self.currentImage.get_rect().width<mid[self.direction]['x']):
                    if((self.x+self.currentImage.get_rect().width<=self.stop or (currentGreen==0 and currentYellow==0) or self.crossed==1) and (self.index==0 or self.x+self.currentImage.get_rect().width<(vehicles[self.direction][self.lane][self.index-1].x - gap2) or vehicles[self.direction][self.lane][self.index-1].turned==1)):                
                        self.x += self.speed
                else:   
                    if(self.turned==0):
                        self.rotateAngle += rotationAngle
                        self.currentImage = pygame.transform.rotate(self.originalImage, -self.rotateAngle)
                        self.x += 2
                        self.y += 1.8
                        if(self.rotateAngle==90):
                            self.turned = 1
                            # path = "images/" + directionNumbers[((self.direction_number+1)%noOfSignals)] + "/" + self.vehicleClass + ".png"
                            # self.x = mid[self.direction]['x']
                            # self.y = mid[self.direction]['y']
                            # self.image = pygame.image.load(path)
                    else:
                        if(self.index==0 or self.y+self.currentImage.get_rect().height<(vehicles[self.direction][self.lane][self.index-1].y - gap2) or self.x+self.currentImage.get_rect().width<(vehicles[self.direction][self.lane][self.index-1].x - gap2)):
                            self.y += self.speed
            else: 
                if((self.x+self.currentImage.get_rect().width<=self.stop or self.crossed == 1 or (currentGreen==0 and currentYellow==0)) and (self.index==0 or self.x+self.currentImage.get_rect().width<(vehicles[self.direction][self.lane][self.index-1].x - gap2) or (vehicles[self.direction][self.lane][self.index-1].turned==1))):                
                # (if the image has not reached its stop coordinate or has crossed stop line or has green signal) and (it is either the first vehicle in that lane or it is has enough gap to the next vehicle in that lane)
                    self.x += self.speed  # move the vehicle



        elif(self.direction=='down'):
            if(self.crossed==0 and self.y+self.currentImage.get_rect().height>stopLines[self.direction]):
                self.crossed = 1
                vehicles[self.direction]['crossed'] += 1
                if self.vehicleClass in emergencyVehicleTypes:
                    emergencyCounts[self.direction] += 1
                    eta = nextJunctionDistanceMeters / max(1e-6, avgEmergencySpeedMps)
                    emergencyETAByDir[self.direction] = eta
                    emergencyMessageUntil[self.direction] = time.time() + 5.0
            if(self.willTurn==1):
                if(self.crossed==0 or self.y+self.currentImage.get_rect().height<mid[self.direction]['y']):
                    if((self.y+self.currentImage.get_rect().height<=self.stop or (currentGreen==1 and currentYellow==0) or self.crossed==1) and (self.index==0 or self.y+self.currentImage.get_rect().height<(vehicles[self.direction][self.lane][self.index-1].y - gap2) or vehicles[self.direction][self.lane][self.index-1].turned==1)):                
                        self.y += self.speed
                else:   
                    if(self.turned==0):
                        self.rotateAngle += rotationAngle
                        self.currentImage = pygame.transform.rotate(self.originalImage, -self.rotateAngle)
                        self.x -= 2.5
                        self.y += 2
                        if(self.rotateAngle==90):
                            self.turned = 1
                    else:
                        if(self.index==0 or self.x>(vehicles[self.direction][self.lane][self.index-1].x + vehicles[self.direction][self.lane][self.index-1].currentImage.get_rect().width + gap2) or self.y<(vehicles[self.direction][self.lane][self.index-1].y - gap2)):
                            self.x -= self.speed
            else: 
                if((self.y+self.currentImage.get_rect().height<=self.stop or self.crossed == 1 or (currentGreen==1 and currentYellow==0)) and (self.index==0 or self.y+self.currentImage.get_rect().height<(vehicles[self.direction][self.lane][self.index-1].y - gap2) or (vehicles[self.direction][self.lane][self.index-1].turned==1))):                
                    self.y += self.speed
            
        elif(self.direction=='left'):
            if(self.crossed==0 and self.x<stopLines[self.direction]):
                self.crossed = 1
                vehicles[self.direction]['crossed'] += 1
                if self.vehicleClass in emergencyVehicleTypes:
                    emergencyCounts[self.direction] += 1
                    eta = nextJunctionDistanceMeters / max(1e-6, avgEmergencySpeedMps)
                    emergencyETAByDir[self.direction] = eta
                    emergencyMessageUntil[self.direction] = time.time() + 5.0
            if(self.willTurn==1):
                if(self.crossed==0 or self.x>mid[self.direction]['x']):
                    if((self.x>=self.stop or (currentGreen==2 and currentYellow==0) or self.crossed==1) and (self.index==0 or self.x>(vehicles[self.direction][self.lane][self.index-1].x + vehicles[self.direction][self.lane][self.index-1].currentImage.get_rect().width + gap2) or vehicles[self.direction][self.lane][self.index-1].turned==1)):                
                        self.x -= self.speed
                else: 
                    if(self.turned==0):
                        self.rotateAngle += rotationAngle
                        self.currentImage = pygame.transform.rotate(self.originalImage, -self.rotateAngle)
                        self.x -= 1.8
                        self.y -= 2.5
                        if(self.rotateAngle==90):
                            self.turned = 1
                            # path = "images/" + directionNumbers[((self.direction_number+1)%noOfSignals)] + "/" + self.vehicleClass + ".png"
                            # self.x = mid[self.direction]['x']
                            # self.y = mid[self.direction]['y']
                            # self.currentImage = pygame.image.load(path)
                    else:
                        if(self.index==0 or self.y>(vehicles[self.direction][self.lane][self.index-1].y + vehicles[self.direction][self.lane][self.index-1].currentImage.get_rect().height +  gap2) or self.x>(vehicles[self.direction][self.lane][self.index-1].x + gap2)):
                            self.y -= self.speed
            else: 
                if((self.x>=self.stop or self.crossed == 1 or (currentGreen==2 and currentYellow==0)) and (self.index==0 or self.x>(vehicles[self.direction][self.lane][self.index-1].x + vehicles[self.direction][self.lane][self.index-1].currentImage.get_rect().width + gap2))):                
                # (if the image has not reached its stop coordinate or has crossed stop line or has green signal) and (it is either the first vehicle in that lane or it is has enough gap to the next vehicle in that lane)
                    self.x -= self.speed  # move the vehicle    
            # if((self.x>=self.stop or self.crossed == 1 or (currentGreen==2 and currentYellow==0)) and (self.index==0 or self.x>(vehicles[self.direction][self.lane][self.index-1].x + vehicles[self.direction][self.lane][self.index-1].currentImage.get_rect().width + gap2))):                
            #     self.x -= self.speed
        elif(self.direction=='up'):
            if(self.crossed==0 and self.y<stopLines[self.direction]):
                self.crossed = 1
                vehicles[self.direction]['crossed'] += 1
                if self.vehicleClass in emergencyVehicleTypes:
                    emergencyCounts[self.direction] += 1
                    eta = nextJunctionDistanceMeters / max(1e-6, avgEmergencySpeedMps)
                    emergencyETAByDir[self.direction] = eta
                    emergencyMessageUntil[self.direction] = time.time() + 5.0
            if(self.willTurn==1):
                if(self.crossed==0 or self.y>mid[self.direction]['y']):
                    if((self.y>=self.stop or (currentGreen==3 and currentYellow==0) or self.crossed == 1) and (self.index==0 or self.y>(vehicles[self.direction][self.lane][self.index-1].y + vehicles[self.direction][self.lane][self.index-1].currentImage.get_rect().height +  gap2) or vehicles[self.direction][self.lane][self.index-1].turned==1)):
                        self.y -= self.speed
                else:   
                    if(self.turned==0):
                        self.rotateAngle += rotationAngle
                        self.currentImage = pygame.transform.rotate(self.originalImage, -self.rotateAngle)
                        self.x += 1
                        self.y -= 1
                        if(self.rotateAngle==90):
                            self.turned = 1
                    else:
                        if(self.index==0 or self.x<(vehicles[self.direction][self.lane][self.index-1].x - vehicles[self.direction][self.lane][self.index-1].currentImage.get_rect().width - gap2) or self.y>(vehicles[self.direction][self.lane][self.index-1].y + gap2)):
                            self.x += self.speed
            else: 
                if((self.y>=self.stop or self.crossed == 1 or (currentGreen==3 and currentYellow==0)) and (self.index==0 or self.y>(vehicles[self.direction][self.lane][self.index-1].y + vehicles[self.direction][self.lane][self.index-1].currentImage.get_rect().height + gap2) or (vehicles[self.direction][self.lane][self.index-1].turned==1))):                
                    self.y -= self.speed

# Initialization of signals with default values
def initialize():
    ts1 = TrafficSignal(0, defaultYellow, defaultGreen, defaultMinimum, defaultMaximum)
    signals.append(ts1)
    ts2 = TrafficSignal(ts1.red+ts1.yellow+ts1.green, defaultYellow, defaultGreen, defaultMinimum, defaultMaximum)
    signals.append(ts2)
    ts3 = TrafficSignal(defaultRed, defaultYellow, defaultGreen, defaultMinimum, defaultMaximum)
    signals.append(ts3)
    ts4 = TrafficSignal(defaultRed, defaultYellow, defaultGreen, defaultMinimum, defaultMaximum)
    signals.append(ts4)
    repeat()

def count_queue_in_direction(dir_number):
    direction = directionNumbers[dir_number]
    count = 0
    for lane in [0,1,2]:
        for vehicle in vehicles[direction][lane]:
            if vehicle.crossed == 0:
                count += 1
    return count

def get_all_queue_lengths():
    return [count_queue_in_direction(i) for i in range(noOfSignals)]

def get_max_wait_time(dir_number):
    # Maximum time a waiting (not yet crossed) vehicle has been in the queue
    direction = directionNumbers[dir_number]
    now = time.time()
    max_wait = 0.0
    for lane in [0,1,2]:
        for vehicle in vehicles[direction][lane]:
            if vehicle.crossed == 0:
                max_wait = max(max_wait, now - getattr(vehicle, 'spawn_time', now))
    return max_wait

def select_next_green_index(current_index):
    # Multi-agent RL decision with rule: open lanes with long waits even if few vehicles.
    # Evaluate a score per lane combining agent value, queue, and waiting penalty.
    queues = get_all_queue_lengths()
    # If all zero, rotate
    if max(queues) == 0:
        return (current_index + 1) % noOfSignals

    scores = []
    # compute global max wait to penalize long waits overall
    max_wait_overall = 0.0
    for d in range(noOfSignals):
        max_wait_overall = max(max_wait_overall, get_max_wait_time(d))

    for d in range(noOfSignals):
        # initialize agent lazily
        if d not in agents:
            agents[d] = RLAgent(d)
        agent = agents[d]
        value = agent.value()
        q = queues[d]
        maxw = get_max_wait_time(d)
        # rule: strong bonus if lane has long wait
        long_wait_bonus = 5.0 if maxw >= long_wait_threshold else 0.0
        score = value + 0.5*q + long_wait_bonus - wait_penalty * max_wait_overall
        scores.append((score, d))

    # choose best, tie-break by cyclic order from current_index
    scores.sort(key=lambda t: t[0], reverse=True)
    best_dirs = [d for (_, d) in scores if queues[d] > 0]
    if not best_dirs:
        return (current_index + 1) % noOfSignals
    # fair tie-break: pick first in cyclic order among equal-top scores
    top_score = scores[0][0]
    candidates = [d for (s, d) in scores if abs(s - top_score) < 1e-6]
    for k in range(1, noOfSignals+1):
        idx = (current_index + k) % noOfSignals
        if idx in candidates and queues[idx] > 0:
            return idx
    return best_dirs[0]

def has_pending_emergency(dir_number):
    direction = directionNumbers[dir_number]
    for lane in [0,1,2]:
        for vehicle in vehicles[direction][lane]:
            if vehicle.crossed == 0 and vehicle.vehicleClass in emergencyVehicleTypes:
                return True
    return False

def distance_to_stop_for_vehicle(vehicle):
    # Estimate distance from vehicle front to its lane stop line (pixels)
    if vehicle.direction == 'right':
        return max(0, stopLines['right'] - (vehicle.x + vehicle.currentImage.get_rect().width))
    if vehicle.direction == 'down':
        return max(0, stopLines['down'] - (vehicle.y + vehicle.currentImage.get_rect().height))
    if vehicle.direction == 'left':
        return max(0, (vehicle.x) - stopLines['left'])
    if vehicle.direction == 'up':
        return max(0, (vehicle.y) - stopLines['up'])
    return 10**9

def nearest_emergency_direction():
    # Returns direction index (0-3) of the nearest waiting emergency vehicle, or None
    best_dir = None
    best_dist = None
    for dnum in range(noOfSignals):
        direction = directionNumbers[dnum]
        # scan all lanes and emergency vehicles not yet crossed
        for lane in [0,1,2]:
            for v in vehicles[direction][lane]:
                if v.crossed == 0 and v.vehicleClass in emergencyVehicleTypes:
                    d = distance_to_stop_for_vehicle(v)
                    if (best_dist is None) or (d < best_dist):
                        best_dist = d
                        best_dir = dnum
    return best_dir

def detect_emergency():
    # Prioritize the closest emergency vehicle across all approaches
    return nearest_emergency_direction()

# Set time according to formula
def setTime():
    global noOfCars, noOfBikes, noOfBuses, noOfTrucks, noOfRickshaws, noOfLanes
    global carTime, busTime, truckTime, rickshawTime, bikeTime
    os.system("say detecting vehicles, "+directionNumbers[(currentGreen+1)%noOfSignals])
#    detection_result=detection(currentGreen,tfnet)
#    greenTime = math.ceil(((noOfCars*carTime) + (noOfRickshaws*rickshawTime) + (noOfBuses*busTime) + (noOfBikes*bikeTime))/(noOfLanes+1))
#    if(greenTime<defaultMinimum):
#       greenTime = defaultMinimum
#    elif(greenTime>defaultMaximum):
#       greenTime = defaultMaximum
    # greenTime = len(vehicles[currentGreen][0])+len(vehicles[currentGreen][1])+len(vehicles[currentGreen][2])
    # noOfVehicles = len(vehicles[directionNumbers[nextGreen]][1])+len(vehicles[directionNumbers[nextGreen]][2])-vehicles[directionNumbers[nextGreen]]['crossed']
    # print("no. of vehicles = ",noOfVehicles)
    noOfCars, noOfBuses, noOfTrucks, noOfRickshaws, noOfBikes = 0,0,0,0,0
    for j in range(len(vehicles[directionNumbers[nextGreen]][0])):
        vehicle = vehicles[directionNumbers[nextGreen]][0][j]
        if(vehicle.crossed==0):
            vclass = vehicle.vehicleClass
            # print(vclass)
            noOfBikes += 1
    for i in range(1,3):
        for j in range(len(vehicles[directionNumbers[nextGreen]][i])):
            vehicle = vehicles[directionNumbers[nextGreen]][i][j]
            if(vehicle.crossed==0):
                vclass = vehicle.vehicleClass
                # print(vclass)
                if(vclass=='car'):
                    noOfCars += 1
                elif(vclass=='bus'):
                    noOfBuses += 1
                elif(vclass=='truck'):
                    noOfTrucks += 1
                elif(vclass=='rickshaw'):
                    noOfRickshaws += 1
    # print(noOfCars)
    greenTime = math.ceil(((noOfCars*carTime) + (noOfRickshaws*rickshawTime) + (noOfBuses*busTime) + (noOfTrucks*truckTime)+ (noOfBikes*bikeTime))/(noOfLanes+1))
    # greenTime = math.ceil((noOfVehicles)/noOfLanes) 
    print('Green Time: ',greenTime)
    if(greenTime<defaultMinimum):
        greenTime = defaultMinimum
    elif(greenTime>defaultMaximum):
        greenTime = defaultMaximum
    # greenTime = random.randint(15,50)
    signals[(currentGreen+1)%(noOfSignals)].green = greenTime
   
def repeat():
    global currentGreen, currentYellow, nextGreen
    global emergencyActive, emergencyTarget, emergencyStage, emergencyPopupStart, emergencyHoldUntil, emergencyPopupText

    while True:
        # MAIN green-phase loop for currentGreen
        while signals[currentGreen].green > 0:
            # Emergency detection (unchanged)
            if emergencyActive == 0:
                target = detect_emergency()
                if target is not None:
                    emergencyActive = 1
                    emergencyTarget = target
                    emergencyStage = 'prepare'
                    emergencyPopupStart = time.time()
                    emergencyHoldUntil = time.time() + 4.0
                    emergencyPopupText = "Emergency Vehicle Detected! Lane: " + directionNumbers[emergencyTarget].upper()

            # If emergency is active and in prepare/serving stages:
            if emergencyActive == 1 and emergencyStage in ['prepare', 'serving']:
                # If emergency is on the same lane as currentGreen -> extend/serve
                if emergencyTarget == currentGreen:
                    if has_pending_emergency(currentGreen):
                        # ensure we are in serving stage (we keep green for this lane)
                        emergencyStage = 'serving'
                        # gently ensure at least a small green remains so vehicles keep moving
                        signals[currentGreen].green = max(signals[currentGreen].green, 1)
                    else:
                        # emergency cleared on this lane
                        emergencyActive = 0
                        emergencyTarget = None
                        emergencyStage = 'idle'
                else:
                    # PREPARE stage: do NOT zero green instantly (fix)
                    if emergencyStage == 'prepare':
                        # compute remaining seconds to the hold expiry
                        remaining = emergencyHoldUntil - time.time()
                        # if the current green is longer than the remaining clearance window,
                        # shorten it so the green will end close to emergencyHoldUntil (but not instantly).
                        if remaining > 0 and signals[currentGreen].green > remaining + 0.5:
                            # set to ceil(remaining) but at least 1 second
                            signals[currentGreen].green = max(1, int(math.ceil(remaining)))
                        # do nothing else here — we let the normal countdown proceed toward the switch

            # When actively serving emergency, force other lanes to hard red
            if emergencyActive == 1 and emergencyStage == 'serving':
                for i in range(noOfSignals):
                    if i != currentGreen:
                        signals[i].green = 0
                        signals[i].yellow = 0
                        signals[i].red = max(signals[i].red, 2)

            # Watchdog: prevent stall (kept as-is)
            dir_name = directionNumbers[currentGreen]
            now = time.time()
            crossed_now = vehicles[dir_name]['crossed']
            if emergencyActive == 1 and emergencyStage == 'serving':
                if crossed_now > crossedProgress[dir_name]['last_count']:
                    crossedProgress[dir_name]['last_count'] = crossed_now
                    crossedProgress[dir_name]['last_time'] = now
                elif now - crossedProgress[dir_name]['last_time'] > 3:
                    # Nudge leading vehicles over stop line and reduce gaps to clear
                    for lane in [0, 1, 2]:
                        if len(vehicles[dir_name][lane]) > 0:
                            v0 = vehicles[dir_name][lane][0]
                            if dir_name == 'right':
                                v0.stop = min(v0.stop, stopLines[dir_name] - 1)
                                v0.x += 1.5
                            elif dir_name == 'down':
                                v0.stop = min(v0.stop, stopLines[dir_name] - 1)
                                v0.y += 1.5
                            elif dir_name == 'left':
                                v0.stop = max(v0.stop, stopLines[dir_name] + 1)
                                v0.x -= 1.5
                            elif dir_name == 'up':
                                v0.stop = max(v0.stop, stopLines[dir_name] + 1)
                                v0.y -= 1.5
                    crossedProgress[dir_name]['last_time'] = now

            printStatus()
            updateValues()
            # compute live per-direction stats for UI (vehicles passed in current green and ETA)
            for dnum in range(noOfSignals):
                dir_name_iter = directionNumbers[dnum]
                # vehicles passed during current green: delta from start context
                start_cross = agentContext[dnum]['crossed_start']
                passed_now = max(0, vehicles[dir_name_iter]['crossed'] - start_cross)
                # store as textual counter in vehicleCountTexts temporarily for UI usage
                # UI render section will show dedicated messages; we keep logic here light
                # (no persistent change required beyond computing per frame)
            

            # Normal detection-trigger for setTime only when no emergency is pending
            if emergencyActive == 0 and (signals[(currentGreen + 1) % (noOfSignals)].red == detectionTime):
                thread = threading.Thread(name="detection", target=setTime, args=())
                thread.daemon = True
                thread.start()

            time.sleep(1)

        # End of green; enter yellow
        currentYellow = 1
        # Ensure clean transition - set proper yellow timer
        signals[currentGreen].green = 0
        signals[currentGreen].red = 0
        signals[currentGreen].yellow = defaultYellow
        vehicleCountTexts[currentGreen] = "0"

        # reset stop coordinates of lanes and vehicles (preserve original)
        for i in range(0, 3):
            stops[directionNumbers[currentGreen]][i] = defaultStop[directionNumbers[currentGreen]]
            for vehicle in vehicles[directionNumbers[currentGreen]][i]:
                vehicle.stop = defaultStop[directionNumbers[currentGreen]]

        # If an emergency is preparing on a different lane, enforce a 4s yellow for current lane
        if emergencyActive == 1 and emergencyStage == 'prepare' and emergencyTarget is not None and emergencyTarget != currentGreen:
            signals[currentGreen].yellow = 4

        # Yellow countdown loop. Allow emergency to cancel yellow only if it's being served on same lane
        while signals[currentGreen].yellow > 0:
            # If emergency is serving on this lane and still pending, cancel yellow
            if emergencyActive == 1 and emergencyStage == 'serving' and emergencyTarget == currentGreen and has_pending_emergency(currentGreen):
                currentYellow = 0
                signals[currentGreen].yellow = 0
                signals[currentGreen].green = max(10, signals[currentGreen].green)
                break

            printStatus()
            updateValues()
            time.sleep(1)

        # End of yellow phase
        currentYellow = 0
        signals[currentGreen].yellow = 0
        signals[currentGreen].green = 0
        signals[currentGreen].red = defaultRed

        if emergencyActive == 1 and emergencyStage == 'prepare':
            # Small all-red (clear intersection)
            for i in range(noOfSignals):
                signals[i].green = 0
                signals[i].yellow = 0
                signals[i].red = 2   # all-red for safety
            # Engage freeze to keep vehicles behind stop line stationary
            global freezeUntil
            freezeUntil = time.time() + 1.5
            time.sleep(1.5)  # slightly longer to avoid overlap flicker

            # Switch directly to emergency lane (skip default nextGreen)
            currentGreen = emergencyTarget
            nextGreen = (currentGreen + 1) % noOfSignals
            queued = count_queue_in_direction(currentGreen)
            base_green = max(10, min(30, queued))
            signals[currentGreen].green = base_green
            signals[currentGreen].yellow = defaultYellow
            signals[currentGreen].red = 0
            # Force all other lanes to firm red so their timers count down cleanly (no 1s blink)
            for i in range(noOfSignals):
                if i != currentGreen:
                    signals[i].green = 0
                    signals[i].yellow = 0
                    signals[i].red = signals[currentGreen].green + signals[currentGreen].yellow + 2
            emergencyStage = 'serving'

            # Important: skip reset/advance logic below in this cycle
            # so old lane does not momentarily re-green
            continue

        # If we are not serving emergency, reset leaving lane safely (do not pre-grant default green)
        if not (emergencyActive == 1 and emergencyStage == 'serving' and has_pending_emergency(currentGreen)):
            # RL: close current episode for agent of currentGreen and compute reward
            # reward: vehicles passed during green - penalty for overall max wait
            dir_idx = currentGreen
            if dir_idx not in agents:
                agents[dir_idx] = RLAgent(dir_idx)
            crossed_before = agentContext[dir_idx]['crossed_start']
            crossed_after = vehicles[directionNumbers[dir_idx]]['crossed']
            vehicles_served = max(0, crossed_after - crossed_before)
            overall_max_wait = max(get_max_wait_time(0), get_max_wait_time(1), get_max_wait_time(2), get_max_wait_time(3))
            reward = float(vehicles_served) - wait_penalty * overall_max_wait
            agents[dir_idx].finish_serving(reward)

            # Set the current signal to red state
            signals[currentGreen].green = 0
            signals[currentGreen].yellow = 0
            signals[currentGreen].red = defaultRed

        # If an emergency was being served and is now done, clear state
        if emergencyActive == 1 and emergencyStage == 'serving':
            if not has_pending_emergency(currentGreen):
                emergencyActive = 0
                emergencyTarget = None
                emergencyStage = 'idle'

        # Advance to next lane only when not actively serving an emergency
        if not (emergencyActive == 1 and emergencyStage == 'serving' and has_pending_emergency(currentGreen)):
            # Pick next lane by queue length (with fair tie-breaking)
            chosen = select_next_green_index(currentGreen)
            currentGreen = chosen
            nextGreen = select_next_green_index(currentGreen)
            # Dynamic green time from queue size
            queued = count_queue_in_direction(currentGreen)
            # scale: every vehicle ~1s, clamped to [defaultMinimum, defaultMaximum]
            dynamic_green = max(defaultMinimum, min(defaultMaximum, queued))
            signals[currentGreen].green = dynamic_green
            signals[currentGreen].yellow = defaultYellow
            signals[currentGreen].red = 0
            # Set red timer for next signal
            signals[nextGreen].red = signals[currentGreen].green + signals[currentGreen].yellow

            # RL: start episode for the newly green lane
            if currentGreen not in agents:
                agents[currentGreen] = RLAgent(currentGreen)
            agents[currentGreen].start_serving()
            agentContext[currentGreen]['crossed_start'] = vehicles[directionNumbers[currentGreen]]['crossed']
            agentContext[currentGreen]['start_time'] = time.time()

        # Continue the outer while True loop (no recursion)
        # set the red time of next to next signal as (yellow time + green time) of next signal
        # else: keep currentGreen as emergency lane and continue loop

# Print the signal timers on cmd
def printStatus():                                                                                           
	for i in range(0, noOfSignals):
		if(i==currentGreen):
			if(currentYellow==0):
				print(" GREEN TS",i+1,"-> r:",max(0,signals[i].red)," y:",max(0,signals[i].yellow)," g:",max(0,signals[i].green))
			else:
				print("YELLOW TS",i+1,"-> r:",max(0,signals[i].red)," y:",max(0,signals[i].yellow)," g:",max(0,signals[i].green))
		else:
			print("   RED TS",i+1,"-> r:",max(0,signals[i].red)," y:",max(0,signals[i].yellow)," g:",max(0,signals[i].green))
	# Emergency logging compatible with dashboard
	total_emergency_passed = emergencyCounts['right'] + emergencyCounts['down'] + emergencyCounts['left'] + emergencyCounts['up']
	print("Emergency Vehicles Passed:", total_emergency_passed)
	# Dashboard line (machine-readable)
	print("DASHBOARD|green=",currentGreen+1,"|yellow=",currentYellow,"|r=",[max(0,s.red) for s in signals],"|y=",[max(0,s.yellow) for s in signals],"|g=",[max(0,s.green) for s in signals],"|emergency=",{"right": emergencyCounts['right'], "down": emergencyCounts['down'], "left": emergencyCounts['left'], "up": emergencyCounts['up']})
	print()

# Update values of the signal timers after every second
def updateValues():
    for i in range(0, noOfSignals):
        if(i==currentGreen):
            if(currentYellow==0):
                signals[i].green=max(0, signals[i].green-1)
                signals[i].totalGreenTime+=1
            else:
                signals[i].yellow=max(0, signals[i].yellow-1)
        else:
            signals[i].red=max(0, signals[i].red-1)

# Generating vehicles in the simulation
def generateVehicles():
    while(True):
        # Bias to normal vehicles; emergency vehicles appear rarely and respect cooldown
        global lastEmergencySpawnAt
        r = random.random()
        now = time.time()
        # Enforce cooldown so an emergency at one junction does not re-spawn for a while
        emergencyAllowed = (now - lastEmergencySpawnAt) >= emergencyMinGapSeconds
        if r < emergencySpawnProb and emergencyAllowed:
            vehicle_type = random.choice([5,6,7])
            lastEmergencySpawnAt = now
        else:
            vehicle_type = random.randint(0,4)
        if(vehicle_type==4):
            lane_number = 0
        else:
            lane_number = random.randint(0,1) + 1
        will_turn = 0
        if(lane_number==2):
            temp = random.randint(0,4)
            if(temp<=2):
                will_turn = 1
            elif(temp>2):
                will_turn = 0
        temp = random.randint(0,999)
        direction_number = 0
        a = [400,800,900,1000]
        if(temp<a[0]):
            direction_number = 0
        elif(temp<a[1]):
            direction_number = 1
        elif(temp<a[2]):
            direction_number = 2
        elif(temp<a[3]):
            direction_number = 3
        Vehicle(lane_number, vehicleTypes[vehicle_type], direction_number, directionNumbers[direction_number], will_turn)
        time.sleep(0.75)

def simulationTime():
    global timeElapsed, simTime
    while(True):
        timeElapsed += 1
        time.sleep(1)
        if(timeElapsed==simTime):
            print_results_and_exit()
    

class Main:
    # Colours 
    black = (0, 0, 0)
    white = (255, 255, 255)

    # Screensize 
    screenWidth = 1400
    screenHeight = 800
    screenSize = (screenWidth, screenHeight)

    # Setting background image i.e. image of intersection
    background = pygame.image.load('images/mod_int.png')

    screen = pygame.display.set_mode(screenSize)
    pygame.display.set_caption("SIMULATION")

    # Loading signal images and font
    redSignal = pygame.image.load('images/signals/red.png')
    yellowSignal = pygame.image.load('images/signals/yellow.png')
    greenSignal = pygame.image.load('images/signals/green.png')
    font = pygame.font.Font(None, 30)
    buttonFont = pygame.font.Font(None, 28)
    smallFont = pygame.font.Font(None, 24)
    exitButtonRect = pygame.Rect(1240, 20, 120, 36)

    # Start threads only after display is ready
    thread4 = threading.Thread(name="simulationTime",target=simulationTime, args=()) 
    thread4.daemon = True
    thread4.start()

    thread2 = threading.Thread(name="initialization",target=initialize, args=())    # initialization
    thread2.daemon = True
    thread2.start()

    thread3 = threading.Thread(name="generateVehicles",target=generateVehicles, args=())    # Generating vehicles
    thread3.daemon = True
    thread3.start()

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                print_results_and_exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE or event.key == pygame.K_s or event.key == pygame.K_q:
                    print_results_and_exit()
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if exitButtonRect.collidepoint(event.pos):
                    print_results_and_exit()

        screen.blit(background,(0,0))   # display background in simulation
        for i in range(0,noOfSignals):  # display signal and set timer according to current status: green, yellow, or red
            # Determine signal state based on timer values, not global variables
            if(i==currentGreen):
                # Current green signal - determine phase based on timer values
                if(signals[i].green > 0):
                    # Green phase
                    signals[i].signalText = signals[i].green
                    screen.blit(greenSignal, signalCoods[i])
                elif(signals[i].yellow > 0):
                    # Yellow phase
                    signals[i].signalText = signals[i].yellow
                    screen.blit(yellowSignal, signalCoods[i])
                else:
                    # Red phase (no green or yellow timer)
                    signals[i].signalText = "STOP"
                    screen.blit(redSignal, signalCoods[i])
            else:
                # Non-current signals - always show red
                if(signals[i].red<=10 and signals[i].red>0):
                    signals[i].signalText = signals[i].red
                elif(signals[i].red==0):
                    signals[i].signalText = "GO"
                else:
                    signals[i].signalText = "---"
                screen.blit(redSignal, signalCoods[i])
        signalTexts = ["","","",""]

        # display signal timer and vehicle count
        for i in range(0,noOfSignals):  
            signalTexts[i] = font.render(str(signals[i].signalText), True, white, black)
            screen.blit(signalTexts[i],signalTimerCoods[i]) 
            displayText = vehicles[directionNumbers[i]]['crossed']
            vehicleCountTexts[i] = font.render(str(displayText), True, black, white)
            screen.blit(vehicleCountTexts[i],vehicleCountCoods[i])

        timeElapsedText = font.render(("Time Elapsed: "+str(timeElapsed)), True, black, white)
        screen.blit(timeElapsedText,(1100,50))

        # Exit button UI
        pygame.draw.rect(screen, (200,40,40), exitButtonRect, border_radius=6)

        exitLabel = buttonFont.render("Exit", True, (255,255,255))
        screen.blit(exitLabel, (exitButtonRect.x + 36, exitButtonRect.y + 8))

        # Emergency popup
        if emergencyActive == 1 and emergencyPopupStart > 0:
            elapsed = time.time() - emergencyPopupStart
            if elapsed <= emergencyPopupDuration:
                # Fade out alpha from 255 to 0
                alpha = max(0, 255 - int((elapsed/emergencyPopupDuration) * 255))
                popup_surface = pygame.Surface((500, 50))
                popup_surface.set_alpha(alpha)
                popup_surface.fill((200,0,0))
                text = font.render(emergencyPopupText, True, white, black)
                popup_surface.blit(text, (20, 15))
                screen.blit(popup_surface, (450, 20))

        # display the vehicles
        for vehicle in simulation:  
            screen.blit(vehicle.currentImage, [vehicle.x, vehicle.y])
            # vehicle.render(screen)
            vehicle.move()
        
        # Directional info overlays: vehicles passed this green, ETA, distance
        # Position text in corners of the screen
        corner_positions = {
            0: (20, 20),    # top-left corner for right direction
            1: (screenWidth - 250, 20),    # top-right corner for down direction  
            2: (screenWidth - 250, screenHeight - 100),  # bottom-right corner for left direction
            3: (20, screenHeight - 100),   # bottom-left corner for up direction
        }
        
        for dnum in range(noOfSignals):
            dir_name = directionNumbers[dnum]
            pos = corner_positions[dnum]
            
            # Direction label
            direction_label = smallFont.render(f"Direction: {dir_name.upper()}", True, (0, 100, 200), (255,255,255))
            screen.blit(direction_label, pos)
            pos = (pos[0], pos[1] + 20)
            
            # count vehicles passed during current green for that direction
            passed = max(0, vehicles[dir_name]['crossed'] - agentContext[dnum]['crossed_start']) if dnum == currentGreen else 0
            # Dynamic ETA calculation based on vehicle density
            eta_seconds = calculate_dynamic_eta(dir_name, passed)
            eta_text = format_eta(eta_seconds)
            
            # Determine density level for color coding
            if passed <= 5:
                density_level = "Very Low"
                density_color = (0, 150, 0)  # Green
            elif passed <= 10:
                density_level = "Low"
                density_color = (0, 200, 0)  # Light Green
            elif passed <= 20:
                density_level = "Medium"
                density_color = (255, 165, 0)  # Orange
            else:
                density_level = "High"
                density_color = (255, 0, 0)  # Red
            
            # Emergency special message if any passed recently
            now = time.time()
            if emergencyETAByDir[dir_name] is not None and emergencyMessageUntil[dir_name] > now:
                em_eta = format_eta(emergencyETAByDir[dir_name])
                msg1 = smallFont.render(f"Emergency ETA: {em_eta}", True, (220, 20, 20), (255,255,255))
                screen.blit(msg1, pos)
                pos = (pos[0], pos[1] + 18)
            
            msg_passed = smallFont.render(f"Passed this green: {passed}", True, black, white)
            screen.blit(msg_passed, pos)
            pos = (pos[0], pos[1] + 18)
            
            # Density level indicator
            msg_density = smallFont.render(f"Density: {density_level}", True, density_color, white)
            screen.blit(msg_density, pos)
            pos = (pos[0], pos[1] + 18)
            
            msg_eta = smallFont.render(f"ETA to next: {eta_text}", True, black, white)
            screen.blit(msg_eta, pos)
            pos = (pos[0], pos[1] + 18)
            
            msg_dist = smallFont.render(f"Next distance: {int(nextJunctionDistanceMeters)} m", True, black, white)
            screen.blit(msg_dist, pos)

        pygame.display.update()

Main()

  
