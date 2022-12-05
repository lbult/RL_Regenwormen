import os
import gym
import time
import math
import random
from gym.spaces.dict import Dict
import numpy as np
from gym.core import Env
import matplotlib.pyplot as plt
from gym import spaces

 

class RL_regenwormen(gym.Env):
  """Custom Environment that follows gym interface"""
  metadata = {'render.modes': ['human']}

  def __init__(self, render):
    super(RL_regenwormen, self).__init__()
    # Define action and observation space
    # They must be gym.spaces objects
    self.done = False

    # define the tiles with index 0-15
    self.values = np.array([1,1,1,1,2,2,2,2,3,3,3,3,4,4,4,4])
    self.tiles = np.ones(16)

    # define the dice and dice during the current round
    self.dice_left = 8
    self.dice = np.zeros(8)
    self.dice_faces = np.ones(6)

    # define current score and tiles owned
    self.tiles_owned = [0]
    self.current_score = 0
    self.total_dice_value = 0

    # define action space
    self.action_space = spaces.Box(low=0,high=1.2,shape=(2,),dtype=np.float32)
    # define observation space
    self.observation_space = spaces.Box(
        -np.inf, np.inf, shape=(33,), dtype=np.float32
    )

    # prepare the first dice throw
    self.dice_choice = np.zeros(8)
    count = 0
    while count < 8:
        if count < self.dice_left-1:
            self.dice_choice[count] = random.randint(1,7)
        else:
            self.dice_choice[count] = 0
        count += 1
    
    # variable to check if a regenworm tile is owned
    self.regenworm = False


    # define the observation space
    self.state = [
        self.tiles[0],self.tiles[4],self.tiles[8],self.tiles[12],
        self.tiles[1],self.tiles[5],self.tiles[9],self.tiles[13],
        self.tiles[2],self.tiles[6],self.tiles[10],self.tiles[14],
        self.tiles[3],self.tiles[7],self.tiles[11],self.tiles[15],
        self.tiles_owned[-1],
        self.current_score,
        self.dice_choice[0],self.dice_choice[1],self.dice_choice[2],self.dice_choice[3],
        self.dice_choice[4],self.dice_choice[5],self.dice_choice[6],self.dice_choice[7],
        self.total_dice_value,
        self.dice_faces[0],self.dice_faces[1],self.dice_faces[2],self.dice_faces[3],
        self.dice_faces[4],self.dice_faces[5]
    ]

  def step(self, action):
    reward = 0
    no_choice = True
    choice = 0
    next_round = False
    prev_count = self.total_dice_value
    self.regenworm = False

    if sum(self.dice_faces) == 0:
        next_round = True
    else:
        if (0.0 <= action[0] < 0.2):
            counter = 0
            for i in self.dice_choice:
                if i == 1:
                    counter += 1

            if (self.dice_faces[0] == 1) and (counter > 0):
                self.dice_faces[0] = 0
                self.dice_left -= counter
                self.total_dice_value += counter * 1
                no_choice = False
            else:
                next_round = True
                reward -= 100000

        elif (0.2 <= action[0] < 0.4) or choice == 2:
            counter = 0
            for i in self.dice_choice:
                if i == 2:
                    counter += 1

            if (self.dice_faces[1] == 1) and (counter > 0):
                self.dice_faces[1] = 0
                self.dice_left -= counter
                self.total_dice_value += counter * 2
                no_choice = False
            else:
                next_round = True
                reward -= 100000

        elif (0.4 <= action[0] < 0.6) or choice == 3:
            counter = 0
            for i in self.dice_choice:
                if i == 3:
                    counter += 1

            if (self.dice_faces[2] == 1) and (counter > 0):
                self.dice_faces[2] = 0
                self.dice_left -= counter
                self.total_dice_value += counter * 3
                no_choice = False
            else:
                next_round = True
                reward -= 100000

        elif (0.6 <= action[0] < 0.8) or choice == 4:
            counter = 0
            for i in self.dice_choice:
                if i == 4:
                    counter += 1

            if (self.dice_faces[3] == 1) and (counter > 0):
                self.dice_faces[3] = 0
                self.dice_left -= counter
                self.total_dice_value += counter * 4
                no_choice = False
            else:
                next_round = True
                reward -= 100000

        elif (0.8 <= action[0] < 1.0) or choice == 5:
            counter = 0
            for i in self.dice_choice:
                if i == 5:
                    counter += 1
            if (self.dice_faces[4] == 1) and (counter > 0):
                reward = 1
                self.dice_faces[4] = 0
                self.dice_left -= counter
                self.total_dice_value += counter * 5
                no_choice = False
            else:
                next_round = True
                reward -= 100000

        elif (1.0 <= action[0] < 1.2) or choice == 6:
            
            counter = 0
            for i in self.dice_choice:
                if i == 6:
                    counter += 1

            if (self.dice_faces[5] == 1) and (counter > 0):
                self.dice_faces[5] = 0
                self.dice_left -= counter
                self.total_dice_value += counter * 5
                self.regenworm = True
        else:
            next_round = True
            reward -= 100000

    if self.total_dice_value < 21:
        reward += -1000*(1-self.total_dice_value/21)
    elif self.total_dice_value > 21:
        reward += 1000*(self.total_dice_value/21)

    reward += 5000*(self.total_dice_value-prev_count)
    #print(self.total_dice_value)
    #print(self.dice_choice)

    self.done = False
    #print(action[0])
    if (action[1] > 0.6) or (sum(self.dice_faces) == 0) or (self.dice_left == 0):
        #print(self.total_dice_value)
        next_round = True
        if 37 > self.total_dice_value >= 21 and self.regenworm:
            if self.tiles[self.total_dice_value-21] == 1:
                delta_score = 100000*self.values[self.total_dice_value-21]
                #delta_score = self.values[self.total_dice_value-21]
                self.tiles[self.total_dice_value-21] = 0
                self.tiles_owned.append(self.total_dice_value)
                print("Got one")
                print(self.tiles)
            elif self.tiles_owned[-1] == 0:
                #   do nothing
                nothing = "do"
                delta_score = -10
            elif self.tiles[self.total_dice_value-21] == 0:
                delta_score = -10000
                #delta_score = -self.values[self.tiles_owned[-1]-21]
                self.tiles[self.tiles_owned[-1]-21] = 1
                del self.tiles_owned[-1]
        else:
            if self.tiles_owned[-1] == 0:
                #   do nothing
                nothing = "do"
                delta_score = -10000
            else:
                delta_score = -10000
                self.tiles[self.tiles_owned[-1]-21] = 1
                del self.tiles_owned[-1]

        # set reward equal to a sum of the total score and gained points
        reward += 0.1 + self.current_score * 0.3 + 0.7 * delta_score
        if self.regenworm:
            reward += 5000

    if next_round:
        # all dice faces can be chosen again
        self.dice_faces = np.ones(6)
        self.dice_left = 8

        if sum(self.tiles) == 0:
            print(self.current_score)
            self.done = True
        
        self.total_dice_value = 0
    elif (action[1] < 0.6) and (self.total_dice_value < 21):
        reward += 10000
    
    count = 0
    while count < 8:
        if count < self.dice_left-1:
            self.dice_choice[count] = random.randint(1,6)
        else:
            self.dice_choice[count] = 0
        count += 1
    
    delta_score = 0
    
    #print(self.tiles)

    self.state = [
        self.tiles[0],self.tiles[4],self.tiles[8],self.tiles[12],
        self.tiles[1],self.tiles[5],self.tiles[9],self.tiles[13],
        self.tiles[2],self.tiles[6],self.tiles[10],self.tiles[14],
        self.tiles[3],self.tiles[7],self.tiles[11],self.tiles[15],
        self.tiles_owned[-1],
        self.current_score,
        self.dice_choice[0],self.dice_choice[1],self.dice_choice[2],self.dice_choice[3],
        self.dice_choice[4],self.dice_choice[5],self.dice_choice[6],self.dice_choice[7],
        self.total_dice_value,
        self.dice_faces[0],self.dice_faces[1],self.dice_faces[2],self.dice_faces[3],
        self.dice_faces[4],self.dice_faces[5]
    ]
    assert len(self.state) == 33


    return np.array(self.state, dtype=np.float32), reward, self.done, {}
  
  def reset(self):
    self.done = False

    # define the tiles with index 0-15
    self.values = np.array([1,1,1,1,2,2,2,2,3,3,3,3,4,4,4,4])
    self.tiles = np.ones(16)

    # define the dice and dice during the current round
    self.dice_left = 8
    self.dice = np.zeros(8)
    self.dice_faces = np.ones(6)

    # define current score and tiles owned
    self.tiles_owned = [0]
    self.current_score = 0
    self.total_dice_value = 0

    # prepare the first dice throw
    self.dice_choice = np.zeros(8)
    count = 0
    while count < 8:
        if count < self.dice_left-1:
            self.dice_choice[count] = random.randint(1,6)
        else:
            self.dice_choice[count] = 0
        count += 1


    # define the observation space
    self.state = [
        self.tiles[0],self.tiles[4],self.tiles[8],self.tiles[12],
        self.tiles[1],self.tiles[5],self.tiles[9],self.tiles[13],
        self.tiles[2],self.tiles[6],self.tiles[10],self.tiles[14],
        self.tiles[3],self.tiles[7],self.tiles[11],self.tiles[15],
        self.tiles_owned[-1],
        self.current_score,
        self.dice_choice[0],self.dice_choice[1],self.dice_choice[2],self.dice_choice[3],
        self.dice_choice[4],self.dice_choice[5],self.dice_choice[6],self.dice_choice[7],
        self.total_dice_value,
        self.dice_faces[0],self.dice_faces[1],self.dice_faces[2],self.dice_faces[3],
        self.dice_faces[4],self.dice_faces[5]
    ]
    return self.state
  
  def render(self, mode='human', close=False):
    # Render the environment to the screen
    return