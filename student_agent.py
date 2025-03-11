import numpy as np
import pickle
import random
import gym

def get_relative_pos(taxi_pos, station_pos):
    if station_pos[0] - taxi_pos[0] > 0:
        if station_pos[1] - taxi_pos[1] > 0:
            return 0
        else:
            return 1
    else:
        if station_pos[1] - taxi_pos[1] > 0:
            return 2
        else:
            return 3

def get_state(obs, on_board):
    taxi_pos = (obs[0], obs[1])
    station1 = (obs[2], obs[3])
    station2 = (obs[4], obs[5])
    station3 = (obs[6], obs[7])
    station4 = (obs[8], obs[9])
    state = (get_relative_pos(taxi_pos, station1),
             get_relative_pos(taxi_pos, station2),
             get_relative_pos(taxi_pos, station3), 
             get_relative_pos(taxi_pos, station4), 
             obs[10], obs[11], obs[12], obs[13], 
             obs[14], obs[15], on_board)
    return state

def get_action(obs):
    # Load the trained Q-table
    try:
        with open('big_decay_q_table.pkl', 'rb') as f:
            q_table = pickle.load(f)
    except FileNotFoundError:
        # If Q-table isn't found, return a random action
        return random.choice([0, 1, 2, 3, 4, 5])
    
    # We need to track if we have a passenger on board
    # Since this isn't directly provided in the observation, we'll maintain this state
    # This is a limitation - in a real-world scenario, you'd need to handle this properly
    # For this sample, we'll assume no passenger to start
    on_board = False
    
    # Convert observation to state representation
    state = get_state(obs, on_board)
    
    # Choose the best action based on Q-table
    if state in q_table:
        return np.argmax(q_table[state])
    else:
        # If state not found in Q-table, return a random action
        return random.choice([0, 1, 2, 3, 4, 5])