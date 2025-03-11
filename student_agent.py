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

on_board = False

def get_action(obs):
    # Load the trained Q-table
    global on_board
    try:
        with open('q.pkl', 'rb') as f:
            q_table = pickle.load(f)
    except FileNotFoundError:
        # If Q-table isn't found, return a random action
        
        return random.choice([0, 1, 2, 3, 4, 5])
    
    # We need to track if we have a passenger on board
    # Convert observation to state representation
    state = get_state(obs, on_board)
    
    # Choose the best action based on Q-table
    if state in q_table:
        action = np.argmax(q_table[state])
        obs_arr = np.array(obs)
        stations = obs_arr[2:10].reshape(4, 2)
        if action == 4 :
            if obs[14] and any(np.array_equal((obs_arr[0], obs_arr[1]), station) for station in stations):
                on_board = True
        if action == 5 :
            if obs[15] and any(np.array_equal((obs_arr[0], obs_arr[1]), station) for station in stations):
                on_board = False
        return action
    else:
        # If state not found in Q-table, return a random action
        return random.choice([0, 1, 2, 3, 4, 5])