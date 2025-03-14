import numpy as np
import pickle
import random
import gym
import sys

def get_relative_pos(taxi_position, station_position):
    """
    Determines the relative position of the station with respect to the taxi.
    The space is divided into four quadrants:
        - 0: Station is to the top-right of the taxi
        - 1: Station is to the top-left of the taxi
        - 2: Station is to the bottom-right of the taxi
        - 3: Station is to the bottom-left of the taxi
    """
    is_left = station_position[0] < taxi_position[0]
    is_below = station_position[1] < taxi_position[1]
    
    return (is_left * 2) + is_below

def get_dropoff_state(destination_visited, obs):
    """State representation for pickup phase with obstacle-direction awareness"""
    taxi_pos = (obs[0], obs[1])
    stations = [(obs[i], obs[i+1]) for i in range(2, 10, 2)]
    
    # Find closest unvisited station (existing code)
    distances = []
    for i, station in enumerate(stations):
        if not destination_visited[i]:
            dist = abs(taxi_pos[0] - station[0]) + abs(taxi_pos[1] - station[1])
            distances.append((i, dist))
    
    if distances:
        closest_idx, _ = min(distances, key=lambda x: x[1])
        closest_station = stations[closest_idx]
    else:
        destination_visited = [0, 0, 0, 0]
        closest_station = min(stations, key=lambda s: abs(taxi_pos[0] - s[0]) + abs(taxi_pos[1] - s[1]))

    # Get relative directions
    rel_row = closest_station[0] - taxi_pos[0]
    rel_col = closest_station[1] - taxi_pos[1]
    
    # Discretize directions
    rel_row_dir = 1 if rel_row > 0 else (-1 if rel_row < 0 else 0)
    rel_col_dir = 1 if rel_col > 0 else (-1 if rel_col < 0 else 0)

    # New: Check obstacles in target directions
    obstacle_in_row_dir = 0
    obstacle_in_col_dir = 0
    
    # Check vertical (row) direction obstacles
    if rel_row_dir == 1:
        obstacle_in_row_dir = obs[11]  # obstacle_south
    elif rel_row_dir == -1:
        obstacle_in_row_dir = obs[10]  # obstacle_north
        
    # Check horizontal (col) direction obstacles
    if rel_col_dir == 1:
        obstacle_in_col_dir = obs[12]  # obstacle_east
    elif rel_col_dir == -1:
        obstacle_in_col_dir = obs[13]  # obstacle_west

    # Passenger status
    at_destination = (rel_col_dir == 0 and rel_row_dir == 0 and obs[15])

    return (
        rel_row_dir,            # Vertical direction to target (-1, 0, 1)
        rel_col_dir,            # Horizontal direction to target (-1, 0, 1)
        obstacle_in_row_dir,    # 1 if obstacle exists in target vertical direction
        obstacle_in_col_dir,    # 1 if obstacle exists in target horizontal direction
        obs[10],                # obstacle_north (full context)
        obs[11],                # obstacle_south
        obs[12],                # obstacle_east
        obs[13],                # obstacle_west
        at_destination          # destination at location
    )

def get_pickup_state(visited, obs):
    """State representation for pickup phase with obstacle-direction awareness"""
    taxi_pos = (obs[0], obs[1])
    stations = [(obs[i], obs[i+1]) for i in range(2, 10, 2)]
    
    # Find closest unvisited station (existing code)
    distances = []
    for i, station in enumerate(stations):
        if not visited[i]:
            dist = abs(taxi_pos[0] - station[0]) + abs(taxi_pos[1] - station[1])
            distances.append((i, dist))
    
    if distances:
        closest_idx, _ = min(distances, key=lambda x: x[1])
        closest_station = stations[closest_idx]
    else:
        visited = [0, 0, 0, 0]
        closest_station = min(stations, key=lambda s: abs(taxi_pos[0] - s[0]) + abs(taxi_pos[1] - s[1]))

    # Get relative directions
    rel_row = closest_station[0] - taxi_pos[0]
    rel_col = closest_station[1] - taxi_pos[1]
    
    # Discretize directions
    rel_row_dir = 1 if rel_row > 0 else (-1 if rel_row < 0 else 0)
    rel_col_dir = 1 if rel_col > 0 else (-1 if rel_col < 0 else 0)

    # New: Check obstacles in target directions
    obstacle_in_row_dir = 0
    obstacle_in_col_dir = 0
    
    # Check vertical (row) direction obstacles
    if rel_row_dir == 1:
        obstacle_in_row_dir = obs[11]  # obstacle_north
    elif rel_row_dir == -1:
        obstacle_in_row_dir = obs[10]  # obstacle_south
        
    # Check horizontal (col) direction obstacles
    if rel_col_dir == 1:
        obstacle_in_col_dir = obs[12]  # obstacle_east
    elif rel_col_dir == -1:
        obstacle_in_col_dir = obs[13]  # obstacle_west

    # Passenger status
    at_passenger = (rel_col_dir == 0 and rel_row_dir == 0 and obs[14])

    return (
        rel_row_dir,            # Vertical direction to target (-1, 0, 1)
        rel_col_dir,            # Horizontal direction to target (-1, 0, 1)
        obstacle_in_row_dir,    # 1 if obstacle exists in target vertical direction
        obstacle_in_col_dir,    # 1 if obstacle exists in target horizontal direction
        obs[10],                # obstacle_north (full context)
        obs[11],                # obstacle_south
        obs[12],                # obstacle_east
        obs[13],                # obstacle_west
        at_passenger            # passenger at location
    )

visited = [0, 0, 0, 0]
destination_visited = [0, 0, 0, 0]
phase = 'pickup'

# def get_action(obs):
#     # Load the trained Q-table
#     global phase, visited, destination_visited
#     try:
#         with open('q_pickup_best.pkl', 'rb') as f:
#             pickup_table = pickle.load(f)
#     except FileNotFoundError:
#         print('pickup table not found!!')
#         return random.choice([0, 1, 2, 3, 4, 5])
#     try:
#         with open('q_dropoff_best.pkl', 'rb') as f:
#             dropoff_table = pickle.load(f)
#     except FileNotFoundError:
#         print('dropoff table not found!!')
#         return random.choice([0, 1, 2, 3, 4, 5])
#     print(phase)
#     if phase == 'pickup':
#         state = get_pickup_state(visited, obs)
#         stations = [(obs[i], obs[i + 1]) for i in range(2, 10, 2)]
#         for station in stations:
#             if np.array_equal((obs[0], obs[1]), station):
#                 visited[stations.index(station)] = True
#         q_table = pickup_table
#     else:
#         state = get_dropoff_state(destination_visited, obs)
#         stations = [(obs[i], obs[i + 1]) for i in range(2, 10, 2)]
#         for station in stations:
#             if np.array_equal((obs[0], obs[1]), station):
#                 destination_visited[stations.index(station)] = True
#         q_table = dropoff_table
#         # q_table = pickup_table
#     # Choose the best action based on Q-table
#     print(state)
#     if state in q_table:
#         action = np.argmax(q_table[state])
#         obs_arr = np.array(obs)
            
#         if action == 4 and obs[14]:
#             if any(np.array_equal((obs_arr[0], obs_arr[1]), station) for station in stations):
#                 print('Switch phase')
#                 phase = 'dropoff'
#         return action
#     else:
#         # If state not found in Q-table, return a random action
#         print('state not found')
#         return random.choice([0, 1, 2, 3, 4, 5])

def get_action(obs):
    global phase, visited, destination_visited
    # Load the trained policies
    try:
        with open('q_pickup_leaderboard.pkl', 'rb') as f:
            pickup_policy = pickle.load(f)
    except FileNotFoundError:
        print('Pickup policy not found!!')
        return random.choice([0, 1, 2, 3, 4, 5])
    try:
        with open('q_dropoff_leaderboard.pkl', 'rb') as f:
            dropoff_policy = pickle.load(f)
    except FileNotFoundError:
        print('Dropoff policy not found!!')
        return random.choice([0, 1, 2, 3, 4, 5])

    # Get current phase state
    if phase == 'pickup':
        state = get_pickup_state(visited, obs)
        policy_logits = pickup_policy
        # Update visited stations
        stations = [(obs[i], obs[i+1]) for i in range(2, 10, 2)]
        for station in stations:
            if np.array_equal((obs[0], obs[1]), station):
                visited[stations.index(station)] = 1
    else:
        state = get_dropoff_state(destination_visited, obs)
        policy_logits = dropoff_policy
        # Update destination visited
        stations = [(obs[i], obs[i+1]) for i in range(2, 10, 2)]
        for station in stations:
            if np.array_equal((obs[0], obs[1]), station):
                destination_visited[stations.index(station)] = 1

    # Get action probabilities using softmax
    if state not in policy_logits:
        # If state unknown, use uniform random policy
        return random.choice([0, 1, 2, 3, 4, 5])
    
    logits = policy_logits[state]
    # Numerically stable softmax
    max_logit = np.max(logits)
    exp_logits = np.exp(logits - max_logit)
    probs = exp_logits / np.sum(exp_logits)
    
    # Sample action from probability distribution
    action = np.random.choice(6, p=probs)

    # Handle phase transition (same as original logic)
    obs_arr = np.array(obs)
    stations = [(obs[i], obs[i+1]) for i in range(2, 10, 2)]
    if action == 4 and obs[14]:  # If pickup action and passenger present
        if any(np.array_equal((obs_arr[0], obs_arr[1]), station) for station in stations):
            print('Switch phase to dropoff')
            phase = 'dropoff'

    return action
