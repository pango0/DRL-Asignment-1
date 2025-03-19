import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Categorical
from hsuan import SimpleTaxiEnv  # Make sure this import works

# Global variables to maintain state across calls
policy_net = None
visited = [0, 0, 0, 0]
on_board = False

def get_dir(taxi, station):
    dx = station[0] - taxi[0]
    dy = station[1] - taxi[1]
    if dx == 0 and dy == 0:
        return 0
    if dx > 0 and dy == 0:
        return 1
    if dx < 0 and dy == 0:
        return 2
    if dx == 0 and dy > 0:
        return 3
    if dx == 0 and dy < 0:
        return 4
    if dx > 0 and dy > 0:
        return 5
    if dx > 0 and dy < 0:
        return 6
    if dx < 0 and dy > 0:
        return 7
    if dx < 0 and dy < 0:
        return 8

class PGNetwork(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.policy = torch.nn.Parameter(torch.zeros(576, 6))
        
    def forward(self, state):
        logits = self.policy[state]
        probs = F.softmax(logits, dim=-1)
        return probs

def get_state(obs, target_station):
    global on_board
    taxi_pos = (obs[0], obs[1])
    dir = get_dir(taxi_pos, target_station)
        
    # Use env.passenger_picked_up to determine if passenger is on board
    at_passenger = int(not on_board and dir == 0 and obs[14])
    at_destination = int(on_board and dir == 0 and obs[15])
    
    # Compute state index
    state_index = dir
    state_index = state_index * 2 + int(obs[10])
    state_index = state_index * 2 + int(obs[11])
    state_index = state_index * 2 + int(obs[12])
    state_index = state_index * 2 + int(obs[13])
    state_index = state_index * 2 + at_passenger
    state_index = state_index * 2 + at_destination
    return state_index

def choose_action(state_idx):
    """Sample action from policy probabilities using softmax"""
    global policy_net
    
    if policy_net is None:
        # Load the policy network if not already loaded
        policy_net = PGNetwork()
        policy_net.load_state_dict(torch.load('20000.pt'))
        policy_net.eval()
        
    logits = policy_net.policy[state_idx]
    dist = Categorical(logits=logits)
    action = dist.sample()
    return action.item()

def get_action(obs):
    """Main function called by external code"""
    global visited, on_board
    # Get stations from observation
    stations = [(obs[i], obs[i+1]) for i in range(2, 10, 2)]
    
    # Reset visited list if all stations are visited
    if visited == [1, 1, 1, 1]:
        visited = [0, 0, 0, 0]
    
    # Select target station
    target_station = None
    for i in range(len(visited)):
        if visited[i] == 0:
            target_station = stations[i]
            break
    
    # Default to first station if needed
    if target_station is None:
        target_station = stations[0]
        
    # Update visited stations based on current position
    for i, station in enumerate(stations):
        if (obs[0], obs[1]) == station:
            visited[i] = 1
    
    # Get state and choose action
    state = get_state(obs, target_station)
    action = choose_action(state)
    if action == 4 and obs[14]:
        for station in stations:
            if get_dir((obs[0], obs[1]), station) == 0:
                on_board = True
    if action == 5:
        on_board = False

    
    return action
