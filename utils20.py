import torch.nn.functional as F
import torch.nn as nn
import argparse
import torch
import numpy as np
from env20 import GameState 

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.nn.functional as F
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, net_width, maxaction):
        super().__init__()
        # shared trunk
        self.net = nn.Sequential(
            nn.Linear(state_dim, net_width),
            nn.ReLU(),
            nn.LayerNorm(net_width),
            nn.Linear(net_width, net_width//2),
            nn.ReLU(),
            nn.LayerNorm(net_width//2),
            nn.Linear(net_width//2, net_width//4),
            nn.ReLU(),
            nn.LayerNorm(net_width//4),
        )
        # two heads
        self.dist_head  = nn.Linear(net_width//4, action_dim)  # for softmax
        self.scale_head = nn.Linear(net_width//4, 1)           # for budget

        self.maxaction = maxaction

    def forward(self, state):
        x = self.net(state)                   # [B, hidden]
        logits = self.dist_head(x)            # [B, action_dim]
        dist   = F.softmax(logits, dim=-1)    # sum to 1

        scale  = torch.sigmoid(self.scale_head(x)).squeeze(-1)  
        # scale in (0,1), shape [B]

        total_power = scale * self.maxaction  # shape [B]
        # expand total_power to [B,action_dim] so we can multiply
        return dist * total_power.unsqueeze(-1)


class Q_Critic(nn.Module):
    def __init__(self, state_dim, action_dim, net_width=1024):
        super().__init__()
        # pertama‐tama embed state saja
        self.l1 = nn.Linear(state_dim, net_width)
        self.ln1 = nn.LayerNorm(net_width)

        # setelah itu concat action, lalu dua layer lagi
        self.l2 = nn.Linear(net_width + action_dim, net_width//2)
        self.ln2 = nn.LayerNorm(net_width//2)

        self.l3 = nn.Linear(net_width//2, net_width//4)
        self.ln3 = nn.LayerNorm(net_width//4)

        # output Q‐value scalar
        self.l4 = nn.Linear(net_width//4, 1)

    def forward(self, state, action):
        """
        state:  Tensor [B, state_dim]
        action: Tensor [B, action_dim]
        """
        x = F.relu(self.ln1(self.l1(state)))             # [B, net_width]
        x = torch.cat([x, action], dim=-1)               # [B, net_width+action_dim]
        x = F.relu(self.ln2(self.l2(x)))                 # [B, net_width//2]
        x = F.relu(self.ln3(self.l3(x)))                 # [B, net_width//4]
        q = self.l4(x)                                   # [B, 1]
        return q

class Q_Critic(nn.Module):
    def __init__(self, state_dim, action_dim, net_width=1024):
        super().__init__()
        # pertama‐tama embed state saja
        self.l1 = nn.Linear(state_dim, net_width)
        self.ln1 = nn.LayerNorm(net_width)

        # setelah itu concat action, lalu dua layer lagi
        self.l2 = nn.Linear(net_width + action_dim, net_width//2)
        self.ln2 = nn.LayerNorm(net_width//2)

        self.l3 = nn.Linear(net_width//2, net_width//4)
        self.ln3 = nn.LayerNorm(net_width//4)

        # output Q‐value scalar
        self.l4 = nn.Linear(net_width//4, 1)

    def forward(self, state, action):
        """
        state:  Tensor [B, state_dim]
        action: Tensor [B, action_dim]
        """
        x = F.relu(self.ln1(self.l1(state)))             # [B, net_width]
        x = torch.cat([x, action], dim=-1)               # [B, net_width+action_dim]
        x = F.relu(self.ln2(self.l2(x)))                 # [B, net_width//2]
        x = F.relu(self.ln3(self.l3(x)))                 # [B, net_width//4]
        q = self.l4(x)                                   # [B, 1]
        return q

def evaluate_policy_reward(channel_gain, state, env, agent, turns=3):
    total_reward = 0
    for j in range(turns):
        for i in range(200):
            # Take deterministic actions at test time
            a = agent.select_action(state, deterministic=True)
            next_loc = env.generate_positions()
            next_channel_gain= env.generate_channel_gain(next_loc)
            s_next, re, dw, tr, info = env.step(a, channel_gain, next_channel_gain)
            #if iterasi == max_iter :
            #    tr ==True
            done = (dw or tr)
            

            total_reward += re
            #iterasi +=1
            #print(i)
            state = s_next
            channel_gain = next_channel_gain
    return int(total_reward/3)

def evaluate_policy(channel_gain, state, env, agent, turns=1):
    env = GameState(20,10)
    total_scores = 0
    total_scores_rand = 0 
    total_EE = 0
    total_EE_rand=0
    total_power = 0
    total_power_rand=0
    # threshold constraint (contoh)
    R_th = 0.15     # minimal data rate per UE [bit/s atau satuan yg kamu pakai]
    P_th = 10  # maksimal total power [W atau satuan yg kamu pakai]

    # Counters untuk constraint
    count_data_ok  = 0
    count_data_ok_rand=0
    count_power_ok = 0
    count_power_ok_rand=0
    total_steps    = 0

    #counter data rate 
    count = 0 
    count_rand=0
    jumlah_data_rate = 0
    jumlah_data_rate_rand=0

    for _ in range(turns):
        done = False
        MAX_STEPS = 1
        step_count = 0

        while not done:
            step_count += 1
            total_steps += 1

            # aksi deterministik
            a = agent.select_action(state, deterministic=True)
            #random_allocation 
            a_rand=env.sample_valid_power()
            # generate next state
            next_loc         = env.generate_positions()
            next_channel_gain= env.generate_channel_gain(next_loc)
            s_next, r, dw, tr, info = env.step(a, channel_gain)
            count_data_ok=info['data_rate_pass']
            data_rate=info['data_rate']
            total_rate=info['total_rate']

            #step dari random 
            s_next1, r1, dw1, tr1, info1 = env.step(a_rand, channel_gain)
            data_rate_rand=info1['data_rate']
            
            print(f'DDPG power : {a}, reward :{r}, total power {np.sum(a)}')
            print(f'random power : {a_rand}, reward :{r1}, total power {np.sum(a_rand)}')
        
            # cek constraint power: total_power ≤ P_th
            if np.sum(a) <= P_th:
                count_power_ok += 1
            # cek constraint power: total_power ≤ P_th untuk random
            if np.sum(a_rand) <= P_th:
                count_power_ok_rand += 1

            # akumulasi reward & metrik lain
            total_scores += r
            total_scores_rand+=r1
            total_EE     += info['EE']
            total_EE_rand +=info1['EE']
            total_power  += info['total_power']
            total_power_rand +=info1['total_power']

            # update loop
            if step_count == MAX_STEPS:
                tr = True
            done = (dw or tr)
            state = s_next
            channel_gain = next_channel_gain

    # hitung rata-rata metrik
    avg_score = total_scores / turns
    avg_score_rand = total_scores_rand / turns
    avg_EE    = total_EE / turns
    avg_EE_rand = total_EE_rand / turns 
    avg_power = total_power / turns
    avg_power_rand = total_power_rand / turns
    return {
        'avg_score':    avg_score,
        'avg_score_rand' : avg_score_rand,
        'avg_EE':       avg_EE,
        'avg_EE_rand':       avg_EE_rand,
        'avg_power':    avg_power,
        'avg_power_rand' : avg_power_rand,
        'data_rate_lolos' : count_data_ok,
        'data_rate' : data_rate,
        'data_rate_rand' : data_rate_rand,
        'total_rate' : total_rate,
    }

#Just ignore this function~
def str2bool(v):
    '''transfer str to bool for argparse'''
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'True','true','TRUE', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'False','false','FALSE', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
