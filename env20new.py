import numpy as np
from scipy.spatial.distance import cdist
from typing import Optional

class GameEnv:
    def __init__(self, nodes, p_max, area_size=(20, 20)):
        self.nodes = nodes
        self.p_max = p_max
        self.area_size = area_size
        self.noise_power = 2e-10
        self.Rmin = 0.15
        self.rng = np.random.default_rng()
        self.positions = None
        self.channel_gain = None

        self.observation_space = 2 * nodes * nodes + 2 * nodes + 1 + 1  # gain, gain_prev, power_prev, data_rate_prev, Rmin, EE_prev  
        self.action_space = nodes

    def reset(self, seed=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self.positions = self.generate_positions()
        self.channel_gain = self.generate_channel_gain(self.positions)

        power_prev = self.sample_valid_power()
        interference = self.compute_interference(power_prev, self.channel_gain)
        data_rate = self.compute_data_rate(self.channel_gain, interference, power_prev)
        EE_prev = self.compute_energy_efficiency(power_prev, data_rate)

        state = np.concatenate([
            self.norm(self.channel_gain).flatten(),
            self.norm(self.channel_gain_prev).flatten(),
            self.norm(power_prev),
            [self.Rmin],
            self.norm(data_rate),
            [EE_prev]
        ])

        self.power_prev = power_prev
        self.channel_gain_prev = self.channel_gain.copy()
        self.data_rate_prev = data_rate
        self.EE_prev = EE_prev

        return state.astype(np.float32), {}

    def step(self, action):
        power = action
        interference = self.compute_interference(power, self.channel_gain)
        sinr = self.compute_sinr(self.channel_gain, interference, power)
        data_rate = self.compute_data_rate_from_sinr(sinr)
        EE = self.compute_energy_efficiency(power, data_rate)

        total_power = np.sum(power)
        total_rate = np.sum(data_rate)
        rate_violation = np.sum(np.maximum(self.Rmin - data_rate, 0.0))

        reward = 10 * np.log(EE + 1e-10) - rate_violation - max(0.0, total_power - self.p_max)

        dw = bool(total_power > self.p_max)

        state = np.concatenate([
            self.norm(self.channel_gain).flatten(),
            self.norm(self.channel_gain_prev).flatten(),
            self.norm(power),
            self.norm(self.data_rate_prev),
            [self.Rmin],
            [EE]
        ])

        info = {
            "EE": EE,
            "data_rate": data_rate,
            "data_rate_pass": sum(data_rate >= self.Rmin),
            "total_power": total_power,
            "total_rate": total_rate
        }

        return state.astype(np.float32), float(reward), dw, False, info

    def sample_valid_power(self):
        rand = self.rng.random(self.nodes)
        rand /= rand.sum()
        return rand * self.p_max

    def generate_positions(self, minDistance=2, subnet_radius=2, minD=0.5):
        bound = self.area_size[0] - 2 * subnet_radius
        X, Y = np.zeros((self.nodes, 1)), np.zeros((self.nodes, 1))
        dist_2 = minDistance ** 2
        nValid = 0
        while nValid < self.nodes:
            newX = bound * (self.rng.uniform() - 0.5)
            newY = bound * (self.rng.uniform() - 0.5)
            if all(((X[:nValid] - newX)**2 + (Y[:nValid] - newY)**2).flatten() > dist_2):
                X[nValid] = newX
                Y[nValid] = newY
                nValid += 1
        X += self.area_size[0] / 2
        Y += self.area_size[1] / 2
        gwLoc = np.concatenate((X, Y), axis=1)
        dist_rand = self.rng.uniform(minD, subnet_radius, (self.nodes, 1))
        angN = self.rng.uniform(0, 2 * np.pi, (self.nodes, 1))
        D_XLoc = X + dist_rand * np.cos(angN)
        D_YLoc = Y + dist_rand * np.sin(angN)
        dvLoc = np.concatenate((D_XLoc, D_YLoc), axis=1)
        return cdist(gwLoc, dvLoc)

    def generate_channel_gain(self, dist, sigmaS=7.0, transmit_power=1.0, lambdA=0.05, plExponent=2.7):
        S = sigmaS * self.rng.standard_normal((self.nodes, self.nodes))
        S_linear = 10 ** (S / 10)
        real, imag = self.rng.standard_normal((self.nodes, self.nodes)), self.rng.standard_normal((self.nodes, self.nodes))
        h = (1 / np.sqrt(2)) * (real + 1j * imag)
        H_power = transmit_power * (4 * np.pi / lambdA) ** (-2) * dist ** (-plExponent) * S_linear * np.abs(h) ** 2
        return H_power

    def compute_interference(self, power, gain):
        interferensi = np.zeros((self.nodes, self.nodes))
        for i in range(self.nodes):
            for j in range(self.nodes):
                if i != j:
                    interferensi[i][j] = gain[i][j] * power[i]
        return interferensi

    def compute_sinr(self, gain, interference, power):
        sinr = np.zeros(self.nodes)
        for i in range(self.nodes):
            num = gain[i][i] * power[i]
            denom = self.noise_power + np.sum([interference[i][j] for j in range(self.nodes) if j != i])
            sinr[i] = num / denom
        return sinr

    def compute_data_rate_from_sinr(self, sinr):
        return np.log(1 + np.maximum(sinr, 0))

    def compute_data_rate(self, gain, interference, power):
        sinr = self.compute_sinr(gain, interference, power)
        return self.compute_data_rate_from_sinr(sinr)

    def compute_energy_efficiency(self, power, data_rate):
        total_power = np.sum(power)
        total_rate = np.sum(data_rate)
        return total_rate / total_power if total_power > 0 else 0

    def norm(self, x):
        x = np.maximum(x, 1e-10)
        x_log = np.log10(x + 1e-10)
        return (x_log - x_log.min()) / (x_log.max() - x_log.min() + 1e-10)
