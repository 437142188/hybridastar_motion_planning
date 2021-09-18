import numpy as np
import os
import math

import torch
from torch.utils.data import Dataset


class plan_dataset(Dataset):
    def __init__(self, list_IDs, path):
        self.list_IDs = list_IDs
        self.path = path
        self.width = 5.0

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        ID = self.list_IDs[index]
        states = np.load(os.path.join(self.path, 'states' + str(ID) + '.npy'))
        plan = np.load(os.path.join(self.path, 'plan' + str(ID) + '.npy'))
        states = torch.from_numpy(states)
        plan = torch.from_numpy(plan)

        #normalize
        states[0] = states[0] / self.width
        states[1] = states[1] / self.width
        states[2] = states[2] / math.pi
        states[3] = states[3] / self.width
        states[4] = states[4] / self.width
        states[5] = states[5] / math.pi

        plan[0] = plan[0] / self.width
        plan[1] = plan[1] / self.width
        plan[2] = plan[2] / math.pi

        return states, plan
