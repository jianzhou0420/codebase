import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
# Create a tensor containing numbers from 1 to 512

q1 = np.random.randint(0, 40, (4, 4))
q2 = torch.tensor(q1)

q11 = np.sum(q1, axis=0)
q22 = torch.sum(q2)
pass
