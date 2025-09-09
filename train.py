from utils.frozenLakeDataset import FrozenLakeDataset
from model import Encoder, Decoder, Dynamics
from torch.utils.data import DataLoader
import torch
import torch.nn as nn

import gymnasium as gym
import pickle

env = gym.make("FrozenLake-v1", render_mode = "rgb_array", is_slippery=False)
state, _ = env.reset()

with open("frozenlake_dataset.pkl", "rb") as f:
    dataset_raw = pickle.load(f)

dataset = FrozenLakeDataset(dataset_raw)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

latent_dim = 32
encoder = Encoder(latent_dim)
decoder = Decoder(latent_dim)
dynamics = Dynamics(latent_dim)
criterion = nn.MSELoss()

optimizer = torch.optim.Adam(list(encoder.parameters()) +
                             list(decoder.parameters()) +
                             list(dynamics.parameters()), lr=1e-3)
# train
epochs = 30
for epoch in range(1, epochs + 1):
    total_loss = 0
    for state, action, next_state in loader:
        latent = encoder(state)
        latent_next_pred = dynamics(latent, action)
        next_state_pred = decoder(latent_next_pred)

        loss = criterion(next_state_pred, next_state)
        optimizer.zero_grad()
        loss.backward()
        
        optimizer.step()
        total_loss += loss.item()

    print(f"Epoch {epoch}/{epochs} - Loss: {total_loss/len(loader):.6f}")

"""
Epoch 1/30 - Loss: 0.012908
Epoch 2/30 - Loss: 0.002881
Epoch 3/30 - Loss: 0.002769
Epoch 4/30 - Loss: 0.002721
Epoch 5/30 - Loss: 0.002610
Epoch 6/30 - Loss: 0.002370
Epoch 7/30 - Loss: 0.001242
Epoch 8/30 - Loss: 0.000692
Epoch 9/30 - Loss: 0.000447
Epoch 10/30 - Loss: 0.000321
Epoch 11/30 - Loss: 0.000215
Epoch 12/30 - Loss: 0.000134
Epoch 13/30 - Loss: 0.000119
Epoch 14/30 - Loss: 0.000104
Epoch 15/30 - Loss: 0.000084
Epoch 16/30 - Loss: 0.000066
Epoch 17/30 - Loss: 0.000063
Epoch 18/30 - Loss: 0.000044
Epoch 19/30 - Loss: 0.000044
Epoch 20/30 - Loss: 0.000035
Epoch 21/30 - Loss: 0.000024
Epoch 22/30 - Loss: 0.000024
Epoch 23/30 - Loss: 0.000023
Epoch 24/30 - Loss: 0.000065
Epoch 25/30 - Loss: 0.000017
Epoch 26/30 - Loss: 0.000017
Epoch 27/30 - Loss: 0.000016
Epoch 28/30 - Loss: 0.000015
Epoch 29/30 - Loss: 0.000013
Epoch 30/30 - Loss: 0.000015
-> frozenlake_worldmodel.pth
"""
torch.save({
    'encoder': encoder.state_dict(),
    'dynamics': dynamics.state_dict(),
    'decoder': decoder.state_dict()
}, "frozenlake_worldmodel.pth")

