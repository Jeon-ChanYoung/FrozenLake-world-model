from torch.utils.data import Dataset
from preprocess import preprocess
import torch 

class FrozenLakeDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        state = self.data[index]["state_frame"] # (256, 256, 3)
        state = preprocess(state)               
        state = state.clone().detach().float()
        next_state = self.data[index]["next_frame"]
        next_state = preprocess(next_state)
        next_state = next_state.clone().detach().float()
        action = torch.tensor(self.data[index]["action"])
        return state, action, next_state