import gymnasium as gym
import torch
import base64
from io import BytesIO
from utils.preprocess import preprocess_128, to_PIL
from model import Encoder_128, Decoder_128, Dynamics

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
current_frame = None

env = gym.make("FrozenLake-v1", render_mode="rgb_array", is_slippery=False)
encoder = Encoder_128().to(device)
decoder = Decoder_128().to(device)
dynamics = Dynamics().to(device)

pth = "frozenlake_worldmodel_128x128_10.pth"

parameters = torch.load(pth)
encoder.load_state_dict(parameters["encoder"])
decoder.load_state_dict(parameters["decoder"])
dynamics.load_state_dict(parameters["dynamics"])

def update_current_frame(frame):
    global current_frame
    if not isinstance(frame, torch.Tensor):
        frame_tensor = preprocess_128(frame)
    else:
        frame_tensor = frame
    current_frame = frame_tensor.detach()
    return current_frame

def tensor_to_base64(tensor_frame):
    pil_img = to_PIL(tensor_frame)
    buf = BytesIO()
    pil_img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")

def reset_env():
    _, _ = env.reset()
    frame = env.render()
    update_current_frame(frame)
    frame_b64 = tensor_to_base64(current_frame)
    return frame_b64

def step_env(action: int):
    global current_frame

    state_tensor = current_frame.unsqueeze(0).to(device)
    action_tensor = torch.tensor([action], device=device)
    latent = encoder(state_tensor)
    latent_next = dynamics(latent, action_tensor)
    next_frame = decoder(latent_next).squeeze(0).cpu()

    next_frame_b64 = tensor_to_base64(next_frame)
    update_current_frame(next_frame)

    return next_frame_b64