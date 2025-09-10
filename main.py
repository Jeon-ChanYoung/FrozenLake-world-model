from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from pydantic import BaseModel
from utils.preprocess import preprocess, to_PIL
import gymnasium as gym
import base64
from PIL import Image
from io import BytesIO
import os
import uvicorn

from model import Encoder

app = FastAPI(
    title = "FrozenLake",
    version="0.1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

env = gym.make("FrozenLake-v1", render_mode = "rgb_array", is_slippery=False)
current_frame = None

def change_current_frame(frame):
    global current_frame
    current_frame = frame

def frame_to_base64(frame):
    tensor_frame = preprocess(frame)
    change_current_frame(tensor_frame)
    img = to_PIL(tensor_frame)
    buf = BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8") 

@app.get("/api/reset")
def reset():
    _, _ = env.reset()
    state_frame = env.render()
    state_frame_b64 = frame_to_base64(state_frame)
    return {"frame": state_frame_b64}

class Action(BaseModel):
    action: int

@app.post("/api/step")
def step(action: Action):
    state_tensor = current_frame.unsqueeze(0)

app.mount(
    path="/static",
    app=StaticFiles(directory="static", html=True),
    name="static"
)

@app.get("/")
def root():
    return FileResponse(os.path.join("static", "index.html"))

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=10000, reload=True)