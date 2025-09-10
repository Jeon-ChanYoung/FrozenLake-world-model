from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import uvicorn

from game import reset_env, step_env

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

@app.get("/api/reset")
def api_reset():
    frame_b64 = reset_env()
    return {"frame": frame_b64}

class Action(BaseModel):
    action: int

@app.post("/api/step")
def api_step(action: Action):
    frame_b64 = step_env(action.action)
    return {"frame": frame_b64}

app.mount(
    path="/static",
    app=StaticFiles(directory="static", html=True),
    name="static"
)

@app.get("/")
def root():
    return FileResponse("static/index.html")

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=10000, reload=True)