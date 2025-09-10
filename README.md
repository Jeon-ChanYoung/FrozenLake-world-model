# FrozenLake-world-model
This is a world model test using the FrozenLake environment. The goal is for the model to predict the next screen based on the user pressing WASD or the arrow keys on the current screen.

<img src="screenshot.png" width="400px" alt="FrozenLake Screenshot">

## Features

- Reinforcement learning environment: [FrozenLake-v1](https://www.gymlibrary.dev/)
- World Model:
  - Encoder (CNN)
  - Dynamics model
  - Decoder (CNN Transpose)
- WASD / Arrow keys control
- FastAPI backend for serving model predictions
- Base64 image streaming for web canvas display

## How to Run
Clone this repository and click the link that appears when you run **main.py**.

## Notes
For smooth learning, the 256x256 size images have been reduced to 64x64. The collect_data-related files are solely for generating training data and are not used for any other purpose.
