# 데이터 생성기. 무작위 에피소드를 저장함. 데이터가 있다면 할 필요없음
import gymnasium as gym
import pickle

env = gym.make("FrozenLake-v1", render_mode = "rgb_array", is_slippery=False)
state, _ = env.reset()
dataset = []
episodes = 1000

for episode in range(1, episodes + 1):
    state, _ = env.reset()
    done = False

    while not done:
        current_frame = env.render()
        action = env.action_space.sample()
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        next_frame = env.render()

        dataset.append({
            "state_frame": current_frame,
            "action": action,
            "next_frame": next_frame,
            "reward": reward,
            "done": done
        })

    if episode % 50 == 0:
        print(f"Collected {episode} episodes")

with open("frozenlake_dataset.pkl", "wb") as f:
    pickle.dump(dataset, f)