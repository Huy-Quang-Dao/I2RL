import gymnasium as gym
import numpy as np

# Tạo môi trường
env = gym.make("Pendulum-v1", render_mode="human")  # "human" để hiển thị giao diện trực quan
state, _ = env.reset()
print(state)

for _ in range(1):  # Chạy 200 bước
    action = np.array([env.action_space.sample()])  # Chọn hành động ngẫu nhiên
    print(action)
    next_state, reward, done, truncated, _ = env.step(action)
    print(reward)
    print(next_state)
    env.render()  # Hiển thị giao diện
    
    if done or truncated:
        break  # Dừng khi episode kết thúc

env.close()
