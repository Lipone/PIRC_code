import numpy as np
import matplotlib.pyplot as plt
import pickle
import os

def generate_rossler_data(t0, tf, dt, initial_state, a=0.2, b=0.2, c=5.7):
    def rossler(t, state):
        x, y, z = state
        dxdt = -y - z
        dydt = x + a * y
        dzdt = b + z * (x - c)
        return np.array([dxdt, dydt, dzdt])

    t_values = np.arange(t0, tf, dt)
    states = np.zeros((len(t_values), len(initial_state)))
    states[0] = initial_state
    for i in range(1, len(t_values)):
        t = t_values[i-1]
        state = states[i-1]
        k1 = rossler(t, state)
        k2 = rossler(t + 0.5*dt, state + 0.5*dt*k1)
        k3 = rossler(t + 0.5*dt, state + 0.5*dt*k2)
        k4 = rossler(t + dt, state + dt*k3)
        states[i] = state + (dt/6)*(k1 + 2*k2 + 2*k3 + k4)
    return t_values, states


t, states = generate_rossler_data(0, 1000, 0.1, np.array([1.0, 1.0, 1.0]))

os.makedirs('data', exist_ok=True)
with open('data/rossler_data.pkl', 'wb') as f:
    pickle.dump((t, states), f)

# 绘制三维图像
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.plot(states[:, 0], states[:, 1], states[:, 2])
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.set_title('Rössler Attractor (3D)')
plt.show()
