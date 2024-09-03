import numpy as np

# Parameter
episodes = 1000
alpha = 0.1  # learning rate
gamma = 0.9  # discount factor
epsilon = 0.2 # exploration rate

# Inisialisasi Q-table
q_table = np.zeros((10, 2))  # 10 states, 2 actions (left, right)

# Fungsi untuk memilih aksi
def choose_action(state, q_table, epsilon):
    if np.random.uniform(0, 1) < epsilon:
        return np.random.choice([0, 1])
    else:
        return np.argmax(q_table[state])

# Fungsi untuk menjalankan satu episode
def play_episode(q_table, epsilon, algorithm="q-learning"):
    state = 2
    total_reward = 0
    path = [2]
    while True:
        action = choose_action(state, q_table, epsilon)
        next_state = int(state + (action - 0.5) * 2)
        if(next_state == 9):
            reward = 100   
        elif(next_state == 0):
            reward = -100
        else:
            reward = -1
        total_reward += reward

        # Q-learning
        if(algorithm == "q-learning"):
            q_table[state, action] = q_table[state, action] + alpha * (reward + gamma * np.max(q_table[next_state]) - q_table[state, action])
        # SARSA
        elif(algorithm == "sarsa"):
            next_action = choose_action(next_state, q_table, epsilon)
            q_table[state, action] = q_table[state, action] + alpha * (reward + gamma * q_table[next_state, next_action] - q_table[state, action])

        if(next_state == 0 or next_state == 9):
            state = 3
        else:
            state = next_state
        path.append(state)

        if total_reward >= 500:
            print("Episode Result : Win!")
            break
        elif total_reward <= -200:
            print("Episode Result : Lose!")
            break

    return total_reward, path

# Q-Learning
print("="*10,"Q-LEARNING","="*10)
total_reward = 0
episode = 0
while total_reward < 500:
    total_reward, path = play_episode(q_table, epsilon)
    episode += 1
    if total_reward >= 500:
        print("Agen berhasil mencapai 500 poin pada episode ke-", episode)
        break

# Tampilkan Q-table dan path terakhir
print("Q-table:\n", q_table)
print("Path:", path)
print("Total Path:", len(path))

# SARSA
print("="*10,"SARSA","="*10)
total_reward = 0
episode = 0
while total_reward < 500:
    total_reward, path = play_episode(q_table, epsilon, "sarsa")
    episode += 1
    if total_reward >= 500:
        print("Agen berhasil mencapai 500 poin pada episode ke-", episode)
        break

# Tampilkan Q-table dan path terakhir
print("Q-table:\n", q_table)
print("Path:", path)
print("Total Path:", len(path))