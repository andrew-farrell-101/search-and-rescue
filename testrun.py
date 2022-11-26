import gym_examples
import gym
import numpy as np
import matplotlib.pyplot as plt

# set up to run on main
def main():
    env = gym.make('gym_examples/GridWorld-v0')
    Q, rewards = q_learning(env, 100, 0.1, 1, 0.1, 0.99)
    # Q appears to be only updating state (0, 0, 5, 5)
    print(Q.shape)
    print(rewards)


def q_learning(env, num_episodes, alpha, gamma, epsilon, epsilon_decay):
    # initialize Q table
    Q = np.zeros((6, 6, 6, 6, 4, 4))

    # initialize rewards
    rewards = np.zeros(num_episodes)
    # for each episode
    for i in range(num_episodes):
        print(f'Episode {i + 1}')
        # initialize state
        state, _ = env.reset()
        terminated = False

        S_idx = tuple(state.flatten())

        # for each step
        while not terminated:
            # epsilon greedy action selection
            if np.random.rand() < epsilon:
                action = env.action_space.sample()
            else:
                action = np.unravel_index(np.random.choice(np.flatnonzero(np.isclose(Q[S_idx], np.max(Q[S_idx])))), Q[S_idx].shape)    # break ties

            next_state, reward, terminated, *_ = env.step(action)

            NS_idx = tuple(next_state.flatten())
            A_idx = tuple(action)
            tsaidx = S_idx + A_idx # to fit the correct indexing shape

            Q[tsaidx] += alpha * (reward + gamma * np.max(Q[NS_idx]) - Q[tsaidx])
            S_idx = NS_idx
            rewards[i] += reward

        # update epsilon
        epsilon = epsilon * epsilon_decay
    return Q, rewards

if __name__ == '__main__':
    main()

