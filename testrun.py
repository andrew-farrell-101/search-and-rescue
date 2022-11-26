import gym_examples
import gym
import numpy as np
import matplotlib.pyplot as plt

# set up to run on main
def main():
    env = gym.make('gym_examples/GridWorld-v0')
    Q, rewards = q_learning(env, 100, 0.1, 1, 0.1, 0.99)
    print(Q.shape)
    print(rewards)


def q_learning(env, num_episodes, alpha, gamma, epsilon, epsilon_decay):
    # initialize Q table
    Q = np.zeros((6, 6, 6, 6, 4, 4))

    # initialize rewards
    rewards = np.zeros(num_episodes)
    # for each episode
    for i in range(num_episodes):
        print(f'Episode {i}')
        # initialize state
        state, _ = env.reset()
        terminated = False

        tsidx = tuple(state.flatten())

        # for each step
        while not terminated:
            # epsilon greedy action selection
            if np.random.rand() < epsilon:
                action = env.action_space.sample()
            else:
                action = np.unravel_index(np.random.choice(np.flatnonzero(np.isclose(Q[tsidx], np.max(Q[tsidx])))), Q[tsidx].shape)    # break ties

            next_state, reward, terminated, *_ = env.step(action)

            tnsidx = tuple(next_state.flatten())
            taidx = tuple(action)
            tsaidx = tsidx + taidx # to fit the correct indexing shape

            Q[tsaidx] += alpha * (reward + gamma * np.max(Q[tnsidx]) - Q[tsaidx])
            state = next_state
            rewards[i] += reward

        # update epsilon
        epsilon = epsilon * epsilon_decay
    return Q, rewards

if __name__ == '__main__':
    main()

