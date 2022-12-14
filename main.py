import gym
import gym_examples
import numpy as np
import matplotlib.pyplot as plt

# set up to run on main
def main():
    env = gym.make('gym_examples/GridWorld-v0')
    rewards = q_learning(env, num_runs=50, num_episodes=1000, alpha= 0.1, gamma=0.99, epsilon=0.1)
    rewards = np.mean(rewards, axis=0)
    plt.plot(rewards)
    plt.title("Q-Learning - Sum of rewards vs Episodes")
    plt.xlabel("Episode")
    plt.ylabel("Sum of Rewards")
    plt.grid()
    plt.show()


def q_learning(env, num_runs=50, num_episodes=50, alpha=0.01, gamma=0.95, epsilon=0.1):
    # initialize rewards
    rewards = np.zeros((num_runs, num_episodes))
    nsteps = np.zeros((num_runs, num_episodes))
    for run in range(num_runs):
        run_epsilon = epsilon
        print(f'Run {run + 1}')
        # initialize Q table
        Q = np.zeros((6, 6, 6, 6, 4, 4))
        # for each episode
        for episode in range(num_episodes):
            # initialize state
            state, _ = env.reset()
            terminated = False

            S_idx = tuple(state.flatten())

            # for each step
            while not terminated:
                # epsilon greedy action selection
                if np.random.rand() < run_epsilon:
                    action = env.action_space.sample()
                else:
                    action = np.unravel_index(np.random.choice(np.flatnonzero(np.isclose(Q[S_idx], np.max(Q[S_idx])))), Q[S_idx].shape)    # break ties

                next_state, reward, terminated, *_ = env.step(action)
                nsteps[run, episode] += 1

                NS_idx = tuple(next_state.flatten())
                A_idx = tuple(action)
                SA_idx = S_idx + A_idx # to fit the correct indexing shape

                Q[SA_idx] += alpha * (reward + gamma * np.max(Q[NS_idx]) - Q[SA_idx])
                S_idx = NS_idx
                rewards[run][episode] += reward
            # update epsilon
            run_epsilon *= 1 / np.sum(nsteps[run])
    return rewards

if __name__ == '__main__':
    main()