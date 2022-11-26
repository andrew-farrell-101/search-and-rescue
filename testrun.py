import gym_examples
import gym


# set up to run on main
def main():
    env = gym.make('gym_examples/GridWorld-v0')
    state, *_ = env.reset()
    terminated = False
    while not terminated:
        # take random actions
        state, reward, terminated = env.step(env.action_space.sample())

if __name__ == '__main__':
    main

