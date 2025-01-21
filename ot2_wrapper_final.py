from math import inf
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from sim_class import Simulation


class OT2Env(gym.Env):
    def __init__(self, render=False):
        super(OT2Env, self).__init__()
        self.render = render
        self.total_distance = 0
        self.initial_distance = 0

        # Create the simulation environment
        self.sim = Simulation(num_agents=1, render=render)

        # Define action and observation space
        # They must be gym.spaces objects
        self.action_space = spaces.Box(low=np.array([-1, -1, -1]), high=np.array([1, 1, 1]), dtype=np.float32, shape=(3,))
        self.observation_space = spaces.Box(
                                    low=np.array([-inf] * 9),  # 3 (pipette position) + 3 (goal position) + 3 (relative vector)
                                    high=np.array([inf] * 9),
                                    dtype=np.float32,
                                    shape=(9,)
                                )

        # keep track of the number of steps
        self.steps = 0

    def reset(self, seed=None):
        # being able to set a seed is required for reproducibility
        if seed is not None:
            np.random.seed(seed)

        # Reset the state of the environment to an initial state
        # set a random goal position for the agent, consisting of x, y, and z coordinates within the working area (you determined these values in the previous datalab task)
        rand_x = round(np.random.uniform(-0.1871, 0.2532), 4)
        rand_y = round(np.random.uniform(-0.1707, 0.2197), 4)
        rand_z = round(np.random.uniform(0.1195, 0.2896), 4)

        #print(f"Goal position: {rand_x}, {rand_y}, {rand_z}")

        self.goal_position = np.array([rand_x, rand_y, rand_z])

        # Call the environment reset function
        observation = self.sim.reset(num_agents=1)

        pipette_pos = observation[f'robotId_{self.sim.robotIds[-1]}']['pipette_position']
        
        # Update the observation in reset and step methods
        relative_position = self.goal_position - pipette_pos

        self.init_distance = np.linalg.norm(relative_position)

        observation = np.concatenate([pipette_pos, self.goal_position, relative_position], dtype=np.float32)

        self.prev_pipette = observation[:3]

        # Reset the number of steps
        self.steps = 0
        self.total_distance = 0

        info = {} # we don't need to return any additional information

        return observation, info

    def step(self, action):
        action = np.append(action, 0)

        # Call the environment step function
        observation = self.sim.run([action]) # Why do we need to pass the action as a list? Think about the simulation class.

        pipette_pos = observation[f'robotId_{self.sim.robotIds[-1]}']['pipette_position']
        relative_position = self.goal_position - pipette_pos
        observation = np.concatenate([pipette_pos, self.goal_position, relative_position], dtype=np.float32)

        pipette_pos = observation[:3]

        distance = np.linalg.norm(relative_position)
        # Calculate alignment reward
        direction_to_goal = relative_position / (distance + 1e-9)  # Normalize direction vector
        movement_direction = action[:3] / (np.linalg.norm(action[:3]) + 1e-9)  # Normalize action vector, avoid divide by zero
        alignment = np.dot(direction_to_goal, movement_direction)  # Cosine similarity for directional alignment

        '''
            Extremely important to note is that maximum distance from
            pipette_offset [0.073, 0.0895, 0.0895] point to any point 
            on the cube is around 0.4186. I will round it to 0.42
        '''

        # Add movement total reward
        self.total_distance += np.linalg.norm(self.prev_pipette-pipette_pos)
        self.prev_pipette = pipette_pos

        # Initialize reward with alignment term (scaled)
        reward_alignment = (alignment+1)/2  # Scale alignment reward to emphasize directional (values from 0 to 1) 

        # Add distance-based reward
        reward_distance = -2 * distance/self.init_distance  # Higher reward closer to the goal from -2 to 0

        reward_bonus1 = 100 * max(0, 0.006 - distance) \
            if distance <= 0.006 else 0  # values from 0 to 0.6

        reward_bonus2 = 300 * max(0, 0.003 - distance) \
            if distance <= 0.003 else 0  # values from 0 to 0.9
        
        step_penalty = -0.05

        reward = reward_alignment + reward_distance \
            + reward_bonus1 + reward_bonus2 + step_penalty

        # increment the number of steps
        self.steps += 1
        
        # next we need to check if the if the task has been completed and if the episode should be terminated
        # To do this we need to calculate the distance between the pipette position and the goal position and if it is below a certain threshold, we will consider the task complete. 
        # What is a reasonable threshold? Think about the size of the pipette tip and the size of the plants.
        if distance < 0.001:
            reward_goal = 250 - (self.steps)/5  # values from 250 to 50
            reward_movement = (self.init_distance - self.total_distance) / self.init_distance * 3
            reward += reward_goal + reward_movement

            print(f'Goal reached in {self.steps} steps')
            print('Initial distance:', self.init_distance)
            print(f'Total distance covered: {self.total_distance}')

            terminated = True 
        else:
            terminated = False

        # next we need to check if the episode should be truncated, we can check if the current number of steps is greater than the maximum number of steps
        if self.steps >= 1000:
            truncated = True
        else:
            truncated = False

        # in info put a dictionary with seprate rewards
        info = {
            'reward_alignment': reward_alignment,
            'reward_distance': reward_distance,
            'reward_bonus1': reward_bonus1,
            'reward_bonus2': reward_bonus2,
            'reward_step_penalty': step_penalty,
            'reward_movement': reward_movement if terminated else 0,
            'reward_goal': reward_goal if terminated else 0,
            'reward_total': reward
        }

        return observation, reward, terminated, truncated, info

    def render(self, mode='human'):
        pass

    def close(self):
        self.sim.close()
