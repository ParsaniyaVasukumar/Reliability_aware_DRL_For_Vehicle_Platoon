# from __future__ import division, print_function
# import numpy as np 
# from Environment import *
# import matplotlib.pyplot as plt

# # This py file using the random algorithm.

# def main():
#     up_lanes = [3.5/2,3.5/2 + 3.5,250+3.5/2, 250+3.5+3.5/2, 500+3.5/2, 500+3.5+3.5/2]
#     down_lanes = [250-3.5-3.5/2,250-3.5/2,500-3.5-3.5/2,500-3.5/2,750-3.5-3.5/2,750-3.5/2]
#     left_lanes = [16-2,20-2,24-2]
#     right_lanes = [2,2+4,2+8]
#     width = 7000
#     height = 24 
#    # Initialize a list to hold sum V2I rate data for each number of vehicles
#     sum_v2i_rates = []
    
#     n_values = [20, 40, 60, 80, 100]  # Different numbers of vehicles
#     number_of_game = 50
#     n_step = 100

#     # Create the environment
#     Env = Environ(down_lanes, up_lanes, left_lanes, right_lanes, width, height)

#     # Run simulations for each number of vehicles
#     for n in n_values:
#         V2I_Rate_List = np.zeros([number_of_game, n_step])
#         print(f"Running simulation for {n} vehicles")

#         for game_idx in range(number_of_game):
#             Env.new_random_game(n)
#             for i in range(n_step):
#                 actions = np.random.randint(0, 20, [n, 3])
#                 power_selection = np.zeros(actions.shape, dtype='int')
#                 actions = np.concatenate((actions[..., np.newaxis], power_selection[..., np.newaxis]), axis=2)
#                 reward, _ = Env.act(actions)
#                 V2I_Rate_List[game_idx, i] = np.sum(reward)

#         # Calculate the mean sum V2I rate over all games and steps
#         sum_v2i_rate = np.mean(V2I_Rate_List)
#         sum_v2i_rates.append(sum_v2i_rate)

#     # Plot the results
#     plt.figure(figsize=(8, 6))
#     plt.plot(n_values, sum_v2i_rates, marker='o', linestyle='-', color='b')
#     plt.title('Number of Vehicles vs Sum Rate of V2I Links')
#     plt.xlabel('Number of Vehicles')
#     plt.ylabel('Sum Rate of V2I Links (Mb/s)')
#     plt.grid(True)
#     plt.xlim(10, 110)  # Set the x-axis limits to encompass the range of n_values
#     plt.savefig('Sum Rate of V2I Links vs Number of Vehicles.png')

# if __name__ == "__main__":
#     main()


from __future__ import division, print_function
import numpy as np 
import matplotlib.pyplot as plt
import tensorflow.compat.v1 as tf
from agent import Agent  # Import the Agent class
from Environment import *

def main():
    up_lanes = [3.5/2,3.5/2 + 3.5,250+3.5/2, 250+3.5+3.5/2, 500+3.5/2, 500+3.5+3.5/2]
    down_lanes = [250-3.5-3.5/2,250-3.5/2,500-3.5-3.5/2,500-3.5/2,750-3.5-3.5/2,750-3.5/2]
    left_lanes = [16-2,20-2,24-2]
    right_lanes = [2,2+4,2+8]
    width = 7000
    height = 24 

    # Initialize a list to hold sum V2I rate data for each number of vehicles
    sum_v2i_rates_random = []
    sum_v2i_rates_drl = []

    n_values = [20, 40, 60, 80, 100]  # Different numbers of vehicles
    number_of_game = 40
    n_step = 100

    # Create the environment
    Env = Environ(down_lanes, up_lanes, left_lanes, right_lanes, width, height)

    # Initialize TensorFlow session for the DRL agent
    with tf.Session() as sess:
        agent = Agent(config=[], environment=Env, sess=sess)  # Initialize the agent
        agent.load_weight_from_pkl()  # Load the trained weights

        # Run simulations for each number of vehicles
        for n in n_values:
            V2I_Rate_List_random = np.zeros([number_of_game, n_step])
            V2I_Rate_List_drl = np.zeros([number_of_game, n_step])
            print(f"Running simulation for {n} vehicles")

            for game_idx in range(number_of_game):
                Env.new_random_game(n)

                # Random method
                for i in range(n_step):
                    actions = np.random.randint(0, 20, [n, 3])
                    power_selection = np.zeros(actions.shape, dtype='int')
                    actions = np.concatenate((actions[..., np.newaxis], power_selection[..., np.newaxis]), axis=2)
                    reward, _ = Env.act(actions)
                    V2I_Rate_List_random[game_idx, i] = np.sum(reward)

                # DRL method
                for i in range(n_step):
                    actions = np.zeros((n, 3, 2), dtype='int32')  # Initialize actions for DRL
                    for vehicle_index in range(n):
                        state = agent.get_state([vehicle_index, 0])  # Get state for the vehicle
                        action = agent.predict(state, step=i)  # Use the DRL agent to predict action
                        actions[vehicle_index, 0, 0] = action % agent.RB_number
                        actions[vehicle_index, 0, 1] = int(np.floor(action / agent.RB_number))
                    reward, _ = Env.act(actions)
                    V2I_Rate_List_drl[game_idx, i] = np.sum(reward)

            # Calculate the mean sum V2I rate over all games and steps
            sum_v2i_rate_random = np.mean(V2I_Rate_List_random)
            sum_v2i_rates_random.append(sum_v2i_rate_random)

            sum_v2i_rate_drl = np.mean(V2I_Rate_List_drl)
            sum_v2i_rates_drl.append(sum_v2i_rate_drl)

    # Plot the results
    plt.figure(figsize=(8, 6))
    plt.plot(n_values, sum_v2i_rates_random, marker='o', linestyle='-', color='b', label='Random Method')
    plt.plot(n_values, sum_v2i_rates_drl, marker='s', linestyle='-', color='r', label='DRL Method')
    plt.title('Number of Vehicles vs Sum Rate of V2I Links')
    plt.xlabel('Number of Vehicles') 
    plt.ylabel('Sum Rate of V2I Links (Mb/s)')
    plt.grid(True)
    plt.xlim(10, 110)  # Set the x-axis limits to encompass the range of n_values
    plt.legend()
    plt.savefig('Sum Rate of V2I Links vs Number of Vehicles.png',dpi=300)
    plt.show()

if __name__ == "__main__":
    main()
    