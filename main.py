from __future__ import division, print_function
import random
from matplotlib import pyplot as plt
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from agent import Agent
from Environment import *
import seaborn as sns
import numpy as np

# Model configuration
flags = tf.compat.v1.flags
flags.DEFINE_string('model', 'm1', 'Type of model')
flags.DEFINE_boolean('dueling', False, 'Whether to use dueling deep q-network')
flags.DEFINE_boolean('double_q', False, 'Whether to use double q-learning')

# Environment configuration
flags.DEFINE_string('env_name', 'Breakout-v0', 'The name of gym environment to use')
flags.DEFINE_integer('action_repeat', 4, 'The number of actions to be repeated')

# Etc
flags.DEFINE_boolean('use_gpu', True, 'Whether to use gpu or not')
flags.DEFINE_string('gpu_fraction', '1/1', 'idx / # of gpu fraction e.g. 1/3, 2/3, 3/3')
flags.DEFINE_boolean('display', False, 'Whether to display the game screen or not')
flags.DEFINE_boolean('is_train', True, 'Whether to do training or testing')
flags.DEFINE_integer('random_seed', 123, 'Value of random seed')

FLAGS = flags.FLAGS

# Set random seed for reproducibility
tf.random.set_seed(FLAGS.random_seed)
random.seed(FLAGS.random_seed)

if FLAGS.gpu_fraction == '':
    raise ValueError("--gpu_fraction should be defined")

def calc_gpu_fraction(fraction_string):
    idx, num = fraction_string.split('/')
    idx, num = float(idx), float(num)
    fraction = 1 / (num - idx + 1)
    print(" [*] GPU : %.4f" % fraction)
    return fraction

# Set GPU memory growth (Optional but recommended)
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)  # Enables dynamic memory allocation on the GPU
        print("GPU is being used!")
    except RuntimeError as e:
        print(e)
else:
    print("No GPU found, using CPU.")

def main(_):
    up_lanes = [3.5/2,3.5/2 + 3.5,250+3.5/2, 250+3.5+3.5/2, 500+3.5/2, 500+3.5+3.5/2]
    down_lanes = [250-3.5-3.5/2,250-3.5/2,500-3.5-3.5/2,500-3.5/2,750-3.5-3.5/2,750-3.5/2]
    left_lanes = [16-2, 20-2, 24-2]
    right_lanes = [2, 2+4, 2+8]
    width = 10000
    height = 24

    vehicle_counts = [30,40,50]  # The vehicle counts we want to tests
    raw_v2i_rates_all = {n: [] for n in vehicle_counts}  # Dictionary to store rates for each count

    for n_vehicles in vehicle_counts:
        print(f"Running simulation with {n_vehicles} vehicles...")
        Env = Environ(down_lanes, up_lanes, left_lanes, right_lanes, width, height, n_Veh=n_vehicles)  # Create environment with n_Vehicles
        Env.new_random_game()  # Initialize a new random game

        # Set GPU options if using GPU
        gpu_fraction = calc_gpu_fraction(FLAGS.gpu_fraction) if FLAGS.use_gpu else None
        if FLAGS.use_gpu:
            for gpu in gpus:
                tf.config.experimental.set_virtual_device_configuration(
                    gpu, [tf.config.experimental.VirtualDeviceConfiguration(memory_limit= gpu_fraction * 1000)])  # Limit memory usage for each GPU
                print(f"GPU {gpu} memory fraction set to {gpu_fraction}")

        with tf.compat.v1.Session() as sess:
            agent = Agent([], Env, sess)  # Initialize the agent
            agent.train()  # Train the agent
            agent.play()  # Play the game after training
            vehicle = [30,40,50]
            agent.plot_power_vs_vehicle_count(vehicle_counts=vehicle)
            # Collect the raw V2I rates
            raw_v2i_rates_all[n_vehicles] = agent.raw_v2i_rates_over_time

    # Now we plot the raw V2I rates for each vehicle count
    plt.figure(figsize=(10, 8))
    # Define the maximum x value you want to plot
    max_x_value = 8000
    max_interval_index = max_x_value // 500  # Calculate how many intervals fit within the max x value

    for n_vehicles, raw_v2i_rates in raw_v2i_rates_all.items():
        # Calculate the mean raw V2I rates over intervals of 250 steps
        mean_raw_v2i_rates = []
        for i in range(0, len(raw_v2i_rates), 500):
            interval_data = raw_v2i_rates[i:i+500]
            if interval_data:  # Check if the interval has data
                mean_raw_v2i_rates.append(np.mean(interval_data))

        # Create x values for plotting
        x_values = np.arange(len(mean_raw_v2i_rates)) * 500 + (500 / 2)  # Midpoint of each interval
        plt.plot(x_values[:max_interval_index], mean_raw_v2i_rates[:max_interval_index], label=f'{n_vehicles} Vehicles')

    plt.xlabel('Time Step', fontsize=10)
    plt.ylabel('Mean Raw V2I Rate (bps)', fontsize=10)
    plt.title('Mean Raw V2I Rate vs Time for Different Vehicle Counts', fontsize=16)
    plt.xlim(0, 8100)  # Limit x-axis to 1000 steps
    plt.ylim(0, 100)
    plt.legend(loc="best")
    plt.grid(True)
    plt.savefig('mean_raw_v2i_rate_vs_time_multiple_vehicles.png', dpi=300)

if __name__ == '__main__':
    tf.compat.v1.app.run()
