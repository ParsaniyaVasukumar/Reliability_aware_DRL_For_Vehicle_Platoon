from __future__ import print_function, division
import os
import time
import random
from tkinter import image_names
import numpy as np
from Environment import *
from base import BaseModel
from replay_memory import ReplayMemory
from utils import save_pkl, load_pkl
import tensorflow.compat.v1 as tf
import matplotlib.pyplot as plt
tf.compat.v1.disable_v2_behavior()
from sklearn.metrics import confusion_matrix
import seaborn as sns

class Agent(BaseModel):
    
    def __init__(self, config, environment, sess):
        self.sess = sess
        self.weight_dir = 'weight'        
        self.env = environment
        # self.history = History(self.config)
        model_dir = './Model/a.model'
        self.memory = ReplayMemory(model_dir) 
        self.max_step = 2000
        self.RB_number = 20
        self.num_vehicle = len(self.env.vehicles)
        print('-------------------------------------------')
        print(self.num_vehicle)
        print('-------------------------------------------')
        self.action_all_with_power = np.zeros([self.num_vehicle, 3, 2],dtype = 'int32')   # this is actions that taken by V2V links with power
        self.action_all_with_power_training = np.zeros([20, 3, 2],dtype = 'int32')   # this is actions that taken by V2V links with power
        self.reward = []
        self.learning_rate = 0.01
        self.learning_rate_minimum = 0.0001
        self.learning_rate_decay = 0.96
        self.learning_rate_decay_step = 500000
        self.target_q_update_step = 100
        self.discount = 0.5
        self.double_q = True
        print("------------")
        print(self.double_q)
        print("------------")
        self.build_dqn()          
        self.V2V_number = 3 * len(self.env.vehicles)    # every vehicle need to communicate with 3 neighbors  
        self.training = True
        #self.actions_all = np.zeros([len(self.env.vehicles),3], dtype = 'int32')
        self.delay_satisfaction_list = []  # 1To store delay satisfaction probabilities
        self.num_vehicles_list = []        # To store the number of vehicles
        self.platoon_size_list = []        # To store the platoon sizes
        self.power_selection_vs_platoon = []  # To store power selection probabilities
        self.V2V_power_dB_List = self.env.V2V_power_dB_List  # Add this line
        # self.environment = environment  # Instance of the Environ class
        self.raw_v2i_rates_over_time = []  # New list to store raw V2I rates

        self.num_vehicle = len(self.env.vehicles)
        self.initialize_action_arrays()  # Initialize action arrays
        self.predicted_v2i_rates_over_time = []  # Add this line to initialize the list

       
    def merge_action(self, idx, action):
        
        self.action_all_with_power[idx[0], idx[1], 0] = action % self.RB_number
        self.action_all_with_power[idx[0], idx[1], 1] = int(np.floor(action/self.RB_number))

    def get_state(self, idx):
        # ===============
        #  Get State from the environment
        # =============
        vehicle_number = len(self.env.vehicles)
        V2V_channel = (self.env.V2V_channels_with_fastfading[idx[0], self.env.vehicles[idx[0]].destinations[idx[1]], :] - 80) / 60
        V2I_channel = (self.env.V2I_channels_with_fastfading[idx[0], :] - 80) / 60
        V2V_interference = (-self.env.V2V_Interference_all[idx[0], idx[1], :] - 60) / 60
        NeiSelection = np.zeros(self.RB_number)

        for i in range(3):
            if idx[0] < len(self.env.vehicles):
                neighbor_idx = self.env.vehicles[idx[0]].neighbors[i]
                if 0 <= neighbor_idx < len(self.env.vehicles):  # Ensure neighbor index is valid
                    if self.training:
                        action_index = min(neighbor_idx, self.action_all_with_power_training.shape[0] - 1)
                        action_index = self.action_all_with_power_training[action_index, idx[1], 0]                        
                        if 0 <= action_index < self.RB_number:  # Ensure action index is valid
                            NeiSelection[action_index] = 1
                    else:
                        action_index = self.action_all_with_power[neighbor_idx, idx[1], 0]
                        action_index = min(action_index, self.RB_number - 1)  # Ensure action index is valid
                        NeiSelection[action_index] = 1

        for i in range(3):
            if i == idx[1]:
                continue
            if idx[0] < len(self.env.vehicles):
                if idx[0] < self.action_all_with_power_training.shape[0]:  # Check if idx[0] is within bounds
                    if self.training:
                        if self.action_all_with_power_training[idx[0], i, 0] >= 0:
                            action_index = self.action_all_with_power_training[idx[0], i, 0]
                            if 0 <= action_index < self.RB_number:  # Ensure action index is valid
                                NeiSelection[action_index] = 1
                    else:
                        if self.action_all_with_power[idx[0], i, 0] >= 0:
                            action_index = self.action_all_with_power[idx[0], i, 0]
                            action_index = min(action_index, self.RB_number - 1)  # Ensure action index is valid
                            NeiSelection[action_index] = 1

        time_remaining = np.asarray([self.env.demand[idx[0], idx[1]] / self.env.demand_amount])
        load_remaining = np.asarray([self.env.individual_time_limit[idx[0], idx[1]] / self.env.V2V_limit])
        return np.concatenate((V2I_channel, V2V_interference, V2V_channel, NeiSelection, time_remaining, load_remaining))

    def predict(self, s_t,  step, test_ep = False):
        # ==========================
        #  Select actions
        # ======================
        ep = 1/(step/1000000 + 1)
        if random.random() < ep and test_ep == False:   # epsion to balance the exporation and exploition
            action = np.random.randint(60)
        else:          
            action =  self.q_action.eval({self.s_t:[s_t]})[0] 
        return action
    def observe(self, prestate, state, reward, action):
        # -----------
        # Collect Data for Training 
        # ---------
        self.memory.add(prestate, state, reward, action) # add the state and the action and the reward to the memory
        #print(self.step)
        if self.step > 0:
            if self.step % 50 == 0:
                #print('Training')
                self.q_learning_mini_batch()            # training a mini batch
                self.save_weight_to_pkl()
            if self.step % self.target_q_update_step == self.target_q_update_step - 1:
                #print("Update Target Q network:")
                self.update_target_q_network()           # ?? what is the meaning ??
    def train(self):        
        num_game, self.update_count, ep_reward = 0, 0, 0.
        total_reward, self.total_loss, self.total_q = 0.,0.,0.
        max_avg_ep_reward = 0
        ep_reward, actions = [], []        
        mean_big = 0
        number_big = 0
        mean_not_big = 0
        number_not_big = 0
        v2i_rates_over_time = []
        time_steps = []

        self.env.new_random_game(20)
        for self.step in (range(0, 8000)): # need more configuration
            if self.step == 0:                   # initialize set some varibles
                num_game, self.update_count,ep_reward = 0, 0, 0.
                total_reward, self.total_loss, self.total_q = 0., 0., 0.
                ep_reward, actions = [], []    

            step_rewards = []  # Collect all rewards for the current step
            # prediction
            # action = self.predict(self.history.get())
            if (self.step % 8000 == 1):
                self.env.new_random_game(20)
            print(self.step)
            state_old = self.get_state([0,0])
            
            #print("state", state_old)
            self.training = True
            for k in range(1):
                for i in range(len(self.env.vehicles)):              
                    for j in range(3): 
                        state_old = self.get_state([i,j]) 
                        action = self.predict(state_old, self.step)                    
                        #self.merge_action([i,j], action)   
                        self.action_all_with_power_training[i, j, 0] = action % self.RB_number
                        self.action_all_with_power_training[i, j, 1] = int(np.floor(action/self.RB_number))                                                      
                        # Collect true and predicted labels
                        reward_train, raw_V2I_rate, predicted_V2I_rate = self.env.act_for_training(self.action_all_with_power_training, [i, j])
                        state_new = self.get_state([i,j]) 
                        self.observe(state_old, state_new, reward_train, action)
                        # Store raw V2I rate
                        self.raw_v2i_rates_over_time.append(raw_V2I_rate)
                        step_rewards.append(reward_train)  # Collect each reward

            # Calculate average reward for the current timestep and store
            avg_reward = np.mean(step_rewards) if step_rewards else 0
            v2i_rates_over_time.append(avg_reward)
            time_steps.append(self.step)

            if (self.step % 8000 == 0) and (self.step > 0):
                # testing 
                self.training = False
                number_of_game = 40
                if (self.step % 10000 == 0) and (self.step > 0):
                    number_of_game = 50 
                if (self.step == 38000):
                    number_of_game = 100               
                V2I_Rate_list = np.zeros(number_of_game)
                Fail_percent_list = np.zeros(number_of_game)
                for game_idx in range(number_of_game):
                    self.env.new_random_game(self.num_vehicle)
                    test_sample = 200
                    Rate_list = []
                    print('test game idx:', game_idx)
                    for k in range(test_sample):
                        action_temp = self.action_all_with_power.copy()
                        for i in range(len(self.env.vehicles)):
                            self.action_all_with_power[i,:,0] = -1
                            sorted_idx = np.argsort(self.env.individual_time_limit[i,:])          
                            for j in sorted_idx:                   
                                state_old = self.get_state([i,j])
                                action = self.predict(state_old, self.step, True)
                                self.merge_action([i,j], action)
                            if i % (len(self.env.vehicles)/10) == 1:
                                action_temp = self.action_all_with_power.copy()
                                reward, percent = self.env.act_asyn(action_temp) #self.action_all)            
                                Rate_list.append(np.sum(reward))
                        #print("actions", self.action_all_with_power)
                    V2I_Rate_list[game_idx] = np.mean(np.asarray(Rate_list))
                    Fail_percent_list[game_idx] = percent
                    #print("action is", self.action_all_with_power)
                    print('failure probability is, ', percent)
                    #print('action is that', action_temp[0,:])
            #print("OUT")
                self.save_weight_to_pkl()
                print ('The number of vehicle is ', len(self.env.vehicles))
                print ('Mean of the V2I rate is that ', np.mean(V2I_Rate_list))
                print('Mean of Fail percent is that ', np.mean(Fail_percent_list)) 
                print(state_old , "this is state old")                  
                #print('Test Reward is ', np.mean(test_result))
                print(state_old , self.step )  

        # Plot the collected data after all steps
        plt.figure(figsize=(10, 6))
        plt.plot(time_steps, v2i_rates_over_time, label='Average Reward', color='blue', linestyle='-')
        plt.xlabel('Time Step')
        plt.ylabel('Average Reward')
        plt.title('Average Reward vs. Time Step')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        plt.tight_layout()
        plt.savefig('reward_vs_time_step_all80.png', dpi=300)
        plt.close()
        
        # After training, plot and save the graphs
        # self.plot_v2i_rate_vs_time(v2i_rates_over_time, time_steps)
        self.plot_raw_v2i_rate_vs_time(self.raw_v2i_rates_over_time, num_steps=8000, interval=500)
                  
    def q_learning_mini_batch(self):

        # Training the DQN model
        # ------ 
        #s_t, action,reward, s_t_plus_1, terminal = self.memory.sample() 
        s_t, s_t_plus_1, action, reward = self.memory.sample()  
        #print() 
        #print('samples:', s_t[0:10], s_t_plus_1[0:10], action[0:10], reward[0:10])        
        t = time.time()        
        if self.double_q:       #double Q learning   
            pred_action = self.q_action.eval({self.s_t: s_t_plus_1})       
            q_t_plus_1_with_pred_action = self.target_q_with_idx.eval({self.target_s_t: s_t_plus_1, self.target_q_idx: [[idx, pred_a] for idx, pred_a in enumerate(pred_action)]})            
            target_q_t =  self.discount * q_t_plus_1_with_pred_action + reward
        else:
            q_t_plus_1 = self.target_q.eval({self.target_s_t: s_t_plus_1})         
            max_q_t_plus_1 = np.max(q_t_plus_1, axis=1)
            target_q_t = self.discount * max_q_t_plus_1 +reward
        _, q_t, loss,w = self.sess.run([self.optim, self.q, self.loss, self.w], {self.target_q_t: target_q_t, self.action:action, self.s_t:s_t, self.learning_rate_step: self.step}) # training the network
        
        print('loss is ', loss)
        self.total_loss += loss
        self.total_q += q_t.mean()
        self.update_count += 1
            

    def build_dqn(self): 
    # --- Building the DQN -------
        self.w = {}
        self.t_w = {}        
        
        initializer = tf. truncated_normal_initializer(0, 0.02)
        activation_fn = tf.nn.relu
        n_hidden_1 = 500
        n_hidden_2 = 250
        n_hidden_3 = 120
        n_input = 82
        n_output = 60
        def encoder(x):
            weights = {                    
                'encoder_h1': tf.Variable(tf.truncated_normal([n_input, n_hidden_1],stddev=0.1)),
                'encoder_h2': tf.Variable(tf.truncated_normal([n_hidden_1, n_hidden_2],stddev=0.1)),
                'encoder_h3': tf.Variable(tf.truncated_normal([n_hidden_2, n_hidden_3],stddev=0.1)),
                'encoder_h4': tf.Variable(tf.truncated_normal([n_hidden_3, n_output],stddev=0.1)),
                'encoder_b1': tf.Variable(tf.truncated_normal([n_hidden_1],stddev=0.1)),
                'encoder_b2': tf.Variable(tf.truncated_normal([n_hidden_2],stddev=0.1)),
                'encoder_b3': tf.Variable(tf.truncated_normal([n_hidden_3],stddev=0.1)),
                'encoder_b4': tf.Variable(tf.truncated_normal([n_output],stddev=0.1)),         
            
            }
            layer_1 = tf.nn.relu(tf.add(tf.matmul(x, weights['encoder_h1']), weights['encoder_b1']))
            layer_2 = tf.nn.relu(tf.add(tf.matmul(layer_1, weights['encoder_h2']), weights['encoder_b2']))
            layer_3 = tf.nn.relu(tf.add(tf.matmul(layer_2, weights['encoder_h3']), weights['encoder_b3']))
            layer_4 = tf.nn.relu(tf.add(tf.matmul(layer_3, weights['encoder_h4']), weights['encoder_b4']))
            return layer_4, weights
        with tf.variable_scope('prediction'):
            self.s_t = tf.placeholder('float32',[None, n_input])            
            self.q, self.w = encoder(self.s_t)
            self.q_action = tf.argmax(self.q, dimension = 1)
        with tf.variable_scope('target'):
            self.target_s_t = tf.placeholder('float32', [None, n_input])
            self.target_q, self.target_w = encoder(self.target_s_t)
            self.target_q_idx = tf.placeholder('int32', [None,None], 'output_idx')
            self.target_q_with_idx = tf.gather_nd(self.target_q, self.target_q_idx)
            print(self.target_q , " this is target  Q") # changed here
            print(self.target_q_idx , "this is target Q ID") # changed here
        with tf.variable_scope('pred_to_target'):
            self.t_w_input = {}
            self.t_w_assign_op = {}
            for name in self.w.keys():
                print('name in self w keys', name)
                self.t_w_input[name] = tf.placeholder('float32', self.target_w[name].get_shape().as_list(),name = name)
                self.t_w_assign_op[name] = self.target_w[name].assign(self.t_w_input[name])   
                    
        
        def clipped_error(x):
            try:
                return tf.select(tf.abs(x) < 1.0, 0.5 * tf.square(x), tf.abs(x) - 0.5)
            except:
                return tf.where(tf.abs(x) < 1.0, 0.5 * tf.square(x), tf.abs(x) - 0.5)

        with tf.variable_scope('optimizer'):
            self.target_q_t = tf.placeholder('float32', None, name='target_q_t')
            self.action = tf.placeholder('int32',None, name = 'action')
            action_one_hot = tf.one_hot(self.action, n_output, 1.0, 0.0, name='action_one_hot')
            q_acted = tf.reduce_sum(self.q * action_one_hot, reduction_indices = 1, name='q_acted')
            self.delta = self.target_q_t - q_acted
            self.global_step = tf.Variable(0, trainable=False)
            self.loss = tf.reduce_mean(tf.square(self.delta), name = 'loss')
            self.learning_rate_step = tf.placeholder('int64', None, name='learning_rate_step')
            self.learning_rate_op = tf.maximum(self.learning_rate_minimum, tf.train.exponential_decay(self.learning_rate, self.learning_rate_step, self.learning_rate_decay_step, self.learning_rate_decay, staircase=True))
            self.optim = tf.train.RMSPropOptimizer(self.learning_rate_op, momentum=0.95, epsilon=0.01).minimize(self.loss) 
        
        tf.initialize_all_variables().run()
        self.update_target_q_network()

    def calculate_power_probabilities(self, platoon):
        power_selection = np.zeros(len(self.V2V_power_dB_List))
        total_active_links=0
        for vehicle in platoon:
            vehicle_index = self.env.vehicles.index(vehicle)
            for j in range(3):  # Assuming 3 V2V links per vehicle
                if self.env.activate_links[vehicle_index, j]:
                    power_choice = self.action_all_with_power[vehicle_index, j, 1]
                    power_selection[power_choice] += 1
                    total_active_links += 1
        total_active_links = np.sum(power_selection)
        if total_active_links > 0:
            power_selection /= total_active_links  # Normalize to get probabilities
        else:
            power_selection = np.zeros_like(power_selection)  # Avoid division by zero

        return power_selection
    
    def get_platoon_of_size(self, size):
        # Assuming self.env.vehicles is a list of all vehicles
        if size <= len(self.env.vehicles):
            return self.env.vehicles[:size]  # Return the first 'size' vehicles as the platoon
        else:
            return self.env.vehicles  # Return all vehicles if size exceeds the number of vehicles
    def plot_power_vs_platoon_size(self):
        plt.figure(figsize=(12, 8))
        self.env.platoon_sizes = [2, 4, 5, 8, 10, 20]

        markers = ['o', 's', 'D', '^', 'v']  # Using a variety of markers

        # Calculate power probabilities for each platoon size
        self.power_probs = []
        for size in self.env.platoon_sizes:
            # Retrieve or create a platoon of the specified size
            platoon = self.get_platoon_of_size(size)  # You need to implement this method
            power_probabilities = self.calculate_power_probabilities(platoon)  # Get probabilities for current platoon size
            self.power_probs.append(power_probabilities)

        self.power_probs = np.array(self.power_probs)
        
        # Plotting
        for i in range(len(self.V2V_power_dB_List)):
                plt.plot(self.env.platoon_sizes, 
                        self.power_probs[:, i],
                        marker=markers[i % len(markers)], 
                        linestyle='-', 
                        label=f'Power {self.V2V_power_dB_List[i]} dB',
                        alpha=0.75,  
                        linewidth=2,  
                        markersize=8)        
        plt.xlabel("Platoon Size", fontsize=14)  # Larger font size for readability
        plt.ylabel("Power Selection Probability", fontsize=14)
        plt.title("Power Selection Probability vs. Platoon Size", fontsize=16, fontweight='bold')
        plt.xticks([2, 4, 5, 8, 10, 20])  # Set the x-ticks
        # plt.ylim(0.15,0.80)
        plt.legend(fontsize=12)  # Larger legend for better readability
        plt.grid(True, linestyle='--', alpha=0.7)  # Adding dashed gridlines with some transparency
        plt.tight_layout()  # Ensuring that the layout fits well in the figure
        plt.savefig('power_vs_platoon_size40(45,30,10).png', dpi=300)

    def plot_raw_v2i_rate_vs_time(self, raw_v2i_rates, num_steps=None, interval=500):
        plt.figure(figsize=(10, 6))

        # If num_steps is specified, slice the raw_v2i_rates
        if num_steps is not None:
            raw_v2i_rates = raw_v2i_rates[:num_steps]  # Take only the first num_steps elements

        # Calculate mean raw V2I rate for every 'interval' steps
        num_intervals = len(raw_v2i_rates) // interval
        mean_raw_v2i_rates = []
        for i in range(num_intervals):
            start_index = i * interval
            end_index = start_index + interval
            mean_raw_v2i_rates.append(np.mean(raw_v2i_rates[start_index:end_index]))

        # Create an x-axis for the mean values
        x_values = np.arange(num_intervals) * interval + (interval / 2)  # Midpoint of each interval

        plt.plot(x_values, mean_raw_v2i_rates, 'r-', marker='o')
        plt.ylim(0,85)
        plt.xlabel('Time Step')
        plt.ylabel('Mean Raw V2I Rate (bps)')
        plt.title('Mean Raw V2I Rate vs Time (500-step intervals)')
        plt.grid(True)
        plt.savefig('mean_raw_v2i_rate_vs_time.png', dpi=300)
        # plt.show()  # Show the plot

    def update_target_q_network(self):    
        for name in self.w.keys():
            self.t_w_assign_op[name].eval({self.t_w_input[name]: self.w[name].eval()})       
        
    def save_weight_to_pkl(self): 
        if not os.path.exists(self.weight_dir):
            os.makedirs(self.weight_dir)
        for name in self.w.keys():
            save_pkl(self.w[name].eval(), os.path.join(self.weight_dir,"%s.pkl" % name))       
    def load_weight_from_pkl(self):
        with tf.variable_scope('load_pred_from_pkl'):
            self.w_input = {}
            self.w_assign_op = {}
            for name in self.w.keys():
                self.w_input[name] = tf.placeholder('float32')
                self.w_assign_op[name] = self.w[name].assign(self.w_input[name])
        for name in self.w.keys():
            self.w_assign_op[name].eval({self.w_input[name]:load_pkl(os.path.join(self.weight_dir, "%s.pkl" % name))})
        self.update_target_q_network() 
    
    image_counter = 0  
    def play(self, n_step = 100, n_episode = 100, test_ep = None, render = False):
        number_of_game = 40
        V2I_Rate_list = np.zeros(number_of_game)
        Fail_percent_list = np.zeros(number_of_game)
        self.load_weight_from_pkl()
        self.training = False
        
        for game_idx in range(number_of_game):
            self.env.new_random_game(self.num_vehicle)
            test_sample = 200
            Rate_list = []
            print('test game idx:', game_idx)
            print('The number of vehicle is ', len(self.env.vehicles))
            time_left_list = []
            power_select_list_0 = []
            power_select_list_1 = []
            power_select_list_2 = []
            self.plot_power_vs_platoon_size()

            for k in range(test_sample):
                print(k)
                action_temp = self.action_all_with_power.copy()
                reward, percent = self.env.act_asyn(action_temp)

                for i in range(len(self.env.vehicles)):
                    self.action_all_with_power[i, :, 0] = -1
                    sorted_idx = np.argsort(self.env.individual_time_limit[i, :])
                    for j in sorted_idx:
                        state_old = self.get_state([i, j])
                        time_left_list.append(state_old[-1])
                        action = self.predict(state_old, 0, True)
                        
                        if state_old[-1] <=0:
                            continue
                        power_selection = int(np.floor(action/self.RB_number))
                        if power_selection == 0:
                            power_select_list_0.append(state_old[-1])

                        if power_selection == 1:
                            power_select_list_1.append(state_old[-1])
                        if power_selection == 2:
                            power_select_list_2.append(state_old[-1])
                        
                        self.merge_action([i, j], action)
                    if i % (len(self.env.vehicles) / 10) == 1:
                        action_temp = self.action_all_with_power.copy()
                        reward, percent = self.env.act_asyn(action_temp)  # self.action_all)
                        Rate_list.append(np.sum(reward))
                # print("actions", self.action_all_with_power)
            
            number_0, bin_edges = np.histogram(power_select_list_0, bins = 10)
            number_1, bin_edges = np.histogram(power_select_list_1, bins = 10)
            number_2, bin_edges = np.histogram(power_select_list_2, bins = 10)
            p_0 = number_0 / (number_0 + number_1 + number_2)
            p_1 = number_1 / (number_0 + number_1 + number_2)
            p_2 = number_2 / (number_0 + number_1 + number_2)
            plt.figure()
            plt.plot(bin_edges[:-1]*0.1 + 0.01, p_0, 'b*-', label='Power Level 45 db')
            plt.plot(bin_edges[:-1]*0.1 + 0.01, p_1, 'rs-', label='Power Level 30 db')
            plt.plot(bin_edges[:-1]*0.1 + 0.01, p_2, 'go-', label='Power Level 10 db')
            # plt.xlim([0,0.12])
            plt.xlabel("Time left for V2V transmission (s)")
            plt.ylabel("Probability of power selection")
            plt.legend()
            plt.grid()
            plt.savefig('deepqnetwork40(45,30,10).png', dpi=300)
            
            V2I_Rate_list[game_idx] = np.mean(np.asarray(Rate_list))
            Fail_percent_list[game_idx] = percent

            print('Mean of the V2I rate is that ', np.mean(V2I_Rate_list[0:game_idx] ))
            print('Mean of Fail percent is that ',percent, np.mean(Fail_percent_list[0:game_idx]))
            print('action is that', action_temp[0,:])

        print('The number of vehicle is ', len(self.env.vehicles))
        print('Mean of the V2I rate is that ', np.mean(V2I_Rate_list))
        print('Mean of Fail percent is that ', np.mean(Fail_percent_list))
        # print('Test Reward is ', np.mean(test_result))

        self.plot_interference_heatmap()
        self.plot_resource_block_utilization()
        self.plot_v2i_rate_distribution(V2I_Rate_list)
        self.plot_failure_probability(Fail_percent_list)

    def plot_power_vs_vehicle_count(self, vehicle_counts):
        plt.figure(figsize=(12, 8))
        
        # Initialize the power selection probabilities list
        self.power_probs_vehicle_count = []
        
        # Calculate power probabilities for each vehicle count
        for n_vehicles in vehicle_counts:
            self.env.n_Veh = n_vehicles  # Set the number of vehicles in the environment
            self.env.new_random_game()  # Initialize a new random game
            self.num_vehicle = len(self.env.vehicles)  # Update the number of vehicles
            self.initialize_action_arrays()  # Reinitialize action arrays
            power_selection = self.calculate_power_probabilities(self.env.vehicles)  # Calculate power probabilities
            self.power_probs_vehicle_count.append(power_selection)
        
        self.power_probs_vehicle_count = np.array(self.power_probs_vehicle_count)

        # Plotting
        markers = ['o', 's', 'D']  # Using a variety of markers for different vehicle counts
        for i in range(len(self.V2V_power_dB_List)):
            plt.plot(vehicle_counts, 
                    self.power_probs_vehicle_count[:, i],
                    marker=markers[i % len(markers)], 
                    linestyle='-', 
                    label=f'Power {self.V2V_power_dB_List[i]} dB',
                    alpha=0.75,   
                    linewidth=2,   
                    markersize=8)         
        
        plt.xlabel("Number of Vehicles", fontsize=14)
        plt.ylabel("Power Selection Probability", fontsize=14)
        plt.title("Power Selection Probability vs. Number of Vehicles", fontsize=16, fontweight='bold')
        plt.xticks(vehicle_counts)  # Set the x-ticks
        plt.legend(fontsize=12)  # Larger legend for better readability
        plt.grid(True, linestyle='--', alpha=0.7)  # Adding dashed gridlines with some transparency
        plt.tight_layout()  # Ensuring that the layout fits well in the figure
        plt.savefig('power_vs_vehicle_count40(45,30,10).png', dpi=300)  # Save the plot

    def initialize_action_arrays(self):
        self.action_all_with_power = np.zeros([self.num_vehicle, 3, 2], dtype='int32')

    def plot_resource_block_utilization(self):
    # Adjust resource_block_usage computation to match self.RB_number
        resource_block_usage = np.sum(self.action_all_with_power[:, :, 0] >= 0, axis=0)
        
        # Check the length of resource_block_usage and self.RB_number
        if len(resource_block_usage) != self.RB_number:
            print(f"Shape mismatch: resource_block_usage has length {len(resource_block_usage)}, "
                f"but self.RB_number is {self.RB_number}. Adjusting...")
            # If there's a mismatch, pad or truncate resource_block_usage
            if len(resource_block_usage) < self.RB_number:
                # Pad with zeros if resource_block_usage is smaller
                resource_block_usage = np.pad(resource_block_usage, 
                                            (0, self.RB_number - len(resource_block_usage)),
                                            constant_values=0)
            else:
                # Truncate if resource_block_usage is larger
                resource_block_usage = resource_block_usage[:self.RB_number]
        
        # Plotting
        plt.figure()
        plt.bar(range(self.RB_number), resource_block_usage, color='purple', alpha=0.6)
        plt.xlabel("Resource Block Index")
        plt.ylabel("Usage Count")
        plt.title("Resource Block Utilization")
        plt.grid()
        plt.savefig("resource_block_utilization80.png", dpi=300)
        print("Saved plot: resource_block_utilization.png")

    def plot_v2i_rate_distribution(self, V2I_Rate_list):
        plt.figure()
        plt.hist(V2I_Rate_list, bins=20, color='blue', alpha=0.7)
        plt.xlabel("V2I Rate (Mbps)")
        plt.ylabel("Frequency")
        plt.title("Distribution of V2I Rates")
        plt.grid()
        plt.savefig("V2I_rate_distribution80.png", dpi=300)
        print("Saved plot: V2I_rate_distribution.png")

    def plot_failure_probability(self, Fail_percent_list):
        plt.figure()
        plt.plot(range(len(Fail_percent_list)), Fail_percent_list, marker='o')
        plt.xlabel("Game Index")
        plt.ylabel("Failure Probability")
        plt.ylim(0, 0.050)
        plt.title("Failure Probability Over Games")
        plt.grid()
        plt.savefig("failure_probability_over_games80.png", dpi=300)
        print("Saved plot: failure_probability_over_games.png")

    def plot_interference_heatmap(self):
        plt.figure()
        plt.imshow(self.env.V2V_Interference, cmap='hot', interpolation='nearest')
        plt.colorbar(label="Interference (dBm)")
        plt.xlabel("Resource Blocks")
        plt.ylabel("Vehicles")
        plt.title("Interference Heatmap")
        plt.savefig("interference_heatmap80.png", dpi=300)
        print("Saved plot: interference_heatmap.png")

def main(_):
 
  up_lanes = [3.5/2,3.5/2 + 3.5,250+3.5/2, 250+3.5+3.5/2, 500+3.5/2, 500+3.5+3.5/2]
  down_lanes = [250-3.5-3.5/2,250-3.5/2,500-3.5-3.5/2,500-3.5/2,750-3.5-3.5/2,750-3.5/2]
  left_lanes = [16-2,20-2,24-2]
  right_lanes = [2,2+4,2+8]
  width = 10000
  height = 24
  Env = Environ(down_lanes,up_lanes,left_lanes,right_lanes, width, height) 
  Env.new_random_game()
  Env.test_channel()
  # Assign vehicles to platoons (NEW FUNCTIONALITY)
  Env.assign_vehicles_to_platoons()  # New method to assign vehicles to different platoon sizes
  # Initialize the agent
  # Initialize the agent
  agent = Agent(Env)  # Assuming the agent interacts with the environment

  # Main simulation loop
  num_episodes = 10  # Define the number of episodes
  max_steps = 100  # Define the maximum steps for each episode

  for episode in range(num_episodes):
      agent.reset()  # Reset the agent's state for each episode

      for step in range(max_steps):
        print("this function is working !")
        actions = agent.choose_action()  # Agent selects actions
        reward = Env.step(actions)  # Environment processes the actions and returns reward
        agent.learn(reward)  # Agent learns based on the reward

      # After completing the steps, plot power selection probabilities vs. platoon size
      agent.plot_power_vs_platoon_size()  # New function to generate the required plot
  '''
  gpu_options = tf.GPUOptions(
      per_process_gpu_memory_fraction=calc_gpu_fraction(FLAGS.gpu_fraction))
  config = tf.ConfigProto()
  config.gpu_options.allow_growth = True
'''
  with tf.Session(config=tf.ConfigProto()) as sess:
    config = []
    agent = Agent(config, Env, sess)
    #agent.play()
    agent.train()
    agent.play()

if __name__ == '__main__':
    tf.app.run()
        

