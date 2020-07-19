import json
import socket
import cv2
import os
import psutil

from tensorflow.keras.layers import Input, Convolution2D, Flatten, Dense, LeakyReLU
from tensorflow.keras.initializers import VarianceScaling
from src.client.agent_client import AgentClient, GameState
from src.utils.utils import *
from src.utils.Vision import Vision
from src.utils.ReplayMemory import ReplayMemory
from threading import Thread


def get_validation_level_numbers():
    val_numbers = []
    for i in range(TOTAL_LEVEL_NUMBER // 100):
        for j in range(10):
            val_numbers.append(i * 100 + j + 1)
    return val_numbers


# Global
TOTAL_LEVEL_NUMBER = 2400  # non-novelty levels
LIST_OF_VALIDATION_LEVELS = get_validation_level_numbers()  # list of levels used for validation

# Action space
ANGLE_RESOLUTION = 20  # the number of possible (discretized) shot angles
TAP_TIME_RESOLUTION = 10  # the number of possible tap times
MAXIMUM_TAP_TIME = 4000  # maximum tap time (in ms)
PHI = 10  # dead shot angle bottom (in degrees)
PSI = 40  # dead shot angle top (in degrees)

# State space
STATE_PIXEL_RESOLUTION = 124  # width and height of (preprocessed) states

# Reward
SCORE_NORMALIZATION = 150000


class ClientDQNAgent(Thread):
    """Deep Q-Network (DQN) agent for playing Angry Birds"""

    def __init__(self, name, dueling=True, latent_dim=512, learning_rate=0.0001):
        super().__init__()

        self.name = name
        self._setup_client_server_connection()
        self.vision = Vision()  # for obtaining sling reference point

        # To use the dueling networks feature
        self.dueling = dueling

        # Training optimizer and parameters
        self._optimizer = tf.optimizers.Adam(learning_rate=learning_rate)

        # Initialize the architecture of the acting and learning part of the DQN (theta)
        self.online_network = self._build_compile_model(latent_dim)

        # Initialize the architecture of a shadowed (target) version of the DQN (theta-),
        # which computes the values during learning
        self.target_network = self.online_network

        # Initialize the memory where all the experience will be memorized
        self.memory = ReplayMemory(STATE_PIXEL_RESOLUTION, SCORE_NORMALIZATION)

        print('DQN agent initialized.')

    def _setup_client_server_connection(self):
        self.id = 28888

        with open('./src/client/server_client_config.json', 'r') as config:
            sc_json_config = json.load(config)
        self.ar = AgentClient(**sc_json_config[0])

        with open('./src/client/server_observer_client_config.json', 'r') as observer_config:
            observer_sc_json_config = json.load(observer_config)
        self.observer_ar = AgentClient(**observer_sc_json_config[0])

        print("Connecting agent to server...")
        try:
            self.ar.connect_to_server()
        except socket.error as e:
            print("Error in client-server communication: " + str(e))

        print("Connecting observer agent to server...")
        try:
            self.observer_ar.connect_to_server()
        except socket.error as e:
            print("Error in client-server communication: " + str(e))

        self.ar.configure(self.id)
        self.observer_ar.configure(self.id)

    def _build_compile_model(self, latent_dim):

        input_frame = Input(shape=(STATE_PIXEL_RESOLUTION, STATE_PIXEL_RESOLUTION, 3))

        conv1 = Convolution2D(32, (8, 8), strides=4, kernel_initializer=VarianceScaling(scale=2.), activation='relu',
                              use_bias=False)(input_frame)

        conv2 = Convolution2D(64, (4, 4), strides=2, kernel_initializer=VarianceScaling(scale=2.), activation='relu',
                              use_bias=False)(conv1)

        conv3 = Convolution2D(64, (3, 3), strides=1, kernel_initializer=VarianceScaling(scale=2.), activation='relu',
                              use_bias=False)(conv2)

        conv4 = Convolution2D(latent_dim, (7, 7), strides=1, kernel_initializer=VarianceScaling(scale=2.),
                              activation='relu',
                              use_bias=False)(conv3)

        latent_feature_1 = Flatten(name='latent')(conv4)

        # Implementation of the Dueling Network principle
        if self.dueling:
            # State value prediction
            latent_feature_2 = Dense(512, activation='relu', name='latent_V')(latent_feature_1)
            state_value = Dense(1, name='V')(latent_feature_2)

            # Advantage prediction
            latent_feature_2 = tf.keras.layers.Dense(128, activation='relu', name='latent_A')(latent_feature_1)
            advantage = Dense(ANGLE_RESOLUTION * TAP_TIME_RESOLUTION, name='A')(latent_feature_2)

            # Q-value = average of both sub-networks
            # q_value = tf.keras.layers.Average(name='Q')([advantage, state_value])
            q_values = tf.add(state_value,
                              tf.subtract(advantage, tf.reduce_mean(advantage, axis=1,
                                                                    keepdims=True,
                                                                    name='A_mean'),
                                          name='Sub'),
                              name='Q')
        else:
            # Direct Q-value prediction
            latent_feature_2 = tf.keras.layers.Dense(128, activation='relu', name='latent_2')(latent_feature_1)
            lrelu_feature = LeakyReLU()(latent_feature_2)
            q_values = Dense(ANGLE_RESOLUTION * TAP_TIME_RESOLUTION, name='Q')(lrelu_feature)

        model = tf.keras.Model(inputs=[input_frame], outputs=[q_values])
        model.compile(loss='huber_loss', optimizer=self._optimizer)

        # model.summary()
        # tf.keras.utils.plot_model(model, to_file='plots/model_plot.png', show_shapes=True, show_layer_names=True)
        return model

    def practice(self, num_episodes, minibatch, sync_period, gamma, epsilon, epsilon_anneal, replay_period,
                 grace_factor, delta, delta_anneal, sim_speed=60):
        """The agent's main training routine.

        :param num_episodes: Number of episodes to play
        :param minibatch: Number of transitions to be learned from when learn() is invoked
        :param sync_period: The number of levels between each synchronization of online and target network. The
                            higher the number, the stronger Double Q-Learning and the less overestimation.
        :param gamma: Discount factor
        :param epsilon: Probability for random shot (epsilon greedy policy)
        :param epsilon_anneal: Decrease factor for epsilon
        :param replay_period: number of levels between each training of the online network
        :param grace_factor: reward modifier, granting X % points on failed levels
        :param delta: Trade-off factor between Monte Carlo target return and one-step target return,
                      delta = 1 means MC return, delta = 0 means one-step return
        :param delta_anneal: Decrease factor for delta
        :param sim_speed: for Science Birds (max. 60)
        """

        self.ar.set_game_simulation_speed(sim_speed)

        print("DQN agent starts practicing...")

        # Load the first level to train in Science Birds
        self.load_next_level()

        # Initialize return list (list of normalized final scores for each level played)
        returns = []
        wins = []
        scores = []

        # For memory tracking
        process = psutil.Process(os.getpid())

        for i in range(1, num_episodes + 1):
            print("\nEpisode %d, Level %d, epsilon = %f" % (i, self.ar.get_current_level(), epsilon))

            # Play 1 episode = 1 level and save observations
            obs, rewards, ret, won, score = self.play_level(grace_factor, epsilon)

            # Handle situations where a level destroyed itself with no shots taken
            if len(obs) == 0:
                self.load_next_level()
                continue

            # Load next level (in advance)
            self.load_next_level()

            # Memorize all the observed information
            self.memory.memorize(obs, rewards, won, gamma, grace_factor)
            returns += [ret]
            wins += [won]
            scores += [score]

            # Every X episodes, plot informative graphs
            if i % 500 == 0:
                plot_win_loss_ratio(wins)
                # plot_priorities(self.memory.get_priorities())  # useful to determine batch size
                plot_moving_average(np.array(returns),
                                    title="Returns gathered so far",
                                    ylabel="Return",
                                    output_path="plots/returns.png")
                plot_moving_average(np.array(scores),
                                    title="Scores gathered so far",
                                    ylabel="Score",
                                    output_path="plots/scores.png")

            print("Current RAM usage: %d MB" % (process.memory_info().rss / 1000000))

            # Simultaneously, update the network weights every train_period levels to fit experience
            if i % replay_period == 0:
                learn_thread = Thread(target=self.learn, args=(gamma, minibatch, delta, grace_factor))
                learn_thread.start()

            # Save model checkpoint
            if i % 1000 == 0:
                self.save_model(model_path="temp/checkpoint", overwrite=True)

            # Save and reload experience to reduce memory load
            if i % 1000 == 0:
                path = "temp/experience_%s.hdf5" % i
                old_path = "temp/experience_%s.hdf5" % (i - 1000)
                self.memory.export_experience(experience_path=path, overwrite=True)
                self.memory.import_experience(experience_path=path)
                if os.path.exists(old_path):
                    os.remove(old_path)

            # Synchronize target and online network every sync_period levels
            if i % sync_period == 0:
                self.target_network = self.online_network

            # Perform Validation of the agent every 2000 levels
            if i % 2000 == 0:
                self.validate()
                self.load_next_level()

            # Cool down
            epsilon *= epsilon_anneal  # reduces randomness (less explore, more exploit)
            delta *= delta_anneal  # shifts target return fom MC to one-step

        print("Practicing finished successfully!")

    def play_level(self, grace_factor, epsilon):
        # Observations during the current level: list of (state, action, score) tuples
        obs = []

        # Initialize current episode's score to 0
        score = 0

        # Initialize variable to monitor the application's state
        appl_state = self.ar.get_game_state()

        # Get the environment state (preprocessed screenshot)
        env_state = self.get_state()

        # Try to solve a level and collect observations
        while appl_state == GameState.PLAYING:
            # Predict the next action to take, i.e. the best shot, and get estimated value
            action, pred_ret = self.plan(env_state, epsilon)
            print("Expected total return: %.3f" % (pred_ret + score / SCORE_NORMALIZATION))

            # Try to plot a saliency map without classes
            # plot_saliency_map(env_state, self.target_network)

            # Perform shot, observe new environment state, level score and application state
            next_env_state, new_score, appl_state = self.shoot(action)

            # Save experienced transition
            obs += [(env_state, action, new_score - score)]

            # Update current score
            score = new_score

            # Update old env_state with new one
            env_state = next_env_state

        # In case the level did solve itself by self-destruction or something other unexpected happens
        if len(obs) == 0 or not (appl_state == GameState.WON or appl_state == GameState.LOST):
            score = self.ar.get_current_score()
            ret = score / SCORE_NORMALIZATION
            return [], [], ret, None, score

        # Prepare observed information
        obs = np.array(obs)
        won = appl_state == GameState.WON
        print("Level %s with score %d." % (("won" if won else "lost"), score))
        rewards = compute_reward(obs[:, 2], won, grace_factor)
        ret = np.sum(rewards)
        score *= won

        return obs, rewards, ret, won, score

    def learn(self, gamma, minibatch, delta, grace_factor, alpha=0.7, beta=0.5):
        """Updates the online network's weights. This is the actual learning procedure of the agent.

        :param gamma: Discount factor
        :param minibatch: Number of transitions to be learned from
        :param delta: trade-off factor between Monte Carlo target return and one-step target return,
                      delta = 1 means MC return, delta = 0 means one-step return
        :param grace_factor:
        :param alpha: For Prioritized Experience Replay: the larger alpha the stronger prioritization
        :param beta: ?
        :return:
        """

        print("\n\033[94mLearning from experience...\033[0m")

        # Obtain a list of useful transitions to learn on
        trans_ids, probabilities = self.memory.recall(minibatch, alpha)

        # Obtain total number of experienced transitions
        exp_len = self.memory.get_length()

        # Compute importance-sampling weights and normalize
        weights = (exp_len * probabilities[trans_ids]) ** (- beta)
        weights /= np.max(weights)

        # Get list of transitions
        states, actions, rewards, next_states, terminals = \
            self.memory.get_transitions(trans_ids, grace_factor)

        # Normalize all states and next states
        states = np.reshape(states / 255, (-1, STATE_PIXEL_RESOLUTION, STATE_PIXEL_RESOLUTION, 3))
        next_states = np.reshape(next_states / 255, (-1, STATE_PIXEL_RESOLUTION, STATE_PIXEL_RESOLUTION, 3))

        # Obtain Monte Carlo return for each transition
        mc_returns = self.memory.get_returns(trans_ids, gamma)

        # Predict returns (i.e. values V(s)) for all states s
        pred_returns = np.max(self.online_network.predict(states), axis=1)

        # Predict next returns
        pred_next_returns = np.max(self.target_network.predict(next_states), axis=1)

        # Compute one-step return
        one_step_returns = rewards + gamma * pred_next_returns

        # Compute convex combination of MC return and one-step return
        target_returns = delta * mc_returns + (1 - delta) * one_step_returns

        # Set target return = reward for all terminal transitions
        target_returns[terminals == True] = rewards[terminals == True]

        # Compute Temporal Difference (TD) errors (the "surprise" of the agent)
        td_errs = target_returns - pred_returns

        # Update transition priorities according to TD errors
        self.memory.set_priorities(trans_ids, np.abs(td_errs))

        # Prepare inputs and targets for fitting
        inputs = states
        targets = self.target_network.predict(states)
        targets[range(len(trans_ids)), actions] = target_returns

        # Update the online network's weights
        self.online_network.fit(inputs, targets, epochs=1, verbose=0,
                                batch_size=minibatch,
                                sample_weight=np.abs(np.multiply(weights, td_errs)))

        print("\033[94mDone with learning.\033[0m")

    def validate(self, grace_factor=0.25):
        """Perform validation of the agent on the validation set. On all validation levels,
        the agent plays without epsilon and does not learn from the experience."""
        print("Start validating...")

        self.ar.set_game_simulation_speed(60)

        # Initialize return list (list of normalized final scores for each level played)
        returns = []
        scores = []
        wins = []

        for level in LIST_OF_VALIDATION_LEVELS:
            print("\nValidating on level", level)

            # load next validation level
            self.ar.load_level(level)

            # let the agent play the level
            _, _, ret, won, score = self.play_level(grace_factor=grace_factor, epsilon=None)

            # check if the level was solved by self-destruction (won is None)
            if won is not None:
                returns += [ret]
                scores += [score]
                wins += [won]
            else:
                print("\033[93mLevel did destruct itself... ¯\\_(ツ)_/¯\033[0m")
                returns += [ret]
                scores += [score]
                wins += [1]

        # plot the results
        plot_validation(returns,
                        title="Validation returns",
                        ylabel="Averaged return",
                        output_path="plots/validation-returns.png")
        plot_validation(scores,
                        title="Validation scores",
                        ylabel="Averaged score",
                        output_path="plots/validation-scores.png")
        plot_validation(wins,
                        title="Validation win-loss ratio",
                        ylabel="Averaged win proportion",
                        output_path="plots/validation-win-loss-ratio.png")

        print("Finished validating.")

        return_avg = np.average(returns)
        score_avg = np.average(scores)
        win_loss_ratio = np.average(wins)
        print("Return average:", return_avg)
        print("Score average:", score_avg)
        print("Win-loss ratio:", win_loss_ratio)

        return return_avg, score_avg, win_loss_ratio

    def plan(self, state, epsilon=None):
        """Given a state of the game, the deep DQN is used to predict a good shot.

        :param state: the preprocessed, unnormalized state pixel matrix
        :param epsilon: If given, epsilon greedy policy will be applied, otherwise the agent plans optimally
        :return: action, consisting of an index, corresponding to some shot parameters
        """

        # Normalize
        norm_state = state / 255

        # Obtain list of action-values Q(s,a)
        q_vals = self.online_network.predict(norm_state)

        # Do epsilon-greedy
        if epsilon is not None and np.random.random(1) < epsilon:
            # Choose action by random
            action = np.random.randint(ANGLE_RESOLUTION * TAP_TIME_RESOLUTION)
        else:
            # Determine optimal action as usual (index of action with highest value)
            action = q_vals.argmax()

        # Estimate the expected value for this level
        val_estimate = np.amax(q_vals)

        return action, val_estimate

    def shoot(self, action):
        """Performs a shot and observes and returns the consequences."""

        # Get sling reference point coordinates
        sling_x, sling_y = self.vision.get_sling_reference()

        # Convert action index into aim vector and tap time
        dx, dy, tap_time = action_to_params(action)

        # Perform the shot
        # print("Shooting with dx = %d, dy = %d, tap_time = %d" % (dx, dy, tap_time))
        self.ar.shoot(sling_x, sling_y, dx, dy, 0, tap_time, isPolar=False)

        # Get the environment state (cropped screenshot)
        env_state = self.get_state()

        # Obtain game score
        score = self.ar.get_current_score()

        # Get the application state
        appl_state = self.ar.get_game_state()

        return env_state, score, appl_state

    def get_state(self):
        """Fetches the current game screenshot and turns it into a cropped and scaled pixel matrix."""

        # Obtain game screenshot
        screenshot, ground_truth = self.ar.get_ground_truth_with_screenshot()

        # Update Vision (to get an up-to-date sling reference point)
        self.vision.update(screenshot, ground_truth)

        # Crop the image to reduce information overload.
        # The cropped image has then dimension (325, 800, 3).
        crop = screenshot[75:400, 40:]

        # Rescale the image into a (smaller) square
        scaled = cv2.resize(crop, (STATE_PIXEL_RESOLUTION, STATE_PIXEL_RESOLUTION))

        # Convert into unsigned byte
        state = np.expand_dims(scaled.astype(np.uint8), axis=0)

        return state

    def load_next_level(self, next_level=None):
        """Loads randomly a non-validation level."""

        # While practicing, pick a random training level
        if next_level is None:
            next_level = pick_random_level_number()
            while next_level in LIST_OF_VALIDATION_LEVELS:
                next_level = pick_random_level_number()

        self.ar.load_level(next_level)

    def learn_from_experience(self, num_epochs, gamma, minibatch, delta, grace_factor, sync_period=128,
                              reset_priorities=True):
        """Tells the agent to learn from the its current experience."""
        # TODO: Test this function

        process = psutil.Process(os.getpid())  # for memory tracking
        self.ar.set_game_simulation_speed(60)  # for validation

        if reset_priorities:
            self.memory.reset_priorities()

        exp_len = self.memory.get_length()
        print("Learning from experience (%d transitions) for %d epochs..." % (exp_len, num_epochs))

        return_avgs, score_avgs, win_loss_ratios = [], [], []

        print("Epoch: 0, current RAM usage: %d MB" % (process.memory_info().rss / 1000000))

        for i in range(1, num_epochs + 1):
            if i % 10 == 0:
                print("Epoch: %d, current RAM usage: %d MB" % (i, (process.memory_info().rss / 1000000)))

            self.learn(gamma, minibatch, delta, grace_factor)

            if i % 100 == 0:
                plot_priorities(self.memory.get_priorities())

            # Synchronize target and online network every sync_period levels
            if i % sync_period == 0:
                self.target_network = self.online_network

            # Perform Validation of the agent every X levels
            if i % 250 == 0:
                return_avg, score_avg, win_loss_ratio = self.validate()
                print("Return average:", return_avg)
                print("Score average:", score_avg)
                print("Win-loss ratio:", win_loss_ratio)
                return_avgs += [return_avg]
                score_avgs += [score_avgs]
                win_loss_ratios += [win_loss_ratio]

        print("Return averages:", return_avgs)
        print("Score averages:", score_avgs)
        print("Win-loss ratios:", win_loss_ratios)

        print("Learning from experience done!")

    def just_play(self):
        print("Just playing around...")
        self.ar.set_game_simulation_speed(3)

        while True:
            self.load_next_level()
            print("\nLevel %d" % self.ar.get_current_level())
            self.play_level(1, epsilon=None)

    def save_model(self, model_path=None, overwrite=False, checkpoint_no=None, temp=False):
        """Saves the current model weights to a specified export path."""
        if model_path is None:
            model_path = ("temp" if temp else "models") + "/" + self.name
        if checkpoint_no is not None:
            model_path += "_episode_" + checkpoint_no
        self.online_network.save_weights(model_path, overwrite=overwrite)
        print("Saved model.")

    def restore_model(self, model_path="temp/checkpoint"):
        print("Restoring model from '%s'." % model_path)
        self.online_network.load_weights(model_path)
        self.target_network = self.online_network

    def save_experience(self, experience_path=None, overwrite=False, compress=False):
        if experience_path is None:
            experience_path = "data/" + self.name
        self.memory.export_experience(experience_path, overwrite, compress)

    def restore_experience(self, experience_path=None, grace_factor=None, gamma=None):
        if experience_path is None:
            experience_path = "data/" + self.name
        self.memory.import_experience(experience_path, grace_factor, gamma)

    def forget(self):
        self.memory = ReplayMemory(STATE_PIXEL_RESOLUTION, SCORE_NORMALIZATION)


def compute_reward(score, won, grace_factor):
    """Turns scores into rewards."""
    reward = score / SCORE_NORMALIZATION
    if not won:
        reward *= grace_factor
    return np.array(reward, dtype='float32')


def action_to_params(action):
    """Converts a given action index into corresponding dx, dy, and tap time."""

    # Convert the action index into index pair, indicating angle and tap_time
    action = np.unravel_index(action, (ANGLE_RESOLUTION, TAP_TIME_RESOLUTION))

    # Formula parameters
    c = 3.6
    d = 1.3

    # Retrieve shot angle alpha
    k = action[0] / ANGLE_RESOLUTION
    alpha = ((1 + 0.5 * c) * k - 3 / 2 * c * k ** 2 + c * k ** 3) ** d * (180 - PHI - PSI) + PHI

    # Convert angle into vector
    dx, dy = angle_to_vector(alpha)

    # Retrieve tap time
    t = action[1]
    tap_time = int(t / TAP_TIME_RESOLUTION * MAXIMUM_TAP_TIME)

    print("Shooting with: alpha = %d °, tap time = %d ms" % (alpha, tap_time))

    return dx, dy, tap_time


def pick_random_level_number():
    return np.random.randint(TOTAL_LEVEL_NUMBER) + 1
