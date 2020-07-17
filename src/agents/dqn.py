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
    for i in range(TOTAL_LEVEL_NUMBER//100):
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

    def __init__(self, dueling=True, latent_dim=512, learning_rate=0.0001):
        super().__init__()

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
        self.memory = ReplayMemory(state_res_per_dim=STATE_PIXEL_RESOLUTION)

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

    def practice(self, num_episodes, minibatch, sync_period, gamma, epsilon, anneal, replay_period, grace_factor,
                 sim_speed=60):
        """The agent's main training routine.

        :param num_episodes: Number of episodes to play
        :param minibatch: Number of transitions to be learned from when learn() is invoked
        :param sync_period: The number of levels between each synchronization of online and target network. The
                            higher the number, the stronger Double Q-Learning and the less overestimation.
        :param gamma: Discount factor
        :param epsilon: Probability for random shot (epsilon greedy policy)
        :param anneal: Decrease factor for epsilon
        :param replay_period: number of levels between each training of the online network
        :param grace_factor: reward modifier, granting X % points on failed levels
        :param sim_speed: for Science Birds (max. 60)
        """

        self.ar.set_game_simulation_speed(sim_speed)

        print("DQN agent starts practicing...")

        # Load the first level to train in Science Birds
        self.load_next_level()

        # Initialize return list (list of normalized final scores for each level played)
        returns = []
        win_loss_ratio = []

        # For memory tracking
        process = psutil.Process(os.getpid())

        for i in range(1, num_episodes + 1):
            # Perform Validation of the agent every 2000 levels
            if i % 2000 == 0:
                self.validate()

                # load a new level for regular training
                self.load_next_level()

            print("\nEpisode %d, Level %d, epsilon = %f" % (i, self.ar.get_current_level(), epsilon))

            # Play 1 episode = 1 level and save observations
            obs, ret, won = self.play_level(grace_factor, epsilon)

            # Handle situations where a level destroyed itself with no shots taken
            if len(obs) == 0:
                self.load_next_level()
                continue

            self.memory.memorize(obs)
            returns += [ret]
            win_loss_ratio += won

            # Every X episodes, plot informative graphs
            if i % 500 == 0:
                plot_win_loss_ratio(win_loss_ratio)
                # plot_priorities(self.memory.get_priorities())  # useful to determine batch size
                plot_scores(np.array(returns) * SCORE_NORMALIZATION)

            print("Current RAM usage: %d MB" % (process.memory_info().rss / 1000000))

            # Load next level (in advance)
            self.load_next_level()

            # Simultaneously, update the network weights every train_period levels to fit experience
            if i % replay_period == 0:
                learn_thread = Thread(target=self.learn, args=(gamma, minibatch))
                learn_thread.start()

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

            # Cool down: reduce epsilon to reduce randomness (less explore, more exploit)
            epsilon *= anneal

        print("Practicing finished successfully!")

    def play_level(self, grace_factor, epsilon):
        # Observations during the current level: list of (s, a, r, s', t) tuples
        obs = []

        # Initialize current episode's return to 0
        ret = 0

        # Initialize variable to monitor the application's state
        appl_state = self.ar.get_game_state()

        # Get the environment state (preprocessed screenshot) and save it
        env_state = self.get_state()

        # Try to solve a level and collect observations (actions, environment state, reward)
        while appl_state == GameState.PLAYING:
            # Plot the current state
            # plot_state(env_state)

            # Predict the next action to take, i.e. the best shot, and get estimated value
            action, val_estimate = self.plan(env_state, epsilon)

            expected_total_return = val_estimate + np.sum(ret)
            print("Expected total return:", expected_total_return)

            # Try to plot a saliency map without classes
            # plot_saliency_map(env_state, self.target_network)

            # Perform shot, observe new environment state, level score and application state
            next_env_state, score, appl_state = self.shoot(action)

            # Compute reward (to be the number of additional points gained with this shot)
            reward = score - ret

            # Observe if this level was terminated
            terminal = not appl_state == GameState.PLAYING

            # Save experienced transition (s, a, r, s', t)
            obs += [(env_state, action, reward, next_env_state, terminal)]

            # Update return
            ret = score

            # Update old env_state with new one
            env_state = next_env_state

        if not (appl_state == GameState.WON or appl_state == GameState.LOST):
            print("Error: unexpected application state. The application state is neither WON nor LOST. "
                  "Skipping this training iteration...")

        # In case the level did solve itself by self-destruction
        if len(obs) == 0:
            return [], 0, []

        # Convert observations list into np.array
        obs = np.array(obs)

        # If the level is lost, punish the return. In the actual case, reward and return would
        # be zero, but we grant some "grace" points, so the network can learn even if it constantly looses.
        if appl_state == GameState.LOST:
            # Grace points on all the rewards given during this level
            obs[:, 2] *= grace_factor

            # Grace points on return
            ret *= grace_factor

            # add a loss to the ratio
            won = [0]
            print("Level lost.")
        else:
            # add a win to the ratio
            won = [1]
            print("Level won!")

        print("Got level score %d." % self.ar.get_current_score())

        return obs, ret, won

    def validate(self):
        """Perform validation of the agent on the validation set. On all validation levels,
        the agent plays without epsilon and does not learn from the experience."""
        print("Start validating...")

        self.ar.set_game_simulation_speed(60)

        # Initialize return list (list of normalized final scores for each level played)
        returns = []
        win_loss_ratio = []

        for level in LIST_OF_VALIDATION_LEVELS:
            print("\nValidating on level", level)

            # load next validation level
            self.ar.load_level(level)

            # let the agent play the level
            ret, won = self.validate_level()

            # check if the level was solved by self-destruction (ret and won are None)
            if (ret is not None) and (won is not None):
                returns += [ret]
                win_loss_ratio += won

        # plot the results
        plot_win_loss_ratio(win_loss_ratio)
        plot_scores(np.array(returns) * SCORE_NORMALIZATION)
        print("Finished validating.")

    def validate_level(self):
        """Let the agent play one level without epsilon."""
        # Initialize current level's return to 0
        score = 0

        # Initialize variable to monitor the application's state
        appl_state = self.ar.get_game_state()

        # Get the environment state (preprocessed screenshot) and save it
        env_state = self.get_state()

        # The level might be solved by self-destruction
        shoot_once = False

        # Try to solve the current level
        while appl_state == GameState.PLAYING:
            # Predict the next action to take, i.e. the best shot, and get estimated value
            action, val_estimate = self.plan(env_state)

            # Perform shot, observe new environment state, level score and application state
            env_state, score, appl_state = self.shoot(action)

            # The level wasn't solved by self-destruction
            shoot_once = True

        if not (appl_state == GameState.WON or appl_state == GameState.LOST):
            print("Error: unexpected application state. The application state is neither WON nor LOST. "
                  "Skipping this validation level...")

        # In case the level did solve itself by self-destruction
        if not shoot_once:
            print("Level was solved by self-destruction, it will not be used for win-ratio and average points.")
            return None, None

        # add a win/loss to the ratio depending on the level outcome
        if appl_state == GameState.LOST:
            won = [0]
            print("Level lost.")
        else:
            won = [1]
            print("Level won!")

        print("Got level score %d." % self.ar.get_current_score())

        return score, won

    def learn(self, gamma, minibatch, alpha=0.7, beta=0.5):
        """Updates the online network's weights. This is the actual learning procedure of the agent.

        :param gamma: Discount factor
        :param minibatch: Number of transitions to be learned from
        :param alpha: For Prioritized Experience Replay: the larger alpha the stronger prioritization
        :param beta: ?
        :return:
        """

        # TODO: Make this more efficient (especially for large batches)
        # TODO: Implement Monte Carlo return
        print("\n\033[94mLearning from experience...\033[0m")

        # Obtain a list of useful transitions to learn on
        trans_ids, probabilities = self.memory.recall(minibatch, alpha)

        # Initialize sample weight, input, and targets list
        td_errs = []
        inputs = []
        targets = []

        # Obtain total number of experienced transitions
        exp_len = self.memory.get_length()

        # Compute importance-sampling weights and normalize
        weights = (exp_len * probabilities[trans_ids]) ** (- beta)
        weights /= np.max(weights)

        # Get list of transitions
        transitions = self.memory.get_transitions(trans_ids)

        # For each transition in the given batch
        for trans_id, (state, action, reward, next_state, terminal) in zip(trans_ids, transitions):

            norm_state = state / 255
            norm_next_state = next_state / 255

            # Predict value V(s) for current state s
            pred_val = np.max(self.online_network.predict(norm_state))

            # Compute TD error (difference between expected and observed (target) return)
            if terminal:
                # If this is the last transition of the episode, take the reward directly
                target_val = reward
            else:
                # Else, use predicted reward of next step
                next_action = np.argmax(self.online_network.predict(norm_next_state))
                next_val = self.target_network.predict(norm_next_state)[0][next_action]
                target_val = reward + gamma * next_val

            td_err = target_val - pred_val

            td_errs += [td_err]

            # Update transition priority
            self.memory.set_priority(trans_id, np.abs(td_err))

            # Predict Q-value matrix for given state and modify the action's Q-value
            target = self.target_network.predict(norm_state)
            target[0][action] = target_val

            # Accumulate training data
            inputs += [norm_state[0]]
            targets += [target[0]]

        inputs = np.asarray(inputs)
        targets = np.asarray(targets)

        # Update the online network's weights
        self.online_network.fit(inputs, targets, epochs=1, verbose=0,
                                batch_size=minibatch,
                                sample_weight=np.abs(np.multiply(weights, td_errs)))

        print("\033[94mDone with learning.\033[0m")

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

        # Obtain normalized game score
        score = self.ar.get_current_score() / SCORE_NORMALIZATION

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
        """Update the current_level variable and load the next level according to given application state."""

        # While practicing, pick a random training level
        if next_level is None:
            next_level = pick_random_level_number()
            while next_level in LIST_OF_VALIDATION_LEVELS:
                next_level = pick_random_level_number()

        self.ar.load_level(next_level)

    def learn_from_experience(self, gamma, minibatch, reset_priorities=False):
        """Tells the agent to learn from the its current experience."""
        # TODO: Rework this function

        if reset_priorities:
            self.memory.reset_priorities()

        exp_len = self.memory.get_length()
        num_epochs = int(exp_len / 40)
        print("Learning from experience for %d epochs..." % num_epochs)

        print(self.memory.get_length())

        for i in range(num_epochs):
            if i % 10 == 0:
                print("Epoch:", i)

            if i % 100 == 0:
                plot_priorities(self.memory.get_priorities())

            self.learn(gamma, minibatch)

    def just_play(self):
        print("Just playing around...")
        self.ar.set_game_simulation_speed(3)

        while True:
            self.load_next_level()
            print("\nLevel %d" % self.ar.get_current_level())
            self.play_level(1, epsilon=None)

    def save_model(self, model_path, overwrite=False):
        """Saves the current model weights to a specified export path."""
        self.online_network.save_weights(model_path, overwrite=overwrite)
        print("Saved model.")

    def restore_model(self, model_path):
        print("Restoring model from '%s'." % model_path)
        self.online_network.load_weights(model_path)
        self.target_network = self.online_network

    def forget(self):
        self.memory = ReplayMemory(STATE_PIXEL_RESOLUTION)


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
    tap_time = int(t / 10 * MAXIMUM_TAP_TIME)

    print("Shooting with: alpha = %d °, tap time = %d ms" % (alpha, tap_time))

    return dx, dy, tap_time


def pick_random_level_number():
    return np.random.randint(TOTAL_LEVEL_NUMBER) + 1
