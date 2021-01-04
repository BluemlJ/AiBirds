import keras
import time

import tensorflow as tf

from src.utils.utils import *
from src.utils.mem import ReplayMemory
from src.agents.nns import *

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

gpus = tf.config.experimental.list_physical_devices('GPU')
memory_config = [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=2048)]
tf.config.experimental.set_virtual_device_configuration(gpus[0], memory_config)

# Miscellaneous
PLOT_PERIOD = 2000  # number of transitions between each learning statistics plot
PRINT_STATS_PERIOD = 100
MA_WINDOW_SIZE = 10000
CHECKPOINT_SAVE_PERIOD = 5000
TEST_PERIOD = 2000


class TFDQNAgent:
    """Deep Q-Network (DQN) agent for playing Tetris, Snake or other games"""

    def __init__(self, env, num_parallel_envs, name, use_dueling=True, use_double=True,
                 latent_dim=64, latent_a_dim=64, latent_v_dim=64, latent_depth=1,
                 learning_rate=0.0001, obs_buf_size=100, exp_buf_size=10000,
                 use_pretrained=False, override=False, **kwargs):
        """
        :param env: The environment in which the agent acts
        :param num_parallel_envs: The number of environments which are executed simultaneously
        :param name: A string identifying the agent in file names
        :param use_dueling: If True the Dueling Networks extension is used
        :param use_double: If True the Double Q-Learning extension is used
        :param latent_dim: Width of the main latent layer (first layer after conv layers)
        :param latent_a_dim: Width of the latent layer of the advantage network from Dueling Networks
        :param latent_v_dim: Width of the latent layer of the state-value network from Dueling Networks
        :param learning_rate: The higher the faster but more unstable the agent learns
        :param kwargs: Arguments for the used environment (e.g., height and width)
        """

        self.name = name
        self.num_par_envs = num_parallel_envs

        self.env_type = env
        self.env = None
        self.env_args = kwargs
        self.image_state_shape = None
        self.numerical_state_shape = None
        self.num_actions = None

        self.setup_env(num_parallel_envs)

        self.out_path = self.setup_out_path(override)

        self.stats = Statistics(env)  # Information collected over the course of training

        # Set wanted features
        self.dueling = use_dueling
        self.double = use_double

        # Training hyperparameters
        self.latent_dim = latent_dim
        self.latent_a_dim = latent_a_dim
        self.latent_v_dim = latent_v_dim
        self.latent_depth = latent_depth
        self._optimizer = tf.optimizers.Adam(learning_rate=learning_rate)
        self.training_loss_fn = keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE)
        self.training_loss_metric = keras.metrics.MeanSquaredError()

        # Initialize the architecture of the acting and learning part of the DQN (theta)
        self.inputs, latent = get_input_model(self.env, self.latent_dim, self.latent_depth)
        self.outputs = None
        self.online_network = self._build_compile_model(latent)
        if use_pretrained:
            self.load_pretrained_model()
        self.target_network = None
        self.init_target_network()

        self.obs_buf_size = obs_buf_size
        self.exp_buf_size = exp_buf_size

        self.memory = None

        hyperparams_to_json(self.out_path, num_parallel_envs, use_dueling, use_double, learning_rate, latent_dim,
                            latent_a_dim, latent_v_dim, obs_buf_size, exp_buf_size)

        print('DQN agent initialized.')

    def _build_compile_model(self, latent):
        # Implementation of the Dueling Network principle
        if self.dueling:
            # State value prediction
            latent_v = Dense(self.latent_v_dim, name='latent_V', activation="relu")(latent)
            state_value = Dense(1, name='V')(latent_v)

            # Advantage prediction
            latent_a = Dense(self.latent_a_dim, name='latent_A', activation="relu")(latent)
            advantage = Dense(self.num_actions, name='A')(latent_a)

            # Q(s, a) = V(s) + A(s, a) - A_mean(s, a)
            q_values = tf.add(state_value,
                              tf.subtract(advantage, tf.reduce_mean(advantage, axis=1,
                                                                    keepdims=True,
                                                                    name='A_mean'),
                                          name='Sub'),
                              name='Q')
        else:
            # Direct Q-value prediction
            latent_feature_2 = tf.keras.layers.Dense(128, name='latent_2')(latent)
            lrelu_feature = LeakyReLU()(latent_feature_2)
            q_values = Dense(self.num_actions, name='Q')(lrelu_feature)

        self.outputs = [q_values]

        model = tf.keras.Model(inputs=self.inputs, outputs=self.outputs)
        model.compile(loss='huber_loss', optimizer=self._optimizer)

        model.summary()
        # the following needs separate GraphViz installation from https://graphviz.gitlab.io/download/
        # this helped for GraphViz bugfix: https://datascience.stackexchange.com/questions/74500
        tf.keras.utils.plot_model(model, to_file=self.out_path + 'model_plot.png', show_shapes=True,
                                  show_layer_names=True)
        return model

    def init_target_network(self):
        if self.double:
            # Initialize the architecture of a separate, shadowed (target) version of the DQN (theta-)
            model = tf.keras.Model(inputs=self.inputs, outputs=self.outputs)
            model.compile(loss='huber_loss', optimizer=self._optimizer)
            self.target_network = model
            self.target_network.set_weights(self.online_network.get_weights())
        else:
            self.target_network = self.online_network

    def load_pretrained_model(self):
        pretrained_path = "out/" + self.env_type.NAME + "/pretrained"
        if not os.path.exists(pretrained_path):
            Exception("You specified to load a pretrained model. However, there is no pretrained model "
                      "at '%s'." % pretrained_path)

        self.online_network.load_weights(pretrained_path + "/pretrained", by_name=True)

    def setup_env(self, num_envs):
        if self.env is not None:
            del self.env
        self.env = self.env_type(**self.env_args, num_par_envs=num_envs)
        self.image_state_shape, self.numerical_state_shape = self.env.get_state_shapes()
        self.num_actions = self.env.get_number_of_actions()

    def practice(self, num_parallel_steps,
                 replay_period, replay_size,
                 batch_size, replay_epochs,
                 sync_period, gamma,
                 epsilon, epsilon_decay_mode, epsilon_decay_rate, epsilon_min,
                 delta, delta_anneal,
                 alpha,
                 verbose=False):
        """The agent's main training routine.

        :param num_parallel_steps: Number of (parallel) transitions to play
        :param replay_period: number of levels between each training of the online network
        :param replay_size: Number of transitions to be learned from when learn() is invoked
        :param batch_size: Size of training batches used in the GPU/CPU to fit the model
        :param replay_epochs: Number of epochs per replay
        :param sync_period: The number of levels between each synchronization of online and target network. The
                            higher the number, the stronger Double Q-Learning and the less overestimation.
                            sync_period == 1 means "Double Q-Learning off"
        :param gamma: Discount factor
        :param epsilon: Probability for random shot (epsilon greedy policy)
        :param epsilon_decay_mode: The function used to decrease epsilon after each parallel step ("lin" or "exp")
        :param epsilon_decay_rate: Decrease factor for epsilon
        :param epsilon_min: Minimum value for epsilon which is not undercut over thr course of practicing
        :param delta: Trade-off factor between Monte Carlo target return and one-step target return,
                      delta = 1 means MC return, delta = 0 means one-step return
        :param delta_anneal: Decrease factor for delta
        :param alpha: For Prioritized Experience Replay: the larger alpha the stronger prioritization
        :param verbose:
        """

        epsilon = Epsilon(value=epsilon, decay_mode=epsilon_decay_mode,
                          decay_rate=epsilon_decay_rate, minimum=epsilon_min)

        # Save all hyperparameters in txt file
        self.write_hyperparams_file(num_parallel_steps, replay_period, replay_size, sync_period, epsilon, alpha,
                                    gamma, delta, delta_anneal)

        logger = Logger(self.out_path)

        # Initialize the memory where all the experience will be memorized
        self.memory = ReplayMemory(memory_size=self.exp_buf_size,
                                   image_state_shape=self.image_state_shape,
                                   numerical_state_shape=self.numerical_state_shape)

        # Initialize observations buffer with fixed size
        obs = Observations(self.obs_buf_size, self.num_par_envs, self.image_state_shape, self.numerical_state_shape)

        # Reset all environments
        self.env.reset()

        print("DQN agent starts practicing...")

        total_timer = time.time()
        comp_timer = total_timer

        for i in range(1, num_parallel_steps + 1):
            states = self.get_states()

            # Predict the next action to take (move, rotate or do nothing)
            actions, _ = self.plan(states, epsilon.get_value(), batch_size)

            # Perform actions, observe new environment state, level score and application state
            rewards, scores, game_overs, times, wins = self.env.step(actions)

            # Save observations
            obs.save_observations(states, actions, scores, rewards, times)

            # Handle finished envs
            fin_env_ids = np.where(game_overs)[0]

            if len(fin_env_ids):
                # For all finished episodes, save their observations in the replay memory
                for env_id in fin_env_ids:
                    obs_states, obs_actions, obs_score_gains, obs_rewards, obs_times = obs.get_observations(env_id)
                    self.memory.memorize(obs_states, obs_actions, obs_score_gains, obs_rewards, gamma)

                    obs_score, obs_return = obs.get_performance(env_id)
                    new_return_record = self.stats.denote_stats(obs_return, obs_score, obs_times[-1], wins[env_id])

                    if new_return_record:
                        transition = self.memory.get_trans_text(self.memory.get_length() - 1, self.env)
                        logger.log_new_record(obs_return, transition)

                # Reset all finished envs and update their corresponding current variables
                self.env.reset_for(fin_env_ids)
                obs.begin_new_episode_for(fin_env_ids)

            # Every X episodes, plot informative graphs
            if i % PLOT_PERIOD == 0:
                self.stats.plot_stats(self.out_path, self.memory)

            # If environment has test levels, test on it
            if i % TEST_PERIOD == 0 and self.env.has_levels():
                self.test_on_levels()

            # Update the network weights every train_period levels to fit experience
            if i % replay_period == 0 and self.memory.get_length() >= replay_size:
                self.learn(replay_size, gamma=gamma, delta=delta, alpha=alpha,
                           batch_size=batch_size, epochs=replay_epochs, logger=logger, verbose=verbose)

                # Decrease learning rate if loss changed too little
                """if self.stats.loss_stagnates() and self._optimizer.lr != 0.00001:
                    self._optimizer.lr.assign(0.00001)
                    log_text = "Learning rate decreased in train cycle %d (episode %d).\n" % \
                               (i // replay_period, self.stats.get_length())
                    logger.log(log_text)"""

            # Save model checkpoint
            if i % CHECKPOINT_SAVE_PERIOD == 0:
                self.save(overwrite=True, checkpoint=True, checkpoint_no=self.stats.get_length())

            # Cut off old experience to reduce buffer load
            if self.memory.get_length() > 0.95 * self.exp_buf_size:
                self.memory.delete_first(n=int(0.2 * self.exp_buf_size))

            # Synchronize target and online network every sync_period levels
            if self.double and i % sync_period == 0:
                self.target_network.set_weights(self.online_network.get_weights())

            if i % PRINT_STATS_PERIOD == 0:
                self.stats.print_stats(i, num_parallel_steps, self.num_par_envs, time.time() - comp_timer,
                                       time.time() - total_timer, PRINT_STATS_PERIOD, epsilon, logger,
                                       MA_WINDOW_SIZE)
                comp_timer = time.time()

            # Cool down
            epsilon.decay()  # reduces randomness (less explore, more exploit)
            delta *= delta_anneal  # shifts target return fom MC to one-step

        self.save()
        logger.close()

        print("Practicing finished successfully!")

    def learn(self, num_instances, gamma, delta, logger, batch_size=1024, epochs=1,
              alpha=0.7, beta=0.5, verbose=False):
        """Updates the online network's weights. This is the actual learning procedure of the agent.

        :param num_instances: Number of transitions to be learned from
        :param gamma: Discount factor
        :param delta: Trade-off factor between Monte Carlo target return and one-step target return,
                      delta = 1 means MC return, delta = 0 means one-step return
        :param logger:
        :param batch_size: Number of transitions per batch during learning
        :param epochs:
        :param alpha: For Prioritized Experience Replay: the larger alpha the stronger prioritization
        :param beta: ?
        :param verbose:
        :return:
        """

        # Obtain a list of useful transitions to learn on
        trans_ids, probabilities = self.memory.recall(num_instances, alpha)

        # Obtain total number of experienced transitions
        exp_len = self.memory.get_length()

        # Compute importance-sampling weights and normalize
        weights = (exp_len * probabilities[trans_ids]) ** (- beta)
        weights /= np.max(weights)

        # Get list of transitions
        states, actions, rewards, next_states, terminals = \
            self.memory.get_transitions(trans_ids)

        # Obtain Monte Carlo return for each transition
        mc_returns = self.memory.get_mc_returns(trans_ids, gamma)

        # Predict returns (i.e. values V(s)) for all states s
        pred_returns = np.max(self.online_network.predict(states, batch_size=batch_size), axis=1)

        # Predict next returns
        pred_next_returns = np.max(self.target_network.predict(next_states, batch_size=batch_size), axis=1)

        # Compute one-step return
        one_step_returns = rewards + gamma * pred_next_returns

        # Compute convex combination of MC return and one-step return
        target_returns = delta * mc_returns + (1 - delta) * one_step_returns

        # Set target return = reward for all terminal transitions
        target_returns[terminals] = rewards[terminals]

        # Compute Temporal Difference (TD) errors (the "surprise" of the agent)
        td_errs = target_returns - pred_returns

        # Update transition priorities according to TD errors
        self.memory.set_priorities(trans_ids, np.abs(td_errs))

        # Prepare inputs and targets for fitting
        inputs = states
        targets = self.target_network.predict(states, batch_size=batch_size)
        targets[range(len(trans_ids)), actions] = target_returns
        sample_weight = np.abs(np.multiply(weights, td_errs))

        # Update the online network's weights
        loss, individual_losses, predictions = self.fit(inputs, targets, epochs=epochs, verbose=verbose,
                                                        batch_size=batch_size, sample_weight=sample_weight)

        self.stats.denote_loss(loss)
        self.stats.log_extreme_losses(individual_losses, trans_ids, predictions, targets,
                                      self.memory, self.env, logger)

        return loss

    def fit(self, x, y, epochs, batch_size, sample_weight, verbose=False):
        start = time.time()

        # Prepare the training dataset
        x_image, x_numerical = x
        train_dataset = tf.data.Dataset.from_tensor_slices((x_image, x_numerical, y, sample_weight))
        train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size)
        train_loss = 0
        individual_losses = np.array([])
        predictions = np.empty((0, self.env.get_number_of_actions()))

        for epoch in range(epochs):
            if verbose:
                print("\rEpoch %d/%d - Batch 0/%d - Loss: --" %
                      ((epoch + 1), epochs, len(train_dataset)), flush=True, end="")

            # Iterate over the batches of the dataset.
            for step, (x_image_batch, x_numerical_batch, y_batch, sample_weight_batch) in enumerate(train_dataset):
                # Train on this batch
                batch_individual_losses, batch_out = self.train_step(x_image_batch, x_numerical_batch, y_batch,
                                                                     sample_weight_batch)
                individual_losses = np.append(individual_losses, batch_individual_losses)
                predictions = np.append(predictions, batch_out, axis=0)

                train_loss = self.training_loss_metric.result().numpy()

                if verbose:
                    print("\rEpoch %d/%d - Batch %d/%d - Total loss: %.4f" %
                          ((epoch + 1), epochs, (step + 1), len(train_dataset), float(train_loss)),
                          flush=True, end="")

            self.training_loss_metric.reset_states()
            if verbose:
                print("")

        if verbose:
            print("Fitting took %.2f s." % (time.time() - start))

        return train_loss, individual_losses, predictions

    @tf.function
    def train_step(self, x_image, x_numerical, y, sample_weight):
        with tf.GradientTape() as tape:
            out = self.online_network([x_image, x_numerical], training=True)
            individual_losses = self.training_loss_fn(y, out, sample_weight=sample_weight)
            cumulated_loss = tf.reduce_mean(individual_losses)

        grads = tape.gradient(cumulated_loss, self.online_network.trainable_weights)

        # Run one step of gradient descent by the optimizer
        self._optimizer.apply_gradients(zip(grads, self.online_network.trainable_weights))

        self.training_loss_metric.update_state(y, out)

        return individual_losses, out

    def learn_entire_experience(self, batch_size, epochs, gamma, delta, alpha):
        experience_length = self.memory.get_length()
        self.learn(experience_length, gamma, delta, batch_size, epochs, alpha)

    def test_performance(self, num_levels=100):
        """Perform validation of the agent on the validation set. On all validation levels,
        the agent plays without epsilon and does not learn from the experience."""

        # Initialize return list (list of normalized final scores for each level played)
        returns = []
        scores = []

        # Play levels and save observations
        print("Start performance test...")
        for i in range(num_levels):
            self.env.reset()
            ret, score = self.play_level(epsilon=None)
            returns += [ret]
            scores += [score]
        print("Finished performance test.")

        return_avg = np.average(returns)
        score_avg = np.average(scores)
        print("Return average:", return_avg)
        print("Score average:", score_avg)

        return return_avg, score_avg

    def plan(self, states, epsilon=None, batch_size=1024):
        """Epsilon greedy policy. With a probability of epsilon, a random action is returned. Else,
        the agent predicts the best shot for the given state.

        :param states: List of state matrices
        :param epsilon: If given, epsilon greedy policy will be applied, otherwise the agent plans optimally
        :param batch_size: Batch size when states are fed into NN for prediction
        :return: action: An index number, corresponding to an action
        """

        # Do epsilon-greedy
        if epsilon is not None and np.random.random(1) < epsilon:
            # Choose action by random
            actions = np.random.randint(self.num_actions, size=self.num_par_envs)

            pred_ret = 0
        else:
            # Obtain list of action-values Q(s,a)
            q_vals = self.online_network.predict(states, batch_size=batch_size)

            pred_ret = np.max(q_vals, axis=1)[0]

            # Determine optimal action
            actions = q_vals.argmax(axis=1)

        return actions, pred_ret

    def get_states(self):
        """Fetches the current game grid."""
        return self.env.get_states()

    def just_play(self, episodes=9999999, epsilon=None, verbose=False):
        print("Just playing around...")

        if self.num_par_envs != 1:
            self.setup_env(1)
            self.num_par_envs = 1

        for i in range(episodes):
            self.env.reset()
            self.env.render()
            self.play_level(epsilon=epsilon, render_environment=True, verbose=verbose)

    def play_level(self, epsilon, render_environment=True, verbose=False):
        ret, score, env_time = 0, 0, 0

        if verbose:
            print("Current ret | Predicted ret | Performed action")
            print("----------------------------------------------")

        # Play a whole level
        game_over = False
        while not game_over:
            state = self.env.get_states()

            # Predict the next action to take (move, rotate or do nothing)
            action, pred_ret = self.plan(state, epsilon)

            if verbose:
                print("{:>11.2f}".format(ret) + " | " +
                      "{:>13.2f}".format(pred_ret) + " | " +
                      "{:>16s}".format(self.env.actions[action[0]]))

            # Perform action, observe new environment state, level score and application state
            reward, score, game_over, env_time, _ = self.env.step(action)

            if render_environment:
                self.env.render()

            ret += reward[0]

        if verbose:
            print("----------------------------------------------")
            print("Level finished with return %.2f, score %d, and time %d.\n" % (ret, score, env_time))

        return ret, score

    def test_on_levels(self, render=False):
        test_env = self.env_type(1)
        test_env.set_mode(test_env.TEST_MODE)
        num_levels = len(test_env.levels_list)
        test_scores = np.zeros(num_levels)

        for level in range(num_levels):
            # Play a whole level
            test_env.reset(lvl_no=level)
            if render:
                test_env.render()
                time.sleep(0.35)

            score = 0
            game_over = False
            while not game_over:
                state = test_env.get_states()

                # Predict the next action to take (move, rotate or do nothing)
                action, _ = self.plan(state, 0)

                # Perform action, observe new environment state, level score and application state
                _, score, game_over, _, _ = test_env.step(action)

                if render:
                    test_env.render()

            test_scores[level] = score

        _, highscores_human = test_env.get_highscores()
        plot_highscores(test_scores, highscores_human, self.out_path)

    def save(self, out_path=None, overwrite=False, checkpoint=False, checkpoint_no=None):
        """Saves the current model weights and statistics to a specified export path."""
        if out_path is None:
            model_path = self.out_path + ("checkpoints/" if checkpoint else "trained_model")
        else:
            model_path = out_path

        if checkpoint_no is not None:
            model_path += "%d" % checkpoint_no

        self.online_network.save_weights(model_path, overwrite=overwrite)

        self.stats.save(self.out_path)

        print("Saved model and statistics.")

    def restore(self, model_name, checkpoint_no=None):
        """Restores model weights and statistics."""
        in_path = "out/%s/%s/" % (self.env.NAME, model_name)

        if checkpoint_no is not None:
            model_path = in_path + "checkpoints/%s" % str(checkpoint_no)
            self.restore_from(model_path, in_path)
        else:
            self.restore_latest(model_name)

    def restore_latest(self, model_name):
        """Restores the most recently saved model (can be a checkpoint or a final model)."""
        in_path = "out/%s/%s/" % (self.env.NAME, model_name)
        model_path = in_path + "trained_model"
        chkpt_dir_path = in_path + "checkpoints/"
        if os.path.exists(model_path + ".index"):
            # Restore final model
            self.restore_from(model_path, in_path)
        else:
            # Restore latest checkpoint
            chkpt_path = tf.train.latest_checkpoint(chkpt_dir_path)
            self.restore_from(chkpt_path, in_path)

    def restore_from(self, model_path, stats_path):
        print("Restoring model from '%s'." % model_path)
        self.online_network.load_weights(model_path)
        self.init_target_network()
        self.stats.load(in_path=stats_path)

    def save_experience(self, experience_path=None, overwrite=False, compress=False):
        if experience_path is None:
            experience_path = "data/%d_%s.hdf5" % (self.memory.get_length(), self.name)
        self.memory.export_experience(experience_path, overwrite, compress)

    def restore_experience(self, experience_path=None, gamma=None):
        if experience_path is None:
            experience_path = "data/" + self.name
        self.memory.import_experience(experience_path, gamma)

    def forget(self):
        self.memory = ReplayMemory(memory_size=self.exp_buf_size,
                                   image_state_shape=self.image_state_shape,
                                   numerical_state_shape=self.numerical_state_shape)

    def print_transitions(self, ids):
        self.memory.print_trans_from(ids, self.env)

    def write_hyperparams_file(self, num_parallel_steps, replay_period, replay_size, sync_period, epsilon, alpha,
                               gamma, delta, delta_anneal):
        hyperparams_file = open(self.out_path + "hyperparams.txt", "w+")
        metadata = "num_parallel_steps: %d\n" % num_parallel_steps + \
                   "num_parallel_envs: %d\n" % self.num_par_envs + \
                   "replay_period: %d\n" % replay_period + \
                   "replay_size: %d\n" % replay_size + \
                   "sync_period: %d\n\n" % (sync_period if self.double else -1) + \
                   "epsilon: %f\n" % epsilon.get_value() + \
                   "epsilon_anneal: %f\n" % epsilon.get_decay() + \
                   "epsilon_min: %f\n\n" % epsilon.get_minimum() + \
                   "alpha: %f\n" % alpha + \
                   "gamma: %f\n" % gamma + \
                   "delta: %f\n" % delta + \
                   "delta_anneal: %f\n\n" % delta_anneal + \
                   "learning_rate: %f\n" % self._optimizer.get_config()['learning_rate'] + \
                   "latent_dim: %d\n" % self.latent_dim + \
                   "latent_a_dim: %d\n" % self.latent_a_dim + \
                   "latent_v_dim: %d\n\n" % self.latent_v_dim + \
                   "obs_buf_size: %d\n" % self.obs_buf_size + \
                   "exp_buf_size: %d\n\n" % self.exp_buf_size
        hyperparams_file.write(metadata)
        self.online_network.summary(print_fn=lambda x: hyperparams_file.write(x + '\n'))
        hyperparams_file.close()

    def setup_out_path(self, override=False):
        out_path = "out/%s/%s/" % (self.env.NAME, self.name)
        if not override:
            check_for_existing_model(out_path)
        os.makedirs(out_path, exist_ok=True)
        return out_path


def load_model(model_name, env, checkpoint_no=None):
    in_path = "out/" + env.NAME + "/" + model_name + "/"
    with open(in_path + "hyperparams.json") as infile:
        hyperparams_dict = json.load(infile)

    agent = TFDQNAgent(env=env, name="tmp", override=True, **hyperparams_dict)
    agent.restore(model_name, checkpoint_no)
    return agent


def load_and_play(model_name, env, checkpoint_no=None, mode=None):
    agent = load_model(model_name, env, checkpoint_no)
    if mode is not None:
        agent.env.set_mode(mode)
    agent.just_play(verbose=True)


def load_and_test(model_name, env, checkpoint_no=None, mode=None, render=False):
    agent = load_model(model_name, env, checkpoint_no)
    if mode is not None:
        agent.env.set_mode(mode)
    agent.test_on_levels(render)
