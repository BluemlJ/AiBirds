import time
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2"  # let TF only print errors
import tensorflow as tf
import numpy as np

from src.utils import ReplayMemory, Statistics, Epsilon, LearningRate, Observations, \
    copy_model, plot_highscores, config2text, check_for_existing_model, config2json, json2config
from src.agents.comp import *

os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # make GPU visible

gpus = tf.config.list_physical_devices('GPU')
memory_config = [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4096)]
tf.config.experimental.set_virtual_device_configuration(gpus[0], memory_config)

# Miscellaneous
PLOT_PERIOD = 5000  # number of transitions between each learning statistics plot
PRINT_STATS_PERIOD = 100
CHECKPOINT_SAVE_PERIOD = 5000
TEST_PERIOD = 2000


class Agent:
    """Deep Q-Network (DQN) agent for playing Tetris, Snake or other games"""

    def __init__(self, env_type, stem_model: StemModel, q_network: QNetwork,
                 name, num_parallel_envs, replay_batch_size, sequence_shift=None, eta=0.9,
                 use_double=True,
                 obs_buf_size=100, mem_size=10000,
                 use_pretrained=False, override=False, **kwargs):
        """Constructor
        :param env_type: The environment in which the agent acts
        :param stem_model: The main stem model
        :param q_network: The Q-network (coming after the stem model)
        :param num_parallel_envs: The number of environments which are executed simultaneously
        :param name: A string identifying the agent in file names
        :param use_double: If True the Double Q-Learning extension is used
        :param kwargs: Arguments for the used environment (e.g., height and width)
        """

        print("Initializing DQN agent...")

        # General
        self.name = name

        # Environment
        self.num_par_envs = num_parallel_envs
        self.env_type = env_type
        self.env, self.state_shape_2d, self.state_shape_1d, self.num_actions = self.init_env(kwargs)

        self.out_path = self.setup_out_path(override)

        # For training
        self.replay_batch_size = replay_batch_size
        self.learning_rate = None
        self.epsilon = None

        # Model architecture
        self.sequential = stem_model.sequential
        self.sequence_len = stem_model.sequence_len
        self.stem_model = stem_model
        self.q_network = q_network
        self.optimizer = tf.optimizers.Adam()
        self.online_learner = self.init_online_learner()

        # Double Q-Learning
        self.double = use_double
        if use_pretrained:
            self.load_pretrained_model()
        self.target_learner = self.init_target_learner()

        # Distributed RL
        self.actor = self.init_actor()

        # Training loss
        self.training_loss_fn = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE)
        self.training_loss_metric = tf.keras.metrics.MeanSquaredError()

        # Buffers, memory, and statistics
        self.obs_buf_size = obs_buf_size
        self.mem_size = mem_size
        self.memory = ReplayMemory(memory_size=self.mem_size,
                                   state_shape_2d=self.state_shape_2d,
                                   state_shape_1d=self.state_shape_1d,
                                   hidden_state_shape=self.stem_model.get_hidden_state_shape(),
                                   sequence_len=self.sequence_len,
                                   sequence_shift=sequence_shift,
                                   eta=eta)
        self.stats = Statistics(self.env_type, self.env, self.memory, log_path=self.out_path)

        self.save_config()
        print('DQN agent initialized.')

    def init_online_learner(self):
        model = DQN(self.stem_model, self.q_network, (self.state_shape_2d, self.state_shape_1d),
                    self.num_actions, self.replay_batch_size)
        model.compile(loss='huber_loss', optimizer=self.optimizer)

        model.summary()

        # the following needs separate GraphViz installation from https://graphviz.gitlab.io/download/
        # this helped for GraphViz bugfix: https://datascience.stackexchange.com/questions/74500
        tf.keras.utils.plot_model(model, to_file=self.out_path + 'model_plot.png', show_shapes=True,
                                  show_layer_names=True, expand_nested=True, dpi=400)
        return model

    def init_target_learner(self):
        if self.double:
            return self.copy_dqn(batch_size=self.replay_batch_size)
        else:
            return self.online_learner

    def init_actor(self):
        return self.copy_dqn(batch_size=self.num_par_envs)

    def copy_dqn(self, batch_size):
        stem_model = copy_model(self.stem_model)
        q_network = copy_model(self.q_network)
        model = DQN(stem_model, q_network, (self.state_shape_2d, self.state_shape_1d),
                    self.num_actions, batch_size)
        model.compile(loss='huber_loss', optimizer=self.optimizer)
        model.set_weights(self.online_learner.get_weights())
        return model

    def init_env(self, env_args):
        env = self.env_type(**env_args, num_par_envs=self.num_par_envs)
        state_shape_2d, state_shape_1d = env.get_state_shapes()
        num_actions = env.get_number_of_actions()
        return env, state_shape_2d, state_shape_1d, num_actions

    def load_pretrained_model(self):
        pretrained_path = "out/" + self.env_type.NAME + "/pretrained"
        if not os.path.exists(pretrained_path):
            Exception("You specified to load a pretrained model. However, there is no pretrained model "
                      "at '%s'." % pretrained_path)

        self.online_learner.load_weights(pretrained_path + "/pretrained", by_name=True)

    def reinit_env(self, num_envs, **kwargs):
        if self.env is not None:
            del self.env
        self.num_par_envs = num_envs
        self.env, self.state_shape_2d, self.state_shape_1d, self.num_actions = self.init_env(kwargs)

    def practice(self, num_parallel_steps,
                 replay_period, replay_size_multiplier, replay_epochs,
                 learning_rate: LearningRate,
                 target_sync_period, actor_sync_period,
                 gamma, epsilon: Epsilon,
                 delta, delta_anneal,
                 alpha,
                 verbose=False):
        """The agent's main training routine.

        :param num_parallel_steps: Number of (parallel) transitions to play
        :param replay_period: Number of parallel steps between each training of the online network
        :param replay_size_multiplier: Factor determining the number of transitions to be learned from each
                   train cycle (the replay size). Each time, the replay size is determined as
                   follows: replay_size = replay_size_multiplier * new_transitions
        :param replay_epochs: Number of epochs per replay
        :param learning_rate: (dynamic) learning rate used for training
        :param target_sync_period: The number of levels between each synchronization of online and target network. The
                   higher the number, the stronger Double Q-Learning and the less overestimation.
                   sync_period == 1 means "Double Q-Learning off"
        :param actor_sync_period: The number of levels between each synchronization of learner and actor.
        :param gamma: Discount factor
        :param epsilon: Epsilon class, probability for random shot (epsilon greedy policy)
        :param delta: Trade-off factor between Monte Carlo target return and one-step target return,
                   delta = 1 means MC return, delta = 0 means one-step return
        :param delta_anneal: Decrease factor for delta
        :param alpha: For Prioritized Experience Replay: the larger alpha the stronger prioritization
        :param verbose:
        """

        self.learning_rate = learning_rate
        self.epsilon = epsilon

        # Save all hyperparameters in txt file
        self.write_hyperparams_file(num_parallel_steps, replay_period, replay_size_multiplier, target_sync_period,
                                    alpha, gamma, delta, delta_anneal)

        # Initialize observations buffer with fixed size
        obs = Observations(self.obs_buf_size, self.num_par_envs,
                           self.state_shape_2d, self.state_shape_1d,
                           self.stem_model.get_hidden_state_shape())

        new_transitions = 0

        # Reset all environments
        self.env.reset()

        print("DQN agent starts practicing...")

        self.stats.start_timer()

        for i in range(1, num_parallel_steps + 1):
            states = self.get_states()

            hidden_states = self.actor.stem_model.get_cell_states()

            # Predict the next action to take (move, rotate or do nothing)
            actions, _ = self.plan(states, epsilon.get_value())

            # Perform actions, observe new environment state, level score and application state
            rewards, scores, game_overs, times, wins = self.env.step(actions)

            # Save observations
            obs.save_observations(states, hidden_states, actions, scores, rewards, times)

            # Handle finished envs
            fin_env_ids = np.where(game_overs)[0]

            if len(fin_env_ids):
                # For all finished episodes, save their observations in the replay memory
                for env_id in fin_env_ids:
                    obs_states, obs_hidden_states, obs_actions, obs_score_gains, obs_rewards, obs_times \
                        = obs.get_observations(env_id)
                    self.memory.memorize(obs_states, obs_hidden_states, obs_actions, obs_score_gains, obs_rewards,
                                         gamma)

                    obs_score, obs_return = obs.get_performance(env_id)
                    self.stats.denote_episode_stats(obs_return, obs_score, obs_times[-1], wins[env_id])

                    new_transitions += len(obs_rewards)

                # Reset all finished envs and update their corresponding current variables
                self.env.reset_for(fin_env_ids)
                obs.begin_new_episode_for(fin_env_ids)

                # Reset actor's LSTM states (if any) to zero
                self.actor.stem_model.reset_cell_states_for(fin_env_ids)

            # Every X episodes, plot informative graphs
            if i % PLOT_PERIOD == 0:
                self.stats.plot_stats(self.out_path)

            # If environment has test levels, test on it
            if i % TEST_PERIOD == 0 and self.env.has_test_levels():
                self.test_on_levels()

            # Update the network weights every train_period levels to fit experience
            if i % replay_period == 0:  # and self.memory.get_length() >= replay_size:
                replay_size = replay_size_multiplier * new_transitions
                learned_trans = self.learn(replay_size, gamma, delta, epochs=replay_epochs,
                                           alpha=alpha, verbose=verbose)
                new_transitions = max(0, new_transitions - learned_trans)

            # Save model checkpoint
            if i % CHECKPOINT_SAVE_PERIOD == 0:
                self.save(overwrite=True, checkpoint=True, checkpoint_no=self.stats.get_num_episodes())

            # Cut off old experience to reduce buffer load
            if self.memory.get_length() > 0.95 * self.mem_size:
                self.memory.delete_first(n=int(0.2 * self.mem_size))

            # Synchronize target and online network every sync_period levels
            if self.double and i % target_sync_period == 0:
                self.target_learner.set_weights(self.online_learner.get_weights())

            # Synchronize learner and actor
            if i % actor_sync_period == 0:
                self.actor.set_weights(self.online_learner.get_weights())

            if i % PRINT_STATS_PERIOD == 0:
                self.stats.print_stats(i, num_parallel_steps, self.num_par_envs,
                                       PRINT_STATS_PERIOD, epsilon)

            # Cool down
            epsilon.decay()  # reduces randomness (less explore, more exploit)
            delta *= delta_anneal  # shifts target return fom MC to one-step

        self.save()
        self.stats.logger.close()

        print("Practicing finished successfully!")

    def learn(self, replay_size, gamma, delta, epochs=1, alpha=0.7, verbose=False):
        """Updates the online network's weights. This is the actual learning procedure of the agent.

        :param replay_size: Number of transitions to be learned from
        :param gamma: Discount factor
        :param delta: Trade-off factor between Monte Carlo target return and one-step target return,
                      delta = 1 means MC return, delta = 0 means one-step return
        :param epochs:
        :param alpha: For Prioritized Experience Replay: the larger alpha the stronger prioritization
        :param verbose:
        :return: number of transitions learned
        """

        if replay_size == 0:
            return 0

        if not self.sequential:
            return self.learn_instances(replay_size, gamma=gamma, delta=delta, alpha=alpha,
                                        epochs=epochs, verbose=verbose)
        else:
            num_sequences = replay_size // self.sequence_len
            if num_sequences >= self.replay_batch_size:
                return self.learn_sequences(num_sequences, gamma=gamma, alpha=alpha,
                                            epochs=epochs, verbose=verbose)
            else:
                return 0

    def learn_instances(self, num_instances, gamma, delta, epochs=1, alpha=0.7, verbose=False):
        """Uses batches of single instances to learn on."""

        # Obtain a list of useful transitions to learn on
        trans_ids, probabilities = self.memory.recall_single_transitions(num_instances, alpha)

        # Obtain total number of experienced transitions
        exp_len = self.memory.get_length()

        # Get list of transitions
        states, hidden_states, actions, rewards, next_states, last_hidden_states, terminals = \
            self.memory.get_transitions(trans_ids)

        # Obtain Monte Carlo return for each transition
        mc_returns = self.memory.get_mc_returns(trans_ids)

        # Predict returns (i.e. values V(s)) for all states s
        q_vals = self.online_learner.predict(states, batch_size=self.replay_batch_size)
        pred_returns = np.max(q_vals, axis=1)

        # Predict next returns
        next_q_vals = self.target_learner.predict(next_states, batch_size=self.replay_batch_size)
        pred_next_returns = np.max(next_q_vals, axis=1)

        # Compute target_returns and temporal difference (TD) errors
        target_returns, td_errs = compute_target_returns(pred_returns, pred_next_returns, rewards, gamma,
                                                         delta, mc_returns, terminals)

        # Update transition priorities according to TD errors
        self.memory.set_priorities(trans_ids, np.abs(td_errs))

        # Prepare inputs and targets for fitting
        inputs = states
        targets = self.target_learner.predict(states, batch_size=self.replay_batch_size)
        targets[range(len(trans_ids)), actions] = target_returns

        # Compute sample weights
        sample_weights = compute_sample_weights(td_errs, probabilities[trans_ids], exp_len)

        # Update learning rate
        self.update_lr()

        # Update the online network's weights
        loss, individual_losses, predictions = self.fit(inputs, targets, epochs=epochs, verbose=verbose,
                                                        batch_size=self.replay_batch_size,
                                                        sample_weights=sample_weights)

        self.stats.denote_learning_stats(loss, individual_losses, self.optimizer.learning_rate.numpy(),
                                         trans_ids, predictions, targets, self.env)

        return len(trans_ids)

    def learn_sequences(self, num_sequences, gamma, epochs=1, alpha=0.7, verbose=False):
        """Uses batches of sequences to learn on."""

        # Obtain a list of useful sequences to learn on
        seq_ids, probabilities = self.memory.recall_sequences(num_sequences, alpha, batch_size=self.replay_batch_size)

        if len(seq_ids) == 0:
            return 0

        seq_num = self.memory.get_num_sequences()

        # Get sequences of transitions
        trans_ids, (states, first_hidden_states, actions, rewards, next_states, last_hidden_states, terminals), mask \
            = self.memory.get_sequences(seq_ids)

        # Predict returns (i.e. values V(s)) for all states s
        q_vals = self.online_learner.set_hidden_and_predict(first_hidden_states, states)
        pred_returns = np.max(q_vals, axis=2)

        # Predict next returns
        next_states_2d, next_states_1d = next_states
        last_states = [next_states_2d[:, np.newaxis, -1], next_states_1d[:, np.newaxis, -1]]
        next_q_vals = self.target_learner.set_hidden_and_predict(last_hidden_states, last_states)
        pred_last_returns = np.max(next_q_vals, axis=2).squeeze(axis=1)

        # Backward target return construction
        target_returns = np.zeros(shape=(len(seq_ids), self.sequence_len))
        target_returns[:, -1] = rewards[:, -1] + gamma * pred_last_returns
        target_returns[:, -1][terminals[:, -1]] = rewards[:, -1][terminals[:, -1]]
        target_returns[:, -1][~ mask[:, -1]] = 0
        for time_step in reversed(range(self.sequence_len - 1)):
            target_returns[:, time_step] = rewards[:, time_step] + gamma * target_returns[:, time_step + 1]
            target_returns[:, time_step][terminals[:, time_step]] = rewards[:, time_step][terminals[:, time_step]]
            target_returns[:, time_step][~ mask[:, time_step]] = 0

        # Temporal difference (TD) error
        td_errs = target_returns - pred_returns
        assert not np.any(np.isnan(td_errs))

        # Update transition priorities according to TD errors
        self.memory.set_priorities(trans_ids, np.abs(td_errs))
        seq_prios = self.memory.update_seq_priorities(seq_ids=seq_ids, trans_ids=trans_ids, mask=mask)

        # Prepare inputs and targets for fitting
        inputs = states
        targets = self.target_learner.set_hidden_and_predict(first_hidden_states, states)
        ids_i, ids_s = np.mgrid[0:len(seq_ids), 0:self.sequence_len]
        targets[ids_i, ids_s, actions] = target_returns

        # Compute sample weights
        sample_weights = compute_sample_weights(seq_prios, probabilities[seq_ids], seq_num)

        # Update learning rate
        self.update_lr()

        # Update the online network's weights
        loss, individual_losses, predictions = self.fit(inputs, targets, epochs=epochs, verbose=verbose,
                                                        batch_size=self.replay_batch_size,
                                                        sample_weights=sample_weights,
                                                        hidden_states=first_hidden_states,
                                                        mask=mask)

        self.stats.denote_learning_stats(loss, individual_losses, self.optimizer.learning_rate.numpy(),
                                         trans_ids, predictions, targets, self.env)

        return np.prod(trans_ids.shape)

    def fit(self, x, y, epochs, batch_size, sample_weights, hidden_states=None, mask=None, verbose=False):
        assert not self.sequential or (hidden_states is not None and mask is not None)
        start = time.time()

        # Prepare the training dataset
        if not self.sequential:
            hidden_states = len(x) * [None]
            mask = len(x) * [None]
        x_2d, x_1d = x
        train_dataset = (x_2d, x_1d, y, sample_weights, hidden_states, mask)
        train_dataset = tf.data.Dataset.from_tensor_slices(train_dataset).batch(batch_size)
        # train_dataset = train_dataset.shuffle(buffer_size=1024) is already random
        predictions = np.zeros(y.shape)
        individual_losses = np.zeros(y.shape[:-1])
        train_loss = 0

        for epoch in range(epochs):
            if verbose:
                print("\rEpoch %d/%d - Batch 0/%d - Loss: --" %
                      ((epoch + 1), epochs, len(train_dataset)), flush=True, end="")

            # Iterate over the batches of the dataset.
            for step, (x_2d_b, x_1d_b, y_b, sample_weight_b, hidden_states_b, mask_b) in enumerate(train_dataset):
                # Train on this batch
                if self.sequential:
                    self.online_learner.reset_states()
                    self.online_learner.stem_model.set_cell_states(hidden_states_b)
                batch_individual_losses, batch_out = self.train_step(x_2d_b, x_1d_b, y_b, sample_weight_b, mask_b)

                if epoch == epochs - 1:
                    at_instance = step * batch_size
                    predictions[at_instance:at_instance + len(y_b)] = batch_out
                    individual_losses[at_instance:at_instance + len(y_b)] = batch_individual_losses

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

        # individual_losses /= epochs

        return train_loss, individual_losses, predictions

    @tf.function
    def train_step(self, x_2d, x_1d, y, sample_weight, mask_b):
        with tf.GradientTape() as tape:
            out = self.online_learner([x_2d, x_1d], training=True)
            weighted_losses = self.training_loss_fn(y, out, sample_weight=sample_weight)
            if self.sequential:
                mask = tf.cast(mask_b, tf.float32)
                weighted_losses = tf.multiply(weighted_losses, mask)
            cumulated_loss = tf.reduce_mean(weighted_losses)

        grads = tape.gradient(cumulated_loss, self.online_learner.trainable_weights)

        # Compute unweighted losses
        losses = self.training_loss_fn(y, out)
        if self.sequential:
            losses = tf.multiply(losses, mask)

        # Run one step of gradient descent by the optimizer
        self.optimizer.apply_gradients(zip(grads, self.online_learner.trainable_weights))

        self.training_loss_metric.update_state(y, out)

        return losses, out

    def plan(self, states, epsilon):
        """Epsilon greedy policy. With a probability of epsilon, a random action is returned. Else,
        the agent predicts the best shot for the given state.

        :param states: List of state matrices
        :param epsilon: If given, epsilon greedy policy will be applied, otherwise the agent plans optimally
        :return: action: An index number, corresponding to an action
        """

        if self.sequential:
            states_2d, states_1d = states
            states_2d = np.expand_dims(states_2d, axis=1)
            states_1d = np.expand_dims(states_1d, axis=1)
            states = (states_2d, states_1d)

        # Obtain list of action-values Q(s,a)
        batch_size = self.num_par_envs if self.sequential else self.replay_batch_size
        q_vals = self.actor.predict(states, batch_size=batch_size)
        if self.sequential:
            q_vals = np.squeeze(q_vals, axis=1)
        pred_rets = np.max(q_vals, axis=1)

        # Do epsilon-greedy
        if np.random.random(1) < epsilon:
            # Random action
            actions = np.random.randint(self.num_actions, size=self.num_par_envs)
        else:
            # Optimal action
            actions = q_vals.argmax(axis=1)

        return actions, pred_rets

    def update_lr(self):
        new_lr = self.learning_rate.get_value(self.stats.current_run_episode_no)
        self.optimizer.learning_rate.assign(new_lr)

    def learn_entire_experience(self, batch_size, epochs, gamma, delta, alpha):
        experience_length = self.memory.get_length()
        self.learn_instances(experience_length, gamma, delta, batch_size, epochs, alpha)

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

    def get_states(self):
        """Fetches the current game grid."""
        return self.env.get_states()

    def just_play(self, episodes=9999999, epsilon=0, verbose=False):
        print("Just playing around...")

        if self.num_par_envs != 1:
            self.reinit_env(1)
            self.num_par_envs = 1
            self.actor = self.init_actor()

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
            action, pred_rets = self.plan(state, epsilon)

            if verbose:
                print("{:>11.2f}".format(ret) + " | " +
                      "{:>13.2f}".format(pred_rets[0]) + " | " +
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
                    # time.sleep(0.35)

            test_scores[level] = score

        _, highscores_human = test_env.get_highscores()
        plot_highscores(test_scores, highscores_human, self.out_path)

    def get_config(self):
        """Keep compatible with load_model()!"""
        stem_config = self.stem_model.get_config()
        q_config = self.q_network.get_config()
        # lr_config = self.learning_rate.get_config()
        agent_config = {"num_parallel_envs": self.num_par_envs,
                        "replay_batch_size": self.replay_batch_size,
                        "sequence_shift": self.memory.sequence_shift,
                        "use_double": self.double,
                        "obs_buf_size": self.obs_buf_size,
                        "mem_size": self.mem_size,
                        "eta": self.memory.eta}

        config = {"stem_model_class": self.stem_model.__class__.__name__,
                  "stem_model_config": stem_config,
                  "q_network_class": self.q_network.__class__.__name__,
                  "q_network_config": q_config,
                  # "lr_config": lr_config,
                  "agent_config": agent_config}

        return config

    def save_config(self):
        config = self.get_config()
        config2json(config, out_path=self.out_path + "config.json")

    def save(self, out_path=None, overwrite=False, checkpoint=False, checkpoint_no=None):
        """Saves the current model weights and statistics to a specified export path."""
        if out_path is None:
            model_path = self.out_path + ("checkpoints/" if checkpoint else "trained_model")
        else:
            model_path = out_path

        if checkpoint_no is not None:
            model_path += "%d" % checkpoint_no

        self.online_learner.save_weights(model_path, overwrite=overwrite)

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
            if chkpt_path is None:
                raise FileNotFoundError("No model found at '%s'." % in_path)
            self.restore_from(chkpt_path, in_path)

    def restore_from(self, model_path, stats_path):
        print("Restoring model from '%s'." % model_path)
        self.online_learner.load_weights(model_path)
        self.target_learner = self.init_target_learner()
        self.actor = self.init_actor()
        self.stats.load(in_path=stats_path)

    def save_experience(self, experience_path=None, overwrite=False, compress=False):
        pass

    def restore_experience(self, experience_path=None, gamma=None):
        pass

    def forget(self):
        self.memory = ReplayMemory(memory_size=self.mem_size,
                                   state_shape_2d=self.state_shape_2d,
                                   state_shape_1d=self.state_shape_1d)

    def print_transitions(self, ids):
        self.memory.print_trans_from(ids, self.env)

    def write_hyperparams_file(self, num_parallel_steps, replay_period, replay_size_multiplier, sync_period,
                               alpha, gamma, delta, delta_anneal):
        hyperparams_file = open(self.out_path + "hyperparams.txt", "w+")
        text = "num_parallel_steps: %d" % num_parallel_steps + \
               "\nnum_parallel_envs: %d" % self.num_par_envs + \
               "\nreplay_period: %d" % replay_period + \
               "\nreplay_size_multiplier: %d" % replay_size_multiplier + \
               "\nsync_period: %d" % (sync_period if self.double else -1) + \
               "\n\nalpha: %f" % alpha + \
               "\ngamma: %f" % gamma + \
               "\neta: %f" % self.memory.eta + \
               "\ndelta: %f" % delta + \
               "\ndelta_anneal: %f" % delta_anneal + \
               "\nobs_buf_size: %d" % self.obs_buf_size + \
               "\nexp_buf_size: %d" % self.mem_size

        text += "\n\nSTEM MODEL PARAMETERS:"
        stem_config = self.stem_model.get_config()
        text += config2text(stem_config)

        text += "\n\nQ-NETWORK PARAMETERS:"
        q_config = self.q_network.get_config()
        text += config2text(q_config)

        text += "\n\nLEARNING RATE:"
        lr_config = self.learning_rate.get_config()
        text += config2text(lr_config)

        text += "\n\nEPSILON:"
        eps_config = self.epsilon.get_config()
        text += config2text(eps_config)

        hyperparams_file.write(text + "\n\n")
        self.online_learner.summary(print_fn=lambda x: hyperparams_file.write(x + '\n'))
        hyperparams_file.close()

    def setup_out_path(self, override=False):
        out_path = "out/%s/%s/" % (self.env.NAME, self.name)
        if not override:
            check_for_existing_model(out_path)
        os.makedirs(out_path, exist_ok=True)
        return out_path


def compute_sample_weights(sample_priorities, sample_probabilities, total_size, beta=0.5):
    """Computes sample weights for training. Part of Prioritized Experience Replay."""
    weights = (total_size * sample_probabilities) ** (- beta)  # importance-sampling weights
    weights /= np.max(weights)  # normalization
    return np.abs(np.multiply(weights, sample_priorities))


def compute_target_returns(pred_returns, pred_next_returns, rewards, gamma, delta, mc_returns, terminals):
    # Compute one-step return
    one_step_returns = rewards + gamma * pred_next_returns

    # Compute convex combination of MC return and one-step return
    target_returns = delta * mc_returns + (1 - delta) * one_step_returns

    # Set target return = reward for all terminal transitions
    target_returns[terminals] = rewards[terminals]

    # Compute Temporal Difference (TD) errors (the "surprise" of the agent)
    td_errs = target_returns - pred_returns

    return target_returns, td_errs


def load_model(model_name, env, checkpoint_no=None):
    in_path = "out/" + env.NAME + "/" + model_name + "/config.json"
    config = json2config(in_path)

    stem_class = eval(config["stem_model_class"])
    stem_config = config["stem_model_config"]
    q_class = eval(config["q_network_class"])
    q_config = config["q_network_config"]
    # lr_config = config["lr_config"]
    agent_config = config["agent_config"]

    stem_model = stem_class(**stem_config)
    q_network = q_class(**q_config)
    # lr = LearningRate(**lr_config)

    agent = Agent(env_type=env, stem_model=stem_model, q_network=q_network,
                  name="tmp", override=True, **agent_config)
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
