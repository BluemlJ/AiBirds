import time
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2"  # let TF only print errors
import tensorflow as tf
import numpy as np

from src.utils import Statistics, ParamScheduler, copy_object_with_config, plot_highscores, config2text, \
    ask_to_override_model, config2json, json2config, remove_folder, log_model_graph, random_choice_along_last_axis
from src.mem.mem import ReplayMemory
from src.envs.env import ParallelEnvironment
from src.agent.comp import *
from src.utils.stack import StateStacker

# Miscellaneous
PLOT_SAVE_STATS_PERIOD = 1000  # number of transitions between each learning statistics plot
PRINT_STATS_PERIOD = 100
CHECKPOINT_SAVE_PERIOD = 5000
TEST_PERIOD = 5000


class Agent:
    """Deep Q-Network (DQN) agent for playing Tetris, Snake or other games"""

    def __init__(self, name, env: ParallelEnvironment, stem_network: StemNetwork,
                 q_network: QNetwork = DoubleQNetwork(),
                 replay_batch_size=512, stack_size=1, optimizer=tf.optimizers.Adam(),
                 use_pretrained=False, override=False, seed=None, **kwargs):
        """Constructor
        :param env: The (instantiated) environment in which the agent acts
        :param stem_network: The main stem model
        :param q_network: The Q-network (coming after the stem model)
        :param name: A string identifying the agent in file names
        """

        print("Initializing DQN agent...")

        # General
        self.name = name
        self.seed = seed
        self.policy = None

        # Environment
        self.env = env
        self.num_par_envs = self.env.num_par_inst
        self.state_shapes = self.env.get_state_shapes()
        self.num_actions = self.env.get_number_of_actions()

        # Frame stacking
        self.stack_size = stack_size
        self.stacker = StateStacker(self.state_shapes, self.stack_size, self.num_par_envs)
        self.stack_shapes = self.stacker.get_stack_shapes()

        if name == "debug":
            override = True
        self.out_path = self.setup_out_path(override)

        # For training
        self.replay_batch_size = replay_batch_size
        self.learning_rate = None
        self.epsilon = None
        self.delta = None
        self.use_mc_return = None  # Monte Carlo

        # Model architecture
        self.stem_network = stem_network
        self.sequential = stem_network.sequential
        self.sequence_len = stem_network.sequence_len
        self.q_network = q_network
        self.optimizer = optimizer
        self.online_learner = self.init_online_learner()
        self.q_net_layer_id = np.where([isinstance(layer, QNetwork) for layer in self.online_learner.layers])[0][0]
        log_model_graph(self.online_learner, self.stack_shapes)

        if use_pretrained:
            self.load_pretrained_model()

        # Double Q-Learning and Distributed RL
        self.target_learner = self.online_learner
        self.actor = self.online_learner

        # Training loss
        self.training_loss_fn = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE)
        self.training_loss_metric = tf.keras.metrics.MeanSquaredError()

        # Memory and statistics
        self.memory = None
        self.stats = Statistics(env=self.env, log_path=self.out_path)

        self.save_config()
        print('DQN agent initialized.')

    def init_online_learner(self) -> tf.keras.Model:
        model = self.init_model(self.stem_network, self.q_network, self.replay_batch_size)
        model.summary()

        # the following needs separate GraphViz installation from https://graphviz.gitlab.io/download/
        # this helped for GraphViz bugfix: https://datascience.stackexchange.com/questions/74500
        tf.keras.utils.plot_model(model, to_file=self.out_path + 'model_plot.png', show_shapes=True,
                                  show_layer_names=True, expand_nested=True, dpi=400)
        return model

    def init_model(self, stem_net: StemNetwork, q_net: QNetwork, batch_size):
        q_net.set_num_actions(self.num_actions)
        q_net.set_sequential(self.sequential)
        inputs, latent = stem_net.get_functional_graph(self.stack_shapes, batch_size)
        outputs = q_net(latent)
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        model.compile(loss='huber_loss', optimizer=self.optimizer)  # Huber loss equiv. to gradient clipping
        return model

    def copy_online_learner(self):
        stem_net = copy_object_with_config(self.stem_network)
        q_net = copy_object_with_config(self.q_network)
        model = self.init_model(stem_net, q_net, self.num_par_envs)
        model.set_weights(self.online_learner.get_weights())
        return model

    def init_replay_memory(self, memory_size, n_step, sequence_shift, eta):
        return ReplayMemory(size=memory_size,
                            state_shapes=self.state_shapes,
                            n_step=n_step,
                            stack_size=self.stack_size,
                            num_par_envs=self.env.num_par_inst,
                            hidden_state_shapes=self.stem_network.get_hidden_state_shape(),
                            sequence_len=self.sequence_len,
                            sequence_shift=sequence_shift,
                            eta=eta)

    def load_pretrained_model(self):
        pretrained_path = "out/" + self.env.NAME + "/pretrained"
        if not os.path.exists(pretrained_path):
            Exception("You specified to load a pretrained model. However, there is no pretrained model "
                      "at '%s'." % pretrained_path)

        self.online_learner.load_weights(pretrained_path + "/pretrained", by_name=True)

    def reinit_env(self, num_par_envs):
        self.num_par_envs = num_par_envs
        new_env = self.env.copy(num_par_envs)
        del self.env
        self.env = new_env

    def get_hidden_states_of(self, model):
        if self.sequential:
            return None  # TODO
        else:
            return None

    def reset_cell_states_for(self, model, ids):
        pass  # TODO

    def practice(self, num_parallel_steps,
                 replay_period, gamma,
                 learning_rate: ParamScheduler = ParamScheduler(0.0001),
                 target_sync_period=None, actor_sync_period=None,
                 replay_size_multiplier=4, replay_epochs=1,
                 epsilon: ParamScheduler = ParamScheduler(0),
                 use_mc_return=False, alpha=0.7,
                 max_replay_size=None,
                 policy="greedy",
                 min_hist_len=0,
                 memory_size=1000000,
                 n_step=1,
                 sequence_shift=None, eta=0.9,
                 verbose=False, **kwargs):
        """The agent's main training routine.

        :param num_parallel_steps: Number of (parallel) transitions to play
        :param replay_period: Number of parallel steps between each training of the online network
        :param replay_size_multiplier: Factor determining the number of transitions to be learned from each
                   hyperparams cycle (the replay size). Each time, the replay size is determined as
                   follows: replay_size = replay_size_multiplier * new_transitions
        :param replay_epochs: Number of epochs per replay
        :param learning_rate: (dynamic) learning rate used for training
        :param target_sync_period: The number of levels between each synchronization of online and target network. The
                   higher the number, the stronger Double Q-Learning and the less overestimation.
                   sync_period == 1 means "Double Q-Learning off"
        :param actor_sync_period: The number of levels between each synchronization of learner and actor.
        :param gamma: Discount factor
        :param epsilon: Epsilon class, probability for random shot (epsilon greedy policy)
        :param use_mc_return: If True, uses Monte Carlo return targets instead of n-step TD targets
        :param alpha: For Prioritized Experience Replay: the larger alpha the stronger prioritization
        :param max_replay_size:
        :param policy:
        :param min_hist_len:
        :param memory_size:
        :param n_step:
        :param sequence_shift:
        :param eta:
        :param verbose:
        """
        self.set_policy(policy)

        self.policy = policy
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.use_mc_return = use_mc_return

        if target_sync_period is not None:
            self.target_learner = self.copy_online_learner()
        if actor_sync_period is not None:
            self.actor = self.copy_online_learner()

        # Init replay memory
        self.memory = self.init_replay_memory(memory_size, n_step, sequence_shift, eta)

        # Save all hyperparameters in txt file
        self.write_hyperparams_file(num_parallel_steps, replay_period, replay_size_multiplier,
                                    target_sync_period, actor_sync_period, alpha, gamma, min_hist_len)

        new_transitions = 0
        returns = np.zeros(self.num_par_envs)

        # Reset all environments
        self.env.reset()

        print("DQN agent starts practicing...")

        self.stats.start_timer()

        for i in range(1, num_parallel_steps + 1):
            done_transitions = (i - 1) * self.num_par_envs

            states = self.get_states()
            self.stacker.add_states(states)
            hidden_states = self.get_hidden_states_of(self.actor)

            # Predict the next action to take (move, rotate or do nothing)
            actions, _ = self.plan_epsilon_greedy(self.stacker.get_stacks(), self.epsilon.get_value(done_transitions))

            # Perform actions, observe new environment state, level score and application state
            rewards, scores, terminals, times, wins, game_overs = self.env.step(actions)
            returns += rewards

            # Save observations
            new_transitions += self.memory.memorize_observations(states, hidden_states, actions, scores,
                                                                 rewards, terminals, gamma)

            # Handle finished envs
            if np.any(game_overs):
                fin_env_ids = np.where(game_overs)[0]

                # Save stats
                for idx in fin_env_ids:
                    self.stats.denote_episode_stats(returns[idx], scores[idx], times[idx], wins[idx],
                                                    idx, self.memory)

                # Reset all finished envs and update their corresponding current return
                self.env.reset_finished()
                returns[fin_env_ids] = 0

                self.stacker.reset_stacks(fin_env_ids)

                # Reset actor's LSTM states (if any) to zero
                self.reset_cell_states_for(self.actor, fin_env_ids)

            # Every X episodes, plot informative graphs
            if i % PLOT_SAVE_STATS_PERIOD == 0:
                self.stats.plot_stats(self.memory, self.out_path)
                self.stats.save(self.out_path)

            # If environment has test levels, test on it
            if i % TEST_PERIOD == 0 and self.env.has_test_levels():
                self.test_on_levels()

            # Update the network weights every train_period levels to fit experience
            if i % replay_period == 0:
                self.reset_noise()  # for Noisy Nets (if activated)
                replay_size = replay_size_multiplier * new_transitions
                if self.memory.get_num_transitions() >= min_hist_len and replay_size > 0:
                    if max_replay_size is not None:
                        replay_size = min(replay_size, max_replay_size)
                    learned_trans = self.learn(replay_size, gamma, epochs=replay_epochs, alpha=alpha, verbose=verbose)
                    new_transitions = max(0, new_transitions - learned_trans)

            # Save model checkpoint
            if i % CHECKPOINT_SAVE_PERIOD == 0:
                self.save(overwrite=True, checkpoint=True, checkpoint_no=self.stats.get_num_episodes())

            # Cut off old experience to reduce buffer load
            if self.memory.get_num_transitions() > 0.95 * self.memory.get_size():
                self.memory.delete_first(n=int(0.2 * self.memory.get_size()))

            # Synchronize target and online network every sync_period levels (Double Q-Learning)
            if target_sync_period is not None and i % target_sync_period == 0:
                self.target_learner.set_weights(self.online_learner.get_weights())

            # Synchronize learner and actor (Distributed RL)
            if actor_sync_period is not None and i % actor_sync_period == 0:
                self.actor.set_weights(self.online_learner.get_weights())

            if i % PRINT_STATS_PERIOD == 0:
                self.stats.print_stats(i, num_parallel_steps, PRINT_STATS_PERIOD, self.num_par_envs,
                                       self.epsilon.get_value(done_transitions), self.num_par_envs)

        self.save()
        self.stats.logger.close()

        print("Practicing finished successfully!")

    def learn(self, replay_size, gamma, epochs=1, alpha=0.7, verbose=False):
        """Updates the online network's weights. This is the actual learning procedure of the agent.

        :param replay_size: Number of transitions to be learned from
        :param gamma: Discount factor
        :param epochs:
        :param alpha: For Prioritized Experience Replay: the larger alpha the stronger prioritization
        :param verbose:
        :return: number of transitions learned
        """

        if replay_size == 0:
            return 0

        if not self.sequential:
            return self.learn_instances(replay_size, gamma=gamma, alpha=alpha,
                                        epochs=epochs, verbose=verbose)
        else:
            num_sequences = replay_size // self.sequence_len
            if num_sequences >= self.replay_batch_size:
                return self.learn_sequences(num_sequences, gamma=gamma, alpha=alpha,
                                            epochs=epochs, verbose=verbose)
            else:
                return 0

    def learn_instances(self, num_instances, gamma, epochs=1, alpha=0.7, verbose=False):
        """Uses batches of single instances to learn on."""

        # Obtain a list of useful transitions to learn on
        trans_ids, probabilities = self.memory.recall(num_instances, alpha)

        # Obtain total number of experienced transitions
        exp_len = self.memory.get_num_transitions()

        # Get list of transitions
        states, _, actions, n_step_rewards, n_step_mask, next_states, _, terminals = \
            self.memory.get_transitions(trans_ids)

        # Obtain Monte Carlo return for each transition
        mc_returns = self.memory.get_mc_returns(trans_ids)

        # Predict returns (i.e. values V(s)) for all states s
        q_vals = self.online_learner.predict(states, batch_size=self.replay_batch_size)
        pred_returns = np.max(q_vals, axis=1)

        # Predict next returns
        next_q_vals = self.target_learner.predict(next_states, batch_size=self.replay_batch_size)
        pred_next_returns = np.max(next_q_vals, axis=1)

        # Compute (n-step or MC) return targets and temporal-difference (TD) errors (the "surprise" of the agent)
        if self.use_mc_return:
            target_returns = mc_returns
        else:
            target_returns = get_n_step_return(pred_next_returns, n_step_rewards, n_step_mask, gamma, terminals)
        td_errs = target_returns - pred_returns

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
                                         trans_ids, predictions, targets, self.env, self.memory)

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
                                         trans_ids, predictions, targets, self.env, self.memory)

        return np.prod(trans_ids.shape)

    def fit(self, x, y, epochs, batch_size, sample_weights, hidden_states=None, mask=None, verbose=False):
        assert not self.sequential or (hidden_states is not None and mask is not None)
        start = time.time()

        # Prepare the training dataset
        if not self.sequential:
            hidden_states = len(x) * [None]
            mask = len(x) * [None]
        train_dataset = (*x, y, sample_weights, hidden_states, mask)
        train_dataset = tf.data.Dataset.from_tensor_slices(train_dataset).batch(batch_size)
        # train_dataset = train_dataset.shuffle(buffer_size=1024, seed=self.seed)
        predictions = np.zeros(y.shape)
        individual_losses = np.zeros(y.shape[:-1])
        train_loss = 0

        for epoch in range(epochs):
            if verbose:
                print("\rEpoch %d/%d - Batch 0/%d - Loss: --" %
                      ((epoch + 1), epochs, len(train_dataset)), flush=True, end="")

            # Iterate over the batches of the dataset.
            for step, (*x, y_b, sample_weight_b, hidden_states_b, mask_b) in enumerate(train_dataset):
                # Train on this batch
                if self.sequential:
                    self.online_learner.reset_states()
                    self.online_learner.stem_model.set_cell_states(hidden_states_b)

                batch_individual_losses, batch_out = \
                    self.train_step(x, y_b, sample_weight_b, mask_b)

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

        return train_loss, individual_losses, predictions

    @tf.function
    def train_step(self, x, y, sample_weight, mask_b):
        with tf.GradientTape() as tape:
            out = self.online_learner(x, training=True)
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

    def plan_epsilon_greedy(self, states, epsilon, output_return=False):
        """Epsilon greedy policy. With a probability of epsilon, a random action is returned. Else,
        the agent predicts the best shot for the given state.

        :param states: List of state matrices
        :param epsilon: If given, epsilon greedy policy will be applied, otherwise the agent plans optimally
        :param output_return:
        :return: action: An index number, corresponding to an action
        """

        if np.random.random(1) < epsilon:
            # Random action
            actions = np.random.randint(self.num_actions, size=self.num_par_envs)
            if self.sequential or output_return:  # Update hidden state
                self.plan(states)
            return actions, [np.nan]
        else:
            # Optimal action
            return self.plan(states)

    def plan(self, states):
        if self.sequential:
            states = [np.expand_dims(state_comp, axis=1) for state_comp in states]

        batch_size = self.num_par_envs if self.sequential else self.replay_batch_size

        q_vals = self.actor.predict(states, batch_size=batch_size)
        if self.sequential:
            q_vals = np.squeeze(q_vals, axis=1)

        # Pick action according to policy
        if self.policy == "greedy":
            actions = q_vals.argmax(axis=1)
        else:
            probs = np.exp(q_vals) / np.sum(np.exp(q_vals), axis=1, keepdims=True)
            actions = random_choice_along_last_axis(probs)

        pred_rets = np.max(q_vals, axis=1)

        return actions, pred_rets

    def update_lr(self):
        new_lr = self.learning_rate.get_value(self.stats.current_run_trans_no)
        self.optimizer.learning_rate.assign(new_lr)

    def reset_noise(self):
        models = {self.online_learner, self.target_learner, self.actor}
        for model in list(models):
            q_net = model.layers[self.q_net_layer_id]
            q_net.reset_noise()

    def set_noisy(self, model, active: bool):
        q_net = model.layers[self.q_net_layer_id]
        q_net.set_noisy(active)

    def set_policy(self, policy):
        assert policy in ["greedy", "softmax"]
        self.policy = policy

    def learn_entire_experience(self, batch_size, epochs, gamma, alpha):
        experience_length = self.memory.get_num_transitions()
        self.learn_instances(experience_length, gamma, batch_size, epochs, alpha)

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

    def just_play(self, num_par_envs=1, episodes=9999999, policy="greedy", epsilon=0, verbose=False):
        print("Just playing around...")
        self.set_policy(policy)

        if self.num_par_envs != num_par_envs:
            self.reinit_env(num_par_envs)
            self.num_par_envs = num_par_envs
            self.actor = self.online_learner
            self.stacker = StateStacker(self.state_shapes, self.stack_size, self.num_par_envs)

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
            self.stacker.add_states(state)

            # Predict the next action to take (move, rotate or do nothing)
            action, pred_rets = self.plan_epsilon_greedy(self.stacker.get_stacks(), epsilon, output_return=True)

            if verbose:
                print("{:>11.2f}".format(ret) + " | " +
                      "{:>13.2f}".format(pred_rets[0]) + " | " +
                      "{:>16s}".format(self.env.actions[action[0]]))

            # Perform action, observe new environment state, level score and application state
            reward, score, _, env_time, _, game_over = self.env.step(action)

            if render_environment:
                self.env.render()

            ret += reward[0]

        if verbose:
            print("----------------------------------------------")
            print("Level finished with return %.2f, score %d, and time %d.\n" % (ret, score, env_time))

        return ret, score

    def test_on_levels(self, render=False):
        test_env = self.env.copy(1)
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
                action, _ = self.plan_epsilon_greedy(state, 0)

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
        stem_config = self.stem_network.get_config()
        q_config = self.q_network.get_config()
        agent_config = {"replay_batch_size": self.replay_batch_size,
                        "stack_size": self.stack_size,
                        "seed": self.seed}

        config = {"stem_model_class": self.stem_network.__class__.__name__,
                  "stem_model_config": stem_config,
                  "q_network_class": self.q_network.__class__.__name__,
                  "q_network_config": q_config,
                  "env_config": self.env.get_config(),
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

        print("Saved model.")

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
        self.target_learner = self.online_learner
        self.actor = self.online_learner
        self.stats.load(in_path=stats_path)

    def save_experience(self, experience_path=None, overwrite=False, compress=False):
        pass

    def restore_experience(self, experience_path=None, gamma=None):
        pass

    def forget(self):
        self.memory = ReplayMemory(**self.memory.get_config())

    def write_hyperparams_file(self, num_parallel_steps, replay_period, replay_size_multiplier, target_sync_period,
                               actor_sync_period, alpha, gamma, min_hist_len):
        hyperparams_file = open(self.out_path + "hyperparams.txt", "w+")
        text = "num_parallel_steps: %d" % num_parallel_steps + \
               "\nnum_parallel_envs: %d" % self.num_par_envs + \
               "\npolicy: %s" % self.policy + \
               "\nreplay_period: %d" % replay_period + \
               "\nreplay_size_multiplier: %d" % replay_size_multiplier + \
               "\nmin_hist_len: %d" % min_hist_len + \
               "\ntarget_sync_period: " + str(target_sync_period) + \
               "\nactor_sync_period: " + str(actor_sync_period) + \
               "\n\nalpha: %f" % alpha + \
               "\ngamma: %f" % gamma + \
               "\nuse_mc_return: " + str(self.use_mc_return) + \
               "\nstack_size: %d" % self.stack_size + \
               "\nseed: %d" % self.seed

        text += "\n\nSTEM MODEL PARAMETERS:"
        text += config2text(self.stem_network.get_config())

        text += "\n\nQ-NETWORK PARAMETERS:"
        text += config2text(self.q_network.get_config())

        text += "\n\nLEARNING RATE:"
        text += config2text(self.learning_rate.get_config())

        text += "\n\nEPSILON:"
        text += config2text(self.epsilon.get_config())

        text += "\n\nMEMORY:"
        text += config2text(self.memory.get_config())

        hyperparams_file.write(text + "\n\n")
        self.online_learner.summary(print_fn=lambda x: hyperparams_file.write(x + '\n'))
        hyperparams_file.close()

    def setup_out_path(self, override=False):
        out_path = "out/%s/%s/" % (self.env.NAME, self.name)
        if os.path.exists(out_path):
            if override:
                remove_folder(out_path)
            else:
                ask_to_override_model(out_path)
        os.makedirs(out_path, exist_ok=True)
        return out_path


def compute_sample_weights(sample_priorities, sample_probabilities, total_size, beta=0.5):
    """Computes sample weights for training. Part of Prioritized Experience Replay."""
    weights = (total_size * sample_probabilities) ** (- beta)  # importance-sampling weights
    weights /= np.max(weights)  # normalization
    return np.abs(np.multiply(weights, sample_priorities))


def get_n_step_return(pred_next_returns, n_step_rewards, n_step_mask, gamma, terminals):
    # Prepare
    n = n_step_rewards.shape[-1]
    step_axis = n_step_rewards.ndim - 1
    final_step_mask = n_step_mask[:, -1] & ~ terminals
    mask = np.append(n_step_mask, np.expand_dims(final_step_mask, axis=1), axis=step_axis)

    # Compute n-step return
    discounts = gamma ** np.arange(n + 1)
    rewards_returns = np.append(n_step_rewards, np.expand_dims(pred_next_returns, axis=1), axis=step_axis)
    rewards_returns_discounted = rewards_returns * discounts
    rewards_returns_discounted[~ mask] = 0
    n_step_returns = np.sum(rewards_returns_discounted, axis=step_axis)

    return n_step_returns


def load_model(model_name, env_type, num_par_envs=None, checkpoint_no=None):
    in_path = "out/" + env_type.NAME + "/" + model_name + "/config.json"
    config = json2config(in_path)

    stem_class = eval(config["stem_model_class"])
    stem_config = config["stem_model_config"]
    q_class = eval(config["q_network_class"])
    q_config = config["q_network_config"]
    env_config = config["env_config"]
    agent_config = config["agent_config"]

    if num_par_envs is not None:
        env_config.update({"num_par_inst": num_par_envs})

    stem_net = stem_class(**stem_config)
    q_net = q_class(**q_config)
    env = env_type(**env_config)

    agent = Agent(env=env, stem_network=stem_net, q_network=q_net,
                  name="tmp", override=True, **agent_config)
    agent.restore(model_name, checkpoint_no)
    return agent


def load_and_play(model_name, env_type, num_par_envs=1, checkpoint_no=None, mode=None, epsilon=0):
    agent = load_model(model_name, env_type, num_par_envs=num_par_envs, checkpoint_no=checkpoint_no)
    if mode is not None:
        agent.env.set_mode(mode)
    agent.just_play(num_par_envs, verbose=True, epsilon=epsilon)


def load_and_test(model_name, env_type, checkpoint_no=None, mode=None, render=False):
    agent = load_model(model_name, env_type, checkpoint_no)
    if mode is not None:
        agent.env.set_mode(mode)
    agent.test_on_levels(render)
