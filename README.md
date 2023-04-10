# AiBirds
**Goal:** Implementation of a fast-learning deep neural network with general applicability and especially well Angry Birds playing performance.

The idea originated from the [Angry Birds AI Competition](http://aibirds.org/).


[Key Features](#key-features) • [Installation](#installation) • [Usage](#usage) •
[Troubleshooting](#troubleshooting) • [Acknowledgements](#acknowledgements) • 
[Bibliography](#bibliography) • [License](#license) • 


## Key Features
* Reinforcement learning framework with following features:
	* Annealing Epsilon-Greedy Policy
	* Dueling Networks
	* Prioritized Experience Replay
	* Double Q-Learning
	* Frame-stacking
	* n-step Expected Sarsa
	* Distributed RL
	* (Sequential Learning (for LSTMs and other RNNs))
	* Monte Carlo Return Targets
* Environments:
    * Angry Birds, with level generator
    * Snake
    * Tetris
    * Chain Bomb (invented)
* Environment API: quick switch between environments without need of changing the model
* Extensive utility library:
    * Plotting: >40 different ways to compare several training run metrics (score, return, time etc.) on different
      domains (transitions, episodes, wall-clock time, etc.)
    * Permanence: save and reload entire training runs, including model weights and statistics
    * Pre-training: unsupervised model pre-training on randomly generated environment data
    * Train sample control: sample states with extreme loss during training are logged and plotted automatically
    * and much more
    
### Soon to come
* Frame stacking
* Angry Birds environment parallelization
* Handling for (practically) infinite episodes


## Installation
Just clone the repo, that's it.


## Usage
Just tune and run the agent from `src/train.py`. You can let the agent practice, observe 
how the agent plays, view plot statistics, and more.

Any generated output (models, plots, statistics etc.) will be saved in `out/`.

### Parameter Overview
| Parameter                | Reasonable value  | Explanation            | If too low, then       | If too high, then           |
| ----------------         | ----------------- | ---------------------- | ---------------------- | --------------------------- |
| **General**
| `num_parallel_inst`      | `500`             | Number of simultaneously executed environments | Training overhead dominates computation time | Possibly worse sample complexity, GPU or RAM out of memory |
| `num_parallel_steps`     | `1000000`         | Number of transitions done per parallel environments | Learning stops before agent performance is optimal | Wasted energy, overfitting |
| `policy`                 | `"greedy"`        | The policy used for planning (`"greedy"` for max Q-value, `"softmax"` for random choice of softmaxed Q-values) | - | - |
| **Model input**
| `stack_size`             | `1`               | Number of recent frames to be stacked for input, useful for envs with time dependency like Breakout | Agent has "no feeling for time", bad performance on envs with time dependency | Unnecessary computation overhead |
| **Learning target**
| `gamma`                  | `0.999`           | Discount factor | Short-sighted strategy, early events dominate return | Far-sighted strategy, late events dominate return, target shift, return explosion |
| `n_step`                 | `1`               | Number of steps used for Temporal Difference (TD) bootstrapping |  |  |
| `use_mc_return`          | `False`           | If True, uses Monte Carlo instead of n-step TD |  |  |
| **Model**
| `latent_dim`             | `128`             | Width of latent layer of stem model | Model cannot learn the game entirely or makes slow training progress | Huge number of (unused) model parameters |
| `latent_depth`           | `1`               | Number of consecutive latent layers | Model cannot learn the game entirely or makes slow training progress | Many (unused) model parameters |
| `lstm_dim`               | `128`             | Width of LSTM layer | Model cannot learn the game entirely, slow training progress or model is bad at remembering | Many (unused) model parameters |
| `latent_v_dim`           | `64`              | Width of latent layer of _value_ part of Q-network | Similar to `latent_dim` | Similar to `latent_dim` |
| `latent_a_dim`           | `64`              | Width of latent layer of _advantage_ part of Q-network | Similar to `latent_dim` | Similar to `latent_dim` |
| **Replay (training)**
| `optimizer`              | `tf.Adam`         | The `tf` optimizer to use | - | - |
| `mem_size`               | `4000000`         | Number of transitions that fit into the replay memory | Overfitting to recent observations | RAM out of memory, too old transitions in replays => in case of RNNs can lead to _recurrent state staleness_ due to representational drift; large computation overhead due to replay sampling |
| `replay_period`          | `64`              | Number of (parallel) steps between each training session of the learner | Training overhead dominates computation time | Slow training progress |
| `replay_size_multiplier` | `4`               | Determines replay size by multiplying number of new transitions with this factor | Too strong focus on new observations, overfitting | Too weak focus on new observations, slow training progress |
| `replay_batch_size`      | `1024`            | Batch size used for learning, depends on GPU | GPU parallelization not used effectively | GPU out of memory |
| `replay_epochs`          | `1`               | Number of epochs per replay | Wasted training data, slow progress | Overfitting |
| `min_hist_len`           | `0`               | Minimum number of observed transitions before training is allowed | Unstable training or overfitting in the beginning | Wasted time |
| `alpha`                  | `0.7`             | Prioritized experience replay exponent, controls the effect of priority | Priorities have low/no effect on training, slower training progress | Too strong priority emphasis, overfitting |
| `use_double`             | `True`            | Whether to use Double Q-Learning to tackle moving target issue | - | - |
| `target_sync_period`     | `256`             | Number of (parallel) steps between synchronization of online learner and target learner | Moving target problem | Slow training progress |
| `actor_sync_period`      | `64`              | Distributed RL: number of (parallel) steps between synchronization of (online) learner and actor | ? | Slow training progress |
| **Learning rate**
| `init_value`             | `0.0004`          | Starting value of learning rate | Slow training progress | No training progress, bad maximum performance or unstable training |
| `half_life_period`       | `4000000`         | Determines learning rate decay, number of played transitions after which learning rate halves | Learning rate decreases too quickly | Learning rate too large |
| `warmup_transitions`     | `0`               | Number of episodes used for (linear) learning rate warm-up | Unstable training | Slow training progress |
| **Epsilon**
| `init_value`             | `1`               | Starting value for epsilon-greedy policy | Too few exploration, slow training progress | Too much exploration, slow training progress |
| `decay_mode`             | `"exp"`           | Shape of epsilon value function over time | - | - |
| `half_life_period`       | `700000`          | Determines epsilon annealing, number of played transitions after which epsilon halves | Similar to `init_value` | Similar to `init_value` |
| `minimum`                | `0`               | Limit to which epsilon converges when decaying over time | Not enough long-time exploration, missed late-game opportunities for better performance | Similar to `init_value` |
| **Sequential training (for RNNs)**
| `sequence_len`           | `20`              | Length of sequences saved in replay buffer and used for learning | Slow training progress | Wasted computation and memory resources |
| `sequence_shift`         | `10`              | Number of transitions sequences are allowed to overlap | Few sequences, some time-dependencies might not be captured | Too many similar sequences, overfitting |
| `eta`                    | `0.9`             | For sequential learning: determines sequence priority. `0`: sequence prio = average instance prio, `1`: sequence prio = max instance prio. | ? | ? |
| **Other**
|                          |                   |  |  |  |


## Contributing
1. Fork it!
2. Create your feature branch: `git checkout -b my-new-feature`
3. Commit your changes: `git commit -am 'Add some feature'`
4. Push to the branch: `git push origin my-new-feature`
5. Submit a pull request :D


## Troubleshooting

### Science Birds (Angry Birds simulator) doesn't show any level objects
* Symptom: level objects don't show up in Science Birds after level was load.
  The actual problem is that the objects did spawn outside of the level boundaries.
  The reason for this turned out to be the OS's language/unit configuration. In my case
  the system language is de_DE and for this, the decimal point is not a point but a comma
  (e.g. 2,7). The problem is that unity3D uses the system configuration for their
  coordination system and coordinates like x=2.5, y=8.4 could not be interpreted correctly.
* Solution: start ScienceBirds with the language en_US.UTF-8 so that unity3D uses points for
  floats instead of commas, or (in case of Windows) set the OS's region to English (U.S.).


## Acknowledgements
+ The team behind [Science Birds](https://gitlab.com/aibirds/sciencebirdsframework) for a good framework


## Bibliography
* [The Angry Birds AI Competition](https://www.aaai.org/ojs/index.php/aimagazine/article/view/2588)
* [Rainbow DQN](https://arxiv.org/pdf/1710.02298.pdf)
* [Deep Q-Network for Angry Birds](https://arxiv.org/pdf/1910.01806.pdf)
* ...


## License
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
