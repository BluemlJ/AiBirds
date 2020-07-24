# AiBirds
<p align="center">
  <a href="#installation">Installation</a> •
  <a href="#usage">Usage</a> •
  <a href="#key-features">Key Features</a> •
  <a href="#troubleshooting">Troubleshooting</a> •
  <a href="#acknoledgements">Acknoledgements</a> •
  <a href="#bibliography">Bibliography</a> •
  <a href="#license">License</a>
</p>

The idea behind this project is to write a "Deep Reinforcement Learning" agent to play Angry Birds. The idea originated from the [Angry Birds AI Competition](http://aibirds.org/).

## Installation
* Download and install the [Science Bird Framework](https://gitlab.com/aibirds/sciencebirdsframework)
 * we recommend to test the framework with the native agent to check installation
* Install the requirements 
* Update the src folder with our src folder
* Update the main.py to start the new dqn agent.

## Usage

1. Run the game playing interface from science birds
	- the runnable jar file is in the root folder of the repository called game_playing_interface.jar
	- use the code below to run:

<code>java -jar  game_playing_interface.jar</code>

2. Run the Science Birds game executable file to start a game instance
	- The Science Birds game used in this framework is a modified version of Lucas N. Ferreira's work that can be found in his [Github repository](https://github.com/lucasnfe/science-birds)

3. Run the new agent

<code>python3 main.py</code>


## Key Features
* Deep Q Network with following improvements
 * Annealing epsilon-Greedy Policy
 * Dueling Networks
 * Prioritized Replay
 * Double Q-Learning
 * Multi-step Learning

## Contributing

1. Fork it!
2. Create your feature branch: `git checkout -b my-new-feature`
3. Commit your changes: `git commit -am 'Add some feature'`
4. Push to the branch: `git push origin my-new-feature`
5. Submit a pull request :D


## Acknowledgements

+ The team behind [Science Birds](https://gitlab.com/aibirds/sciencebirdsframework) for a good framework

## Bibligraphy
* [The Angry Birds AI Competition](https://www.aaai.org/ojs/index.php/aimagazine/article/view/2588)
* [Rainbow DQN](https://arxiv.org/pdf/1710.02298.pdf)
* [Deep Q-Network for Angry Birds](https://arxiv.org/pdf/1910.01806.pdf)
* ...

## License
see liecense file
[![License](https://poser.pugx.org/aimeos/aimeos-typo3/license.svg)](https://packagist.org/packages/aimeos/aimeos-typo3)
