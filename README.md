# AiBirds
**Goal:** Implementation of a fast-learning deep neural network with general applicability and especially well Angry Birds playing performance.

The idea originated from the [Angry Birds AI Competition](http://aibirds.org/).

<p align="center">
  <a href="#installation">Installation</a> •
  <a href="#usage">Usage</a> •
  <a href="#key-features">Key Features</a> •
  <a href="#troubleshooting">Troubleshooting</a> •
  <a href="#acknoledgements">Acknoledgements</a> •
  <a href="#bibliography">Bibliography</a> •
  <a href="#license">License</a>
</p>

## Installation
No dedicated installation required anymore.


## Usage
Just tune and run the agent from `src/play.py`. You can let the agent practice, observe how the agent plays, plot statistics, and more.

Any generated output (models, plots, statistics etc.) will be saved in `out/`.

### Some helpful tips
* You can generate new levels using the integrated level generator.
* Increase the environment speed to accelerate the learning process.


## Key Features
* Deep Q Network with following improvements
	* Annealing epsilon-Greedy Policy
	* Dueling Networks
	* Prioritized Experience Replay
	* Double Q-Learning
	* Optional Multi-step Learning
* Applicability to environments other than Angry Birds. This repo does also ship with Tetris and Snake.


## Contributing
1. Fork it!
2. Create your feature branch: `git checkout -b my-new-feature`
3. Commit your changes: `git commit -am 'Add some feature'`
4. Push to the branch: `git push origin my-new-feature`
5. Submit a pull request :D

## Troubleshooting

### System language problems (especially for German contributors)
* Symptom: level objects don't show up after level was load. The actual problem is that the objects did spawn outside of the level boundaries. The reason for this turned out to be the OS's language/unit configuration. In my case the system language is de_DE and for this, the decimal point is not a point but a comma (e.g. 2,7). The problem is that unity3D uses the system configuration for their coordination system and coordinates like x=2.5, y=8.4 could not be interpreted correctly.
* Solution: start ScienceBirds with the language en_US.UTF-8 so that unity3D uses points for floats instead of commas, or (in case of Windows) set the OS's region to English (U.S.).


## Acknowledgements
+ The team behind [Science Birds](https://gitlab.com/aibirds/sciencebirdsframework) for a good framework

## Bibliography
* [The Angry Birds AI Competition](https://www.aaai.org/ojs/index.php/aimagazine/article/view/2588)
* [Rainbow DQN](https://arxiv.org/pdf/1710.02298.pdf)
* [Deep Q-Network for Angry Birds](https://arxiv.org/pdf/1910.01806.pdf)
* ...

## License
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
