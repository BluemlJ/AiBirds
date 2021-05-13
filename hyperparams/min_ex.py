from src.envs import *
import src.agent.model as model

hyperparams = {
    "name": "minimum_example",
    "env": Snake(num_par_inst=100),
    "stem_network": model.StemNetwork2D1D(128),
    "num_parallel_steps": 20000,
    "gamma": 0.995,
    "replay_period": 32,
}
