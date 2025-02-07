# Constraints as Rewards: Reinforcement Learning for Robots without Reward Functions

This repository provides the official pytorch implementation of QRSAC-Lagrangian algorithm presented in the paper:
"Constraints as Rewards: Reinforcement Learning for Robots without Reward Functions".

[![Project Site](https://img.shields.io/badge/Project-Web-green)](https://sony.github.io/CaR) &nbsp;
[![Github](https://img.shields.io/badge/Github-Repo-orange?logo=github)](https://github.com/sony/CaR/) &nbsp; 
[![arXiv](https://img.shields.io/badge/arXiv-2501.04228-red?logo=arxiv)](https://arxiv.org/abs/2501.04228)

Contact:

- Yu ISHIHARA: yu.ishihara@sony.com

## Prerequisites

- Install pytorch by following the instruction on the [official website](https://pytorch.org/).
- Install [gymnasium](https://github.com/Farama-Foundation/Gymnasium)
```sh
$ pip install gymnasium
```
- Install car
```sh
$ pip install .
```

## How to run the training script

Go to [scripts](./scripts/) and execute the train_pendulum.py.

```sh
$ cd scripts
# It runs with QRSAC algorithm with CaR by default
# The default constraint is episodic. Eq. (18) of the paper.
$ python train_pendulum.py
```

To run the algorithm with timestep constraint (Eq. (17)) do:

```sh
$ python train_pendulum.py --constraint-type timestep
```

To run QRSAC without CaR (Training with original rewards) do:

```sh
$ python train_pendulum.py --without-car
```

## Citation
```
@article{ishihara2025car,
         title={Constraints as Rewards: Reinforcement Learning for Robots without Reward Functions}, 
         author={Yu Ishihara and Noriaki Takasugi and Kotaro Kawakami and Masaya Kinoshita and Kazumi Aoyama},
         journal={arXiv preprint arXiv:2501.04228},
         year={2025},
}
```

## License

- MIT License.
