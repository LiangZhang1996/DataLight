# 1. Introduction
Official code for article [DataLight: Offline Data-Driven Traffic Signal Control](https://arxiv.org/abs/2303.10828)


# 2. Requirements
`python=3.6`, `tensorflow=2.4`, `cityflow`

# 3. Quick start
## 3.1 Prepare offline dataset
Refer to `./memory/readme.md` to prepare the offline dataset.

`random10.pkl`, `medium10.pkl`, and  `expert10.pkl` are the default offline datsets.

## 3.2 Model training
### 3.2.1 Base model
Configure the used memory at line `28` and run `run_DataLight.py`

### 3.2.2 Run baselines
For online RL baselines, utilize [AttentionLight](https://github.com/LiangZhang1996/AttentionLight.git)

For offline RL baselines:
- For BC, configure the used memory and run `run_BC.py`
- For BCQ, configure the used memory and run `run_BCQ.py`
- For CQL, configure the used memory and run `run_CQL.py`
- For DTL, configure the used memory and run `run_DT.py`

## 3.3 Model testing
Configure the save directory and run `run_test.py`, `run_test_BC.py`, and `run_test_DT.py` to test the model on other datasets.

## 3.4 Ablation study and case study
- For low data scenario, configure the data percent at `line 41` and run `run_data_perccent.py`
- For state study, configure the state representations at `line 60` and run `run_ablation1.py`
- For reward study, configure the reward at `line 61` and run `run_DataLight.py`
- For network w/o sequential encoding, configure the state `line 60` and run `run_ablation1.py`
- For training with cyclical offline dataset, configure the used memory as `cycle20.pkl` and run `run_DataLight.py`


## License

This project is licensed under the GNU General Public License version 3 (GPLv3) - see the LICENSE file for details.

