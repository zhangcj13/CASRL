# CASRL
The source code for paper: [CASRL: Collision Avoidance with Spiking Reinforcement Learning Among Dynamic, Decision-Making Agents](https://doi.org/10.1109/IROS58592.2024.10802416).

<!-- To view our paper, please refer: [CASRL: Collision Avoidance with Spiking Reinforcement Learning Among Dynamic, Decision-Making Agents](https://doi.org/10.1109/IROS58592.2024.10802416).  -->

# Project Demo

<table>
  <tr>
    <td width="33%" align="center">
      <!-- <strong>Feature One</strong><br> -->
      <img src="images/random2.gif" alt="Feature One Demo" width="100%">
      <br>
      <em>4-random</em>
    </td>
    <td width="33%" align="center">
      <!-- <strong>Feature Two</strong><br> -->
      <img src="images/swap2.gif" alt="Feature Two Demo" width="100%">
      <br>
      <em>2-pair</em>
    </td>
    <td width="33%" align="center">
      <!-- <strong>Feature Three</strong><br> -->
      <img src="images/circle2.gif" alt="Feature Three Demo" width="100%">
      <br>
      <em>4-circle</em>
    </td>
  </tr>
</table>

## Method
![image](images/network.png)

## Prepare Python env 
python=3.7.16. if use conda, you can build python as:
```
conda create --name casrl python==3.7.16
conda activate casrl
```
```
# install torch
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
# install packages
pip install -r requirement.txt
# install baselines
pip install -e {your_dir}/baselines-master
# Tensorboard may conflict with the version in Torch, you can update it
pip install tensorboard==1.15.0
```

## Train 
4-agent stage1 
```
python train_dppo_stage1.py
```
10-agent stage2, load the weight from stage1
```
python train_dppo_stage2.py
```
For different ARCH and params, edit the TrainPhase_PPOS_STAGE in file './gym_collision_avoidance/envs/config.py'

Use
```
tensorboard --logdir=./runs , http://localhost:6006
```
to view the training process
## Eval 
```
python run_500eval.py
```
Choose 'NUM_AGENTS_TO_TEST' and 'POLICIES_TO_TEST' of FullEvalTest in file './gym_collision_avoidance/envs/config.py' for different testcase. You can also make your own policy in in './gym_collision_avoidance/envs/policies' for eval.

Use 
```
python plot_comparison.py
```
to plot eval result as:
![result](images/comparison.png)

## Citation
If our work help to your research, please cite our paper, thx.
```
@inproceedings{zhang2024casrl,
  title={Casrl: Collision avoidance with spiking reinforcement learning among dynamic, decision-making agents},
  author={Zhang, Chengjun and Yip, Ka-Wa and Yang, Bo and Zhang, Zhiyong and Yuan, Mengwen and Yan, Rui and Tang, Huajin},
  booktitle={2024 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
  pages={8031--8038},
  year={2024},
  organization={IEEE}
}
```

## Thanks to these amazing projects:
- [rl_collision_avoidance](https://github.com/mit-acl/rl_collision_avoidance)
- [gym-collision-avoidance](https://github.com/mit-acl/gym-collision-avoidance)
- [Transformers-RL](https://github.com/dhruvramani/Transformers-RL)