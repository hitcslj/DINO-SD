# DINO-SD
### [Paper](https://arxiv.org/abs/2405.17102) | [Data]() | [中文解读]()
<br/>

> DINO-SD: Champion Solution for ICRA 2024 RoboDepth Challenge

>[Yifan Mao*](), [Ming Li*](), [Jian Liu*](https://hitcslj.github.io/), [Jiayang Liu](), [Zihan Qin](), [Chunxi Chu](), [Jialei Xu](), [Wenbo Zhao](), [Junjun Jiang](), [Xianming Liu]() 

<p align='center'>
<img src="assets/dino_sd.png" width='80%'/>
</p>


## Introduction
Surround-view depth estimation is a crucial task aims to acquire the depth maps of the surrounding views. It has many applications in real world scenarios such as autonomous driving, AR/VR and 3D reconstruction, etc. However, given that most of the data in the autonomous driving dataset is collected in daytime scenarios, this leads to poor depth model performance in the face of out-of-distribution(OoD) data. While some works try to improve the robustness of depth model under OoD data, these methods either require additional training data or lake generalizability. In this report, we introduce the DINO-SD, a novel surround-view depth estimation model. Our DINO-SD does not need additional data and has strong robustness. Our DINO-SD get the best performance in the track4 of ICRA 2024 RoboDepth Challenge.

## Model Zoo

| model     | dataset | Abs Rel ↓  | Sq Rel  ↓ | RMSE ↓  | RMSE Log  ↓ | a1 ↑ | a2 ↑ | a3 ↑ | download |  
|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|
| Baseline | nuScenes | 0.304 | 3.060 | 8.527 |  0.400 | 0.544 | 0.784 | 0.891 |  [model](https://cloud.tsinghua.edu.cn/f/72717fa447f749e38480/?dl=1) |
| DINO-SD | nuScenes | 0.187 | 1.468 | 6.236 | 0.276 | 0.734 | 0.895 | 0.953 | [model]() |

## Install
* python 3.8, pytorch 2.2.2, CUDA 11.4, RTX 3090
```bash
git clone https://github.com/hitcslj/DINO-SD.git
conda create -n dinosd python=3.8
conda activate dinosd
pip install -r requirements.txt
```
Since we use [dgp codebase](https://github.com/TRI-ML/dgp) to generate groundtruth depth, you should also install it. 

## Data Preparation
Datasets are assumed to be downloaded under `data/<dataset-name>`.


### nuScenes
* Please download the official [nuScenes dataset](https://www.nuscenes.org/download) to `data/nuscenes/raw_data`
* Export depth maps for evaluation 
```bash
cd tools
python export_gt_depth_nusc.py val
```
* The final data structure should be:
```
DINO-SD
├── data
│   ├── nuscenes
│   │   │── raw_data
│   │   │   │── samples
|   |   |   |── sweeps
|   |   |   |── maps
|   |   |   |── v1.0-trainval
|   |   |── depth
│   │   │   │── samples
```

## Training
Train model.
```bash
python -m torch.distributed.launch --nproc_per_node 8 --num_workers=8 run.py  --model_name nusc  --config configs/nusc.txt 
```

## Evaluation
```bash
python -m torch.distributed.launch --nproc_per_node ${NUM_GPU}  run.py  --model_name test  --config configs/${TYPE}.txt --models_to_load depth encoder   --load_weights_folder=${PATH}  --eval_only 
```

## Acknowledgement

Our code is based on [SurroundDepth](https://github.com/weiyithu/SurroundDepth).

## Citation

If you find this project useful in your research, please consider cite:
```
@article{mao2024dino,
  title={DINO-SD: Champion Solution for ICRA 2024 RoboDepth Challenge},
  author={Mao, Yifan and Li, Ming and Liu, Jian and Liu, Jiayang and Qin, Zihan and Chu, Chunxi and Xu, Jialei and Zhao, Wenbo and Jiang, Junjun and Liu, Xianming},
  journal={arXiv preprint arXiv:2405.17102},
  year={2024}
}
```