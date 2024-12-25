
The official implementation of AAAI2025 paper "DSRC: Learning Density-insensitive and Semantic-aware V2X Collaborative Representation against Corruptions.".![DSRC_Overview](https://github.com/Terry9a/DSRC/blob/main/image.png)

> [**DSRC: Learning Density-insensitive and Semantic-aware V2X Collaborative Representation against Corruptions**](https://arxiv.org/abs/2412.10739), <br>
> Jingyu Zhang* , Yilei Wang*, Lang Qian, Peng Sun, Zengwen Li†,Sudong Jiang, Maolin Liu, Liang Song† <br>
> Accepted by AAAI2025

## Abstract

As a potential application of Vehicle-to-Everything (V2X) communication, multi-agent collaborative perception has achieved significant success in 3D object detection. While these methods have demonstrated impressive results on standard benchmarks, the robustness of such approaches in the face of complex real-world environments requires additional verification. To bridge this gap, we introduce the first comprehensive benchmark designed to evaluate the robustness of collaborative perception methods in the presence of natural corruptions typical of real-world environments. Furthermore, we propose DSRC, a robustness-enhanced collaborative perception method aiming to learn Densityinsensitive and Semantic-aware collaborative Representation against Corruptions. DSRC consists of two key designs: i) a semantic-guided sparse-to-dense distillation framework, which constructs multi-view dense objects painted by ground truth bounding boxes to effectively learn density insensitive and semantic-aware collaborative representation; ii) a feature-to-point cloud reconstruction approach to better fuse critical collaborative representation across agents. To thoroughly evaluate DSRC, we conduct extensive experiments on real-world and simulated datasets. The results demonstrate that our method outperforms SOTA collaborative perception methods in both clean and corrupted conditions.

## Installation

```bash
# Setup conda environment
conda create -n dsrc python=3.7 -y
conda activate dsrc

pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html

# spconv 2.0 install, choose the correct cuda version for you
pip install spconv-cu113

# Install dependencies
pip install -r requirements.txt
# Install bbx nms calculation cuda version
python opencood/utils/setup.py build_ext --inplace

# install opencood into the environment
python setup.py develop
```

## Data Downloading
### 1. OPV2V
All the data can be downloaded from [google drive](https://drive.google.com/drive/folders/1dkDeHlwOVbmgXcDazZvO6TFEZ6V_7WUu). If you have a good internet, you can directly
download the complete large zip file such as `train.zip`. In case you suffer from downloading large files, we also split each data set into small chunks, which can be found 
in the directory ending with `_chunks`, such as `train_chunks`. After downloading, please run the following command to each set to merge those chunks together:
```python
cat train.zip.part* > train.zip
unzip train.zip
```
### 2. DAIR-V2X
Download the raw data of [DAIR-V2X](https://thudair.baai.ac.cn/index) and  the [complemeted annotations](https://siheng-chen.github.io/dataset/dair-v2x-c-complemented/).


## Quick Start


### **Train** 
### Step1: Train the teacher model

First, ensure that the `root_dir` in the YAML file (e.g., `opencood/hypes_yaml/point_pillar_base_multi_scale_teacher.yaml`) is set to the training dataset path, such as `opv2v/train`.
Then, you can use the following command to train your teacher model from scratch or a continued checkpoint:

```python
python opencood/tools/train.py --hypes_yaml ${CONFIG_FILE} [--model_dir  ${CHECKPOINT_FOLDER}]
```
The explanation of the optional arguments are as follows:

- `hypes_yaml`: the path of the training configuration file, e.g.  `opencood/hypes_yaml/point_pillar_base_multi_scale_teacher.yaml`. You can change the configuration parameters in this provided yaml file.
- `model_dir` (optional) : the path of the checkpoints. This is used to fine-tune the trained models. When the `model_dir` is
given, the trainer will discard the `hypes_yaml` and load the `config.yaml` in the checkpoint folder.


### Step2: Train the student model

Copy the checkpoint folder of the teacher model and rename it as `student_train_folder`, keep only the last checkpoint and rename it as epoch_1.pth, then change the `core_method` in config.yaml under the checkpoint folder to `point_pillar_base_multi_scale_student`.

```python
python opencood/tools/train.py --model_dir student_train_folder
```


### **Test** 
Before you run the following command, first make sure the `validation_dir` in config.yaml under your checkpoint folder
refers to the testing dataset path, e.g. `opv2v/test`.

```python
python opencood/tools/inference.py --model_dir ${student_train_folder} 
```
If you want to test the performance of the model in different environments, you can uncomment the application code for each environment simulation function in the file `opencood/data_utils/datasets/basedataset.py`, such as `apply_motion_blur_to_numpy`.

## Citation
 If you are using our DSRC for your research, please cite the following paper:
  ```bibtex
@article{zhang2024dsrc,
  title={DSRC: Learning Density-insensitive and Semantic-aware Collaborative Representation against Corruptions},
  author={Zhang, Jingyu and Wang, Yilei and Qian, Lang and Sun, Peng and Li, Zengwen and Jiang, Sudong and Liu, Maolin and Song, Liang},
  journal={arXiv preprint arXiv:2412.10739},
  year={2024}
}
```
## Acknowledgment
Many thanks to the high-quality dataset and codebase, including [Robo3D](https://github.com/ldkong1205/Robo3D), [OpenCOOD](https://github.com/DerrickXuNu/OpenCOOD) and  [Where2comm](https://github.com/MediaBrain-SJTU/Where2comm.git).
