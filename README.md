# Efficient One-stage Video Object Detection by Exploiting Temporal Consistency (ECCV22)

[![License](https://img.shields.io/badge/license-BSD-blue.svg)](LICENSE)

By [Guanxiong Sun](https://sunguanxiong.github.io).

This repo contains the PyTorch implementations of the paper "Efficient One-stage Video Object Detection by Exploiting Temporal Consistency" published in ECCV 2022.

The code based on two open-source toolboxes: [mmtracking](https://github.com/open-mmlab/mmtracking) and [mmdetection](https://github.com/open-mmlab/mmdetection).

## Main Results

Pretrained models are now available at [Baidu](https://pan.baidu.com/s/1qjIAD3ohaJO8EF1mZ4nLEg) (code: neck) and Google Drive.


|  Model  |  Backbone  |  AP  | AP50 | AP75 | AP small | AP medium | AP large | Link |
| :------: | :--------: | :--: | :--: | :--: | :------: | :-------- | :------- | :--- |
| FCOS+LPN | ResNet-101 | 54.0 | 79.7 | 59.3 |   9.8   | 26.6      | 60.4     | xxx  |

## Installation

### Requirements:

- python 3.7
- pytorch 1.8.0
- torchvision 0.9.0
- mmcv-full 1.3.17
- GCC 7.5.0
- CUDA 10.1

### Installation

```bash
# create conda environment
conda create --name eovod -y python=3.7
conda activate eovod

# install PyTorch 1.8.0 with cuda 10.2
conda install pytorch==1.8.0 torchvision==0.9.0 cudatoolkit=10.2 -c pytorch

# install mmcv-full 1.3.17
pip install mmcv-full==1.3.17 -f https://download.openmmlab.com/mmcv/dist/cu102/torch1.8.0/index.html

# install other requirements
pip install -r requirements.txt
```

See [here](https://github.com/open-mmlab/mmcv#installation) for different versions of MMCV compatible to different PyTorch and CUDA versions.

Optionally you can compile mmcv from source if you need to develop both mmcv and mmdet. Refer to the [guide](https://github.com/open-mmlab/mmcv#installation) for details.

## Data preparation

### Download Datasets

Please download ILSVRC2015 DET and ILSVRC2015 VID dataset from [here](http://image-net.org/challenges/LSVRC/2015/downloads). After that, we recommend to symlink the path to the datasets to `datasets/`. And the path structure should be as follows:

./data/ILSVRC/
./data/ILSVRC/Annotations/DET
./data/ILSVRC/Annotations/VID
./data/ILSVRC/Data/DET
./data/ILSVRC/Data/VID
./data/ILSVRC/ImageSets
**Note**: List txt files under `ImageSets` folder can be obtained from
[here](https://github.com/msracver/Flow-Guided-Feature-Aggregation/tree/master/data/ILSVRC2015/ImageSets).

### Convert Annotations

We use [CocoVID](mmdet/datasets/parsers/coco_video_parser.py) to maintain all datasets in this codebase. In this case, you need to convert the official annotations to this style. We provide scripts and the usages are as following:

```bash
# ImageNet DET
python ./tools/convert_datasets/ilsvrc/imagenet2coco_det.py -i ./data/ILSVRC -o ./data/ILSVRC/annotations

# ImageNet VID
python ./tools/convert_datasets/ilsvrc/imagenet2coco_vid.py -i ./data/ILSVRC -o ./data/ILSVRC/annotations

```
## Usage

### Training

#### Training on a single GPU

```shell
python tools/train.py ${CONFIG_FILE} [optional arguments]
```
#### Training on multiple GPUs

We provide `tools/dist_train.sh` to launch training on multiple GPUs.
The basic usage is as follows.

```shell
bash ./tools/dist_train.sh \
    ${CONFIG_FILE} \
    ${GPU_NUM} \
    [optional arguments]
```
Optional arguments remain the same as stated above.

If you would like to launch multiple jobs on a single machine, e.g., 2 jobs of 4-GPU training on a machine with 8 GPUs,
you need to specify different ports (29500 by default) for each job to avoid communication conflict.

If you use `dist_train.sh` to launch training jobs, you can set the port in commands.

```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=29500 ./tools/dist_train.sh ${CONFIG_FILE} 4
CUDA_VISIBLE_DEVICES=4,5,6,7 PORT=29501 ./tools/dist_train.sh ${CONFIG_FILE} 4
```
#### Example

1. Train EOVOD(FCOS) and then evaluate AP at the last epoch.

   ```shell
   ./tools/dist_train.sh configs/vid/time_swin_lite/faster_rcnn_time_swint_lite_fpn_0.000025_3x_tricks_stride3_train.py 8
   ```

### Inference

This section will show how to test existing models on supported datasets.
The following testing environments are supported:

- single GPU
- single node multiple GPU
- multiple nodes

During testing, different tasks share the same API and we only support `samples_per_gpu = 1`.

You can use the following commands for testing:

```shell
# single-gpu testing
python tools/test.py ${CONFIG_FILE} [--checkpoint ${CHECKPOINT_FILE}] [--out ${RESULT_FILE}] [--eval ${EVAL_METRICS}]

# multi-gpu testing
./tools/dist_test.sh ${CONFIG_FILE} ${GPU_NUM} [--checkpoint ${CHECKPOINT_FILE}] [--out ${RESULT_FILE}] [--eval ${EVAL_METRICS}]
```
Optional arguments:

- `CHECKPOINT_FILE`: Filename of the checkpoint. You do not need to define it when applying some MOT methods but specify the checkpoints in the config.
- `RESULT_FILE`: Filename of the output results in pickle format. If not specified, the results will not be saved to a file.
- `EVAL_METRICS`: Items to be evaluated on the results. Allowed values depend on the dataset, e.g., `bbox` is available for ImageNet VID, `track` is available for LaSOT, `bbox` and `track` are both suitable for MOT17.
- `--cfg-options`: If specified, the key-value pair optional cfg will be merged into config file
- `--eval-options`: If specified, the key-value pair optional eval cfg will be kwargs for dataset.evaluate() function, it’s only for evaluation
- `--format-only`: If specified, the results will be formatted to the official format.

#### Examples of testing VID model

Assume that you have already downloaded the checkpoints to the directory `checkpoints/`.

1. Test DFF on ImageNet VID, and evaluate the bbox mAP.

   ```shell
   python tools/test.py configs/vid/tdvit/dff_faster_rcnn_r101_dc5_1x_imagenetvid.py \
       --checkpoint checkpoints/dff_faster_rcnn_r101_dc5_1x_imagenetvid_20201218_172720-ad732e17.pth \
       --out results.pkl \
       --eval bbox
   ```
2. Test DFF with 8 GPUs on ImageNet VID, and evaluate the bbox mAP.

   ```shell
   ./tools/dist_test.sh configs/vid/dff/dff_faster_rcnn_r101_dc5_1x_imagenetvid.py 8 \
       --checkpoint checkpoints/dff_faster_rcnn_r101_dc5_1x_imagenetvid_20201218_172720-ad732e17.pth \
       --out results.pkl \
       --eval bbox
   ```
