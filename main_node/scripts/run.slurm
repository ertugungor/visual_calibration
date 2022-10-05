#!/bin/bash
#SBATCH --nodes=1
#SBATCH -J sample                                                 # job name
###SBATCH --workdir=/home/ertugungor/                             # working directory
#SBATCH --gres=gpu:2
#SBATCH --output=/home/ertugrul/logs/%j.out     # output file
#SBATCH --error=/home/ertugrul/logs/%j.err      # error file

HOME_DIR=/home/ertugrul
DATA_DIR=$HOME_DIR/data
DEV_DIR=$HOME_DIR/dev

# singularity exec \
#     --nv \
#     --bind $DATA_DIR:/workspace/data,$DEV_DIR:/workspace/dev \
#     library://ertugungor/calibration/default:v2 \
#     python $DEV_DIR/mmdetection/tools/train.py $DEV_DIR/mmdetection/configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco_subset_train.py
#     # python $DEV_DIR/mmdetection/tools/train.py $DEV_DIR/mmdetection/configs/mask_rcnn/mask_rcnn_r50_fpn_1x_coco_subset_train.py
#     # python3 $DEV_DIR/coco_subsample.py
#     # python $DEV_DIR/mmdetection/tools/train.py $DEV_DIR/mmdetection/configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py
# exit 0

## $1 = eval or test
## $2 = val or train
## $3 = iteration number: (e.g. 1,2) for eval.sh / dataset name: lvis or coco for dist_test.sh
## $4 = dataset name: lvis or coco for eval.sh

if [ "$1" = "test" ]; then
  # # Run dist_test.sh to obtain model results
  singularity exec \
    --nv \
    --bind $DATA_DIR:/workspace/data,$DEV_DIR:/workspace/dev \
    library://ertugungor/calibration/default:v2 $DEV_DIR/visual_calibration/main_node/scripts/dist_test.sh $2 $3
else [ "$1" = "eval" ]
  # Run eval.sh to evaluate model results from previous step (AP, oLRP computation)
  singularity exec \
    --nv \
    --bind $DATA_DIR:/workspace/data,$DEV_DIR:/workspace/dev \
    library://ertugungor/calibration/default:v2 $DEV_DIR/visual_calibration/main_node/scripts/eval.sh $2 $3 $4
fi

## sbatch /home/ertugrul/dev/visual_calibration/main_node/scripts/run.slurm test train/val lvis/coco
## sbatch /home/ertugrul/dev/visual_calibration/main_node/scripts/run.slurm eval train/val 1/2/3 lvis/coco
