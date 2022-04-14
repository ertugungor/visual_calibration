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

## $1 = eval or test
## $2 = val or train
## $3 = iteration number: (e.g. 1,2) for eval / dataset name: lvis or coco for test
## $4 = dataset name: lvis or coco for eval

if [ "$1" = "test" ]; then
  # # Run dist_test.sh to obtain model results
  singularity exec \
    --nv \
    --bind $DATA_DIR:/workspace/data,$DEV_DIR:/workspace/dev \
    library://ertugungor/default/image:latest $DEV_DIR/visual_calibration/main_node/scripts/dist_test.sh $2 $3
else [ "$1" = "eval" ]
  # Run eval.sh to evaluate model results from previous step (AP, oLRP computation)
  singularity exec \
    --nv \
    --bind $DATA_DIR:/workspace/data,$DEV_DIR:/workspace/dev \
    library://ertugungor/default/image:latest $DEV_DIR/visual_calibration/main_node/scripts/eval.sh $2 $3 $4
fi

## sbatch /home/ertugrul/dev/visual_calibration/main_node/scripts/run.slurm test train/val lvis/coco
## sbatch /home/ertugrul/dev/visual_calibration/main_node/scripts/run.slurm eval train/val 1/2/3 lvis/coco
