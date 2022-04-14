# DEV_DIR and DATA_DIR are mounted directories into singularity container

DEV_DIR=/workspace/dev
DATA_DIR=/workspace/data

: '
dist_test.sh
config file
checkpoint file
# of gpus, misc evaluation parameters
'

## $1= dataset type "train" or "val"
## $2= dataset name "coco" or "lvis" currently

#file duplication is temporary solution
if [ ! $# -eq 2 ]; then
  echo "Wrong number of arguments"
  exit 1
elif [ "$1" = "val" ]; then
  if [ "$2" = "coco" ]; then
    CHECKPOINT_FILE=$DATA_DIR/mmdetection/checkpoints/mask_rcnn_r50_fpn_1x_coco_20200205-d4b0c5d6.pth
    CONFIG_FILE=$DEV_DIR/mmdetection/configs/mask_rcnn/mask_rcnn_r50_fpn_1x_coco_val.py
  elif [ "$2" = "lvis"]; then
    CHECKPOINT_FILE=$DATA_DIR/mmdetection/checkpoints/mask_rcnn_r50_fpn_sample1e-3_mstrain_1x_lvis_v1-aa78ac3d.pth
    CONFIG_FILE=$DEV_DIR/mmdetection/configs/lvis/mask_rcnn_r50_fpn_sample1e-3_mstrain_1x_lvis_v1_val.py
  fi
elif [ "$1" = "train" ]; then
  if [ "$2" = "coco" ]; then
    CHECKPOINT_FILE=$DATA_DIR/mmdetection/checkpoints/mask_rcnn_r50_fpn_1x_coco_20200205-d4b0c5d6.pth
    CONFIG_FILE=$DEV_DIR/mmdetection/configs/mask_rcnn/mask_rcnn_r50_fpn_1x_coco_train.py
  elif [ "$2" = "lvis" ]; then
    CHECKPOINT_FILE=$DATA_DIR/mmdetection/checkpoints/mask_rcnn_r50_fpn_sample1e-3_mstrain_1x_lvis_v1-aa78ac3d.pth
    CONFIG_FILE=$DEV_DIR/mmdetection/configs/lvis/mask_rcnn_r50_fpn_sample1e-3_mstrain_1x_lvis_v1_train.py
  fi
fi

echo "dist_test scripts is being run with $1 dataset type.. and $2 dataset"
echo "using the config file: $CONFIG_FILE"
echo "using the checkpoint file: $CHEKPOINT_FILE"

$DEV_DIR/mmdetection/tools/dist_test.sh \
  $CONFIG_FILE \
  $CHECKPOINT_FILE \
  2 --eval bbox --eval-options "jsonfile_prefix=$DATA_DIR/mask_rcnn_test_results"

# Model checkpoints

# Mask R-CNN trained on COCO
# https://download.openmmlab.com/mmdetection/v2.0/mask_rcnn/mask_rcnn_r50_fpn_1x_coco/mask_rcnn_r50_fpn_1x_coco_20200205-d4b0c5d6.pth

# Mask R-CNN trained on LVIS v1
# https://download.openmmlab.com/mmdetection/v2.0/lvis/mask_rcnn_r50_fpn_sample1e-3_mstrain_1x_lvis_v1/mask_rcnn_r50_fpn_sample1e-3_mstrain_1x_lvis_v1-aa78ac3d.pth
