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

# file duplication is temporary solution
if [[ ! $# -eq 1 ]]; then
  echo "Wrong number of arguments"
  exit 1
elif [ "$1" == "val" ]; then
  $CONFIG_FILE=$DEV_DIR/mmdetection/configs/lvis/mask_rcnn_r50_fpn_sample1e-3_mstrain_1x_lvis_v1_val.py 
elif [ "$1" == "train" ]; then
  $CONFIG_FILE=$DEV_DIR/mmdetection/configs/lvis/mask_rcnn_r50_fpn_sample1e-3_mstrain_1x_lvis_v1_train.py 
fi

$DEV_DIR/mmdetection/tools/dist_test.sh \
  $CONFIG_FILE \
  $DATA_DIR/mmdetection/checkpoints/mask_rcnn_r50_fpn_sample1e-3_mstrain_1x_lvis_v1-aa78ac3d.pth \
  2 --eval --eval-options "jsonfile_prefix=$DATA_DIR/mask_rcnn_test_results"
