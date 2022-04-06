# DEV_DIR and DATA_DIR are mounted directories into singularity container

DEV_DIR=/workspace/dev
DATA_DIR=/workspace/data

# After obtaining results with dist_test,
# 1- Put resulting json objects to a proper place
# 2- Provide that path as 2nd parameter to demo.py as below
# 3- Depending on if the results are for val set or train set
#    pass the proper annotation json file as 1st parameter.
# 4- Clearly we look for bbox results.(3rd parameter)
# 5- This script produces lrp optimal results, move them to a proper place

## $1= dataset type "train" or "val"
## #2= iteration number: (e.g. 1,2)

if [[ ! $# -eq 2 ]]; then
  echo "Wrong number of arguments"
  exit 1
elif [ "$1" == "val" ]; then
    python3 $DEV_DIR/LRP-Error/lvis-api/demo.py \
        $DATA_DIR/lvis_v1/annotations/lvis_v1_val.json \
        $DATA_DIR/mask_rcnn_lvis_results/val_set/$2/mask_rcnn_test_results.bbox.json \
        bbox
  elif [ "$1" == "train" ]; then
    python3 $DEV_DIR/LRP-Error/lvis-api/demo.py \
        $DATA_DIR/lvis_v1/annotations/lvis_v1_train.json \
        $DATA_DIR/mask_rcnn_lvis_results/train_set/$2/mask_rcnn_test_results.bbox.json \
        bbox
  fi
fi

