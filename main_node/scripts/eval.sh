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
## $3 = dataset name: lvis or coco

if [ ! $# -eq 2 ]; then
  echo "Wrong number of arguments"
  exit 1
elif [ "$1" = "val" ]; then
  if [ "$2" = "coco" ]; then
    ANNOTATION_FILE=$DATA_DIR/coco/annotations/instances_val2017.json
    RESULT_FILE=$DATA_DIR/mask_rcnn_coco_results/val/$2/test/mask_rcnn_test_results.bbox.json
  elif [ "$2" = "lvis"]; then
    ANNOTATION_FILE=$DATA_DIR/lvis_v1/annotations/lvis_v1_val.json
    RESULT_FILE=$DATA_DIR/mask_rcnn_lvis_results/val/$2/test/mask_rcnn_test_results.bbox.json
  fi
elif [ "$1" = "train" ]; then
  if [ "$2" = "coco" ]; then
    ANNOTATION_FILE=$DATA_DIR/coco/annotations/instances_train2017.json
    RESULT_FILE=$DATA_DIR/mask_rcnn_coco_results/train/$2/test/mask_rcnn_test_results.bbox.json
  elif [ "$2" = "lvis"]; then
    ANNOTATION_FILE=$DATA_DIR/lvis_v1/annotations/lvis_v1_train.json
    RESULT_FILE=$DATA_DIR/mask_rcnn_lvis_results/train/$2/test/mask_rcnn_test_results.bbox.json
  fi
fi

echo "eval script is being run with $1 dataset type.. and $2 dataset"
echo "using the annotation file: $ANNOTATION_FILE"
echo "using the result file: $RESULT_FILE"

python3 $DEV_DIR/LRP-Error/lvis-api/demo.py \
    $ANNOTATION_FILE \
    $RESULT_FILE \
    bbox \
    $1
