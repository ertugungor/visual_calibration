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

## $1 = dataset type "train" or "val"
## $2 = iteration number: (e.g. 1,2)
## $3 = dataset name: lvis or coco

if [ ! $# -eq 3 ]; then
  echo "Wrong number of arguments"
  exit 1
elif [ "$1" = "val" ]; then
  if [ "$3" = "coco" ]; then
    ANNOTATION_FILE=$DATA_DIR/coco/annotations/instances_val2017.json
    # RESULT_FILE=$DATA_DIR/mask_rcnn_coco_results/val/$2/test/mask_rcnn_test_results.bbox.json
    # RESULT_FILE=$DATA_DIR/experiments/hyper_param_expr/coco/val_shift_topk_100/test/$2/val_test_results.bbox.json
    # RESULT_FILE=$DATA_DIR/experiments/hyper_param_expr/coco_subset/val_shift_topk_100/test/$2/val_test_results.bbox.json

    # 2x training results
    # RESULT_FILE=$DATA_DIR/experiments/hyper_param_expr/coco_subset/val_shift_topk_100_2x/test/$2/val_test_results.bbox.json

    # 2x + oversampling training results
    # RESULT_FILE=$DATA_DIR/experiments/hyper_param_expr/coco_subset/val_shift_topk_100_2x_oversampling/test/$2/val_test_results.bbox.json

    # 2x + oversampling training results (train shift)
    # RESULT_FILE=$DATA_DIR/experiments/hyper_param_expr/coco_subset/train_shift_topk_100_2x_oversampling/test/$2/val_test_results.bbox.json

    # 2x + oversampling training results + multi iteration shifting
    # RESULT_FILE=/home/ertugrul/data/experiments/hyper_param_expr/coco_subset/val_shift_topk_100_2x_oversampling/test/mult_iter/0.2/0.2/val_test_results.bbox.json
    # RESULT_FILE=/home/ertugrul/data/experiments/hyper_param_expr/coco_subset/val_shift_topk_100_2x_oversampling/test/mult_iter/0.2/0.2/0.2/val_test_results.bbox.json
    # RESULT_FILE=/home/ertugrul/data/experiments/hyper_param_expr/coco_subset/val_shift_topk_100_2x_oversampling/test/mult_iter/0.4/0.4/val_test_results.bbox.json
    RESULT_FILE=/home/ertugrul/data/experiments/hyper_param_expr/coco_subset/val_shift_topk_100_2x_oversampling/test/mult_iter/0.4/0.4/0.4/val_test_results.bbox.json
  elif [ "$3" = "lvis" ]; then
    ANNOTATION_FILE=$DATA_DIR/lvis_v1/annotations/lvis_v1_val.json
    # RESULT_FILE=$DATA_DIR/mask_rcnn_lvis_results/val/$2/test/mask_rcnn_test_results.bbox.json
    RESULT_FILE=$DATA_DIR/experiments/hyper_param_expr/test/$2/val_mask_rcnn_test_results.bbox.json
    # RESULT_FILE=$DATA_DIR/max_det_expr/max_det_500/$2/val/test/val_mask_rcnn_test_results.bbox.json
    # RESULT_FILE=$DATA_DIR/num_det_expr/$2/val_mask_rcnn_test_results.bbox.json
  fi
elif [ "$1" = "train" ]; then
  if [ "$3" = "coco" ]; then
    ANNOTATION_FILE=$DATA_DIR/coco/annotations/instances_train2017.json
    # RESULT_FILE=$DATA_DIR/mask_rcnn_coco_results/train/$2/test/mask_rcnn_test_results.bbox.json

    # 2x + oversampling training results (train shift)
    RESULT_FILE=$DATA_DIR/experiments/hyper_param_expr/coco_subset/train_shift_topk_100_2x_oversampling/test/$2/train_test_results.bbox.json
  elif [ "$3" = "lvis" ]; then
    ANNOTATION_FILE=$DATA_DIR/lvis_v1/annotations/lvis_v1_train.json
    RESULT_FILE=$DATA_DIR/mask_rcnn_lvis_results/train/$2/test/mask_rcnn_test_results.bbox.json
    RESULT_FILE=$DATA_DIR/max_det_expr/max_det_500/$2/train/test/train_mask_rcnn_test_results.bbox.json
    RESULT_FILE=$DATA_DIR/num_det_expr/$2/val_mask_rcnn_test_results.bbox.json
  fi
fi

if [ "$3" = "coco" ]; then
  SCRIPT_FILE=$DEV_DIR/LRP-Error/pycocotools/demo.py
elif [ "$3" = "lvis" ]; then
  SCRIPT_FILE=$DEV_DIR/LRP-Error/lvis-api/demo.py
else
  echo "Unknown dataset!"
  exit 1
fi

echo "eval script is being run with $1 dataset type and $3 dataset"
echo "the script $SCRIPT_FILE will be run.."
echo "using the annotation file: $ANNOTATION_FILE"
echo "using the result file: $RESULT_FILE"

python3 $SCRIPT_FILE \
        $ANNOTATION_FILE \
        $RESULT_FILE \
        bbox \
        $1
