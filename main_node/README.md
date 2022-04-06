1- Download lvis dataset and prepare lvis api
2- Download checkpoints for mmdetection model
3- Change the file mmdetection/configs/_base_/datasets/lvis_v1_instance.py:
  - To change annotations path. dist_test.sh runs test config from here. So put validation or train set to test dict in this file.
4- Run dist_test.sh and eval.sh to first run model and then evaluate results and save LRP-opt thresholds.