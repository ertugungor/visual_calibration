from collections import defaultdict
import json
import random
import copy
import mmdet.datasets.coco as coco

filepath = '/home/ertugrul/data/coco/annotations/coco_subsample_train.json'
filepath = '/home/ertugrul/data/coco/annotations/instances_train2017.json'

# with open('/home/ertugrul/data/coco/annotations/instances_val2017.json', 'r') as val_annotations:
with open(filepath, 'r') as annotations:
  json_data = json.load(annotations)

dataset = coco.CocoDataset(filepath, [], test_mode=True)

def get_group_freqs(img_lists):
  freq_groups = {"rare":0, "common":0, "frequent":0}
  for img_list in img_lists:
    size = len(set(img_list))
    if size <= 10:
      freq_groups["rare"]+=1
    elif size <=100:
      freq_groups["common"]+=1
    else:
      freq_groups["frequent"]+=1
  return freq_groups

def check_removability(img, target):
  cat_list = img_idx_cat_id_list[img]
  img_lists = []
  for cat in cat_list:
    img_lists.append(cat_img_list[cat])
  group_freqs = get_group_freqs(img_lists)
  max_key = max(group_freqs, key=group_freqs.get)
  if max_key == target:
    return False
  return True

def update(cat_id_img_freq, cat_set):
  for cat in cat_set:
    if (cat,1) in cat_id_img_freq.items():
      return False
  for cat in cat_set:
    cat_id_img_freq[cat]-=1
  return True

def get_status(cat_id_img_freq):
  freq_groups = {"rare":0, "common":0, "frequent":0}
  for _, freq in cat_id_img_freq.items():
    if freq <= 1000:
      freq_groups["rare"]+=1
    elif freq <= 2500:
      freq_groups["common"]+=1
    else:
      freq_groups["frequent"]+=1
  return freq_groups

def get_freq_group_elements(cat_id_img_freq):
  freq_group_cat_id = {"rare":[], "common":[], "frequent":[]}
  for cat_id, img_freq in cat_id_img_freq.items():
    if img_freq <= 1000:
      freq_group_cat_id["rare"].append(cat_id)
    elif img_freq <= 2500:
      freq_group_cat_id["common"].append(cat_id)
    else:
      freq_group_cat_id["frequent"].append(cat_id)

  print(f"freq_group_cat_id: {freq_group_cat_id}")

# set of unique categories in each image
img_idx_cat_id_set = defaultdict(set)
# set of images a category appears in
cat_id_img_idx_set = defaultdict(set)
cat_id_img_freq = defaultdict(int)

print(f"# of images: {len(json_data['images'])}")

for img_idx in range(len(json_data['images'])):
  img_id =  json_data['images'][img_idx]['id']
  ann_ids = dataset.coco.get_ann_ids(img_ids=[img_id])
  anns = dataset.coco.load_anns(ann_ids)
  for ann in anns:
    cat_id = ann["category_id"]
    cat_id_img_idx_set[cat_id].add(img_idx)
    img_idx_cat_id_set[img_idx].add(cat_id)

for cat_id,img_idx_set in cat_id_img_idx_set.items():
  cat_id_img_freq[cat_id]=len(img_idx_set)

print("frequency group info before subsampling..")
get_freq_group_elements(cat_id_img_freq)

result_img_idx = copy.deepcopy(list(img_idx_cat_id_set.keys()))
img_idx_cat_id_set = dict(sorted(img_idx_cat_id_set.items(), key=lambda x: len(x[1]), reverse=True))
dataset_size = len(img_idx_cat_id_set.keys())
min_dataset_size = 0.05 * dataset_size

print(f"initial dataset size: {dataset_size}")
print(f"initial frequency groups: {get_status(cat_id_img_freq)}")
subset_data = {'info': json_data['info'], 'licenses': json_data['licenses'], 'images': list(), 'annotations': list(), 'categories': json_data['categories']}

for img_idx, cat_set in img_idx_cat_id_set.items():
  if (dataset_size <= min_dataset_size):
    break
  if not update(cat_id_img_freq, cat_set):
    continue
  status = get_status(cat_id_img_freq)
  if status["rare"] >= 27 or status["frequent"] <= 29:
    break
  result_img_idx.remove(img_idx)
  dataset_size-=1

print(f"size of result_imgs: {len(result_img_idx)}")
print(f"dataset_size: {dataset_size}")
print(f"frequency groups after subsampling: {get_status(cat_id_img_freq)}")
sorted_cat_id_img_freq = dict(sorted(cat_id_img_freq.items(), key=lambda x: x[1], reverse=True))
print(f"category to image frequency: {sorted_cat_id_img_freq}")

print("frequency group info after subsampling..")
get_freq_group_elements(cat_id_img_freq)

for img_idx in result_img_idx:
  subset_data['images'].append(json_data['images'][img_idx])
  img_id =  json_data['images'][img_idx]['id']
  ann_ids = dataset.coco.get_ann_ids(img_ids=[img_id])
  anns = dataset.coco.load_anns(ann_ids)
  for ann in anns:
      subset_data['annotations'].append(ann)

with open('/home/ertugrul/coco_subsample.json', 'w') as out_file:
  json.dump(subset_data, out_file)

'''

1- iterate over annotations and build a map for (image_id, category_list) => img_idx_cat_id_list
2- iterate over annotations and build a map for (category_id, image_frequency) => cat_id_img_freq
3- sort `img_idx_cat_id_list` map based on their category_list size in decreasing order
 - remove the first image, update `cat_id_img_freq`
 - check percentages of 3 groups, stop or continue.



{
"segmentation": [[300.41,383.06,297.55,386.23,288.99,375.46,285.19,363.09,284.56,349.14,282.02,339.95,286.77,345.34,292.8,348.19,295.33,352.0,301.04,366.9,301.36,382.43],[283.29,326.95,283.29,318.71,290.58,318.39,287.41,323.15,284.56,326.64]],
"area": 485.4098500000001,
"iscrowd": 0,
"image_id": 363188,
"bbox": [282.02,318.39,19.34,67.84],
"category_id": 44,
"id": 1867794
},


'''
