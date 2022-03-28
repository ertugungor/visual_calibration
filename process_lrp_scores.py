from collections import defaultdict
import pickle
import numpy as np
from matplotlib import pyplot as plt
from lvis_cls_mapping import index_to_freq_group
from lvis_cls_mapping import CLASSES
from os import path
from os import mkdir

RESULTS_PATH_PREFIX = "/workspace/lrp_score_plots/data/calibration/iterations"
SLRP_STATS_PATH_PREFIX = "/workspace/lrp_score_plots/slrp/stats"
SLRP_PLOTS_PATH_PREFIX = "/workspace/lrp_score_plots/slrp/plots/"

def transform(data, type):
  if type == "lrp":
    data = np.insert(data, 0, [1,1])
    data = np.insert(data, len(data), data[-1])
  elif type == "score":
    data = np.insert(data, 0, [1,data[0]])
    data = np.insert(data, len(data), 0)
  return data

def read_data(iterations, data_keys, dataset_types):
  iter_values_map = defaultdict(defaultdict)
  for iter in iterations:
    for dataset_type in dataset_types:
      for key in data_keys:
        file_name = path.join(RESULTS_PATH_PREFIX, str(iter), f"{dataset_type}", f"{dataset_type}_{key}")
        with open(file_name, 'rb') as in_file:
          iter_values_map[iter][f"{dataset_type}_{key}"] = pickle.load(in_file)
  return iter_values_map

def draw_slrp_plot(x_points, y_points, colors, curve_labels, axis_labels, title, fig_name):
  # plot val and train sLRP plots
  for idx in range(len(x_points)):
    plt.plot(x_points[idx], y_points[idx], colors[idx], label=curve_labels[idx])

  plt.legend()

  plt.xlim(-0.005, 1.005)
  plt.ylim(-0.005, 1.005)
  plt.xticks([0.00,0.20,0.40,0.60,0.80,1.00])
  plt.yticks([0.00,0.20,0.40,0.60,0.80,1.00])

  plt.xlabel(axis_labels[0], fontsize=10)
  plt.ylabel(axis_labels[1], fontsize=10)

  plt.title(title)
  plt.savefig(fig_name)
  plt.cla()

def create_plots(iterations):
  data_keys = ["lrp_values", "dt_scores", "tps", "fps"]
  dataset_types = ["val", "train"]
  results_data = read_data(iterations, data_keys, dataset_types)
  for iter in iterations:
    values = results_data[iter]
    val_lrp_values, val_dt_scores, val_tps, val_fps = values[0],values[1],values[2],values[3]
    train_lrp_values, train_dt_scores, train_tps, train_fps = values[4],values[5],values[6],values[7]

    dir_path = path.join(SLRP_PLOTS_PATH_PREFIX, str(iter))
    if not path.isdir(dir_path):
      mkdir(dir_path)

    for key in train_lrp_values:
      (cat_id,area_id) = key
      # consider "all" area type
      if area_id != 0:
        continue

      if key in val_lrp_values:
        val_lrps = val_lrp_values[key]
      else:
        print(f"val_lrp_values missing key: {key}")
        val_lrps = None
      if key in val_dt_scores:
        val_scores = val_dt_scores[key]
      else:
        val_scores = None
        print(f"val_dt_scores missing key: {key}")
      if key in train_lrp_values:
        train_lrps = train_lrp_values[key]
      else:
        train_lrps = None
        print(f"train_lrp_values missing key: {key}")
      if key in train_dt_scores:
        train_scores = train_dt_scores[key]
      else:
        train_scores = None
        print(f"train_dt_scores missing key: {key}")

      # eliminate missing values
      if val_lrps is None or val_scores is None:
        print(f"val missing value with key: {key}")
      if train_lrps is None or train_scores is None:
        print(f"train missing value with key: {key}")

      # plot val and train sLRP plots
      x_points = []
      y_points = []
      colors = []
      curve_labels = []
      axis_labels = ["score", "lrp"]
      cls_freq_group = index_to_freq_group(cat_id)
      cls_name = CLASSES[cat_id]
      fig_name = path.join(dir_path, f"{cat_id}_{cls_name}.png")
      if not path.isdir(dir_path):
        mkdir(dir_path)

      title = f"{cls_name}({cls_freq_group})"
      if (train_scores is not None and train_lrps is not None):
        x_points.append(transform(train_scores, "score"))
        y_points.append(transform(train_lrps, "lrp"))
        colors.append("b")
        curve_labels.append("train")
      if (val_scores is not None and val_lrps is not None):
        x_points.append(transform(val_scores, "score"))
        y_points.append(transform(val_lrps, "lrp"))
        colors.append("r")
        curve_labels.append("val")
      draw_slrp_plot(x_points, y_points, colors, curve_labels, axis_labels, title, fig_name)

def obtain_slrp_stats(iterations, count_settings):
  for iter in iterations:
    for dataset_type in ["train", "val"]:
      slrp_result_path = file_name = path.join(RESULTS_PATH_PREFIX, str(iter), f"{dataset_type}", f"{dataset_type}_lrp_opt_thr")
      with open(slrp_result_path, 'rb') as in_file:
        lrp_opt_thrs = pickle.load(in_file)

      num_nans = { "rare":0, "common":0, "frequent":0}
      num_negs = { "rare":0, "common":0, "frequent":0}

      print_count_per_cat_info = True
      for (count_nans,count_negs) in count_settings:
        lrp_opt_sums = { "rare":[0,0], "common":[0,0], "frequent":[0,0]}
        for cat_id,val in enumerate(lrp_opt_thrs):
          cls_freq_group = index_to_freq_group(cat_id)
          if np.isnan(val):
            num_nans[cls_freq_group]+=1
            if count_nans:
              val = 0
            else:
              continue
          elif val<0:
            num_negs[cls_freq_group]+=1
            if count_negs:
              val = 0
            else:
              continue

          lrp_opt_sums[cls_freq_group][0]+= val
          lrp_opt_sums[cls_freq_group][1]+= 1

        count_nans_name = "counting nans" if count_nans else None
        count_negs_name = "counting negs" if count_negs else None
        if count_nans_name and count_negs_name:
          info = f"{count_nans_name} and {count_negs_name}"
        elif count_nans_name and not count_negs_name:
          info = f"{count_nans_name}"
        elif not count_nans_name and count_negs_name:
          info = f"{count_negs_name}"
        else:
          info = "Ignoring both"

        dir_path = path.join(SLRP_STATS_PATH_PREFIX, str(iter))
        stats_out_path = path.join(dir_path, f"{dataset_type}_opt_thr_stats.txt")
        if not path.isdir(dir_path):
          mkdir(dir_path)
        with open(stats_out_path, 'a') as out_file:
          if print_count_per_cat_info:
            print_count_per_cat_info = False
            out_file.write("Per category occurrences of NaN and negative values\n\n")
            out_file.write(f"# of NaN values: {num_nans}\n")
            out_file.write(f"# of -1 values: {num_negs}\n\n")
            out_file.write("Per category averages of sLRP optimal thresholds based on counting NaN and negative values:\n\n")
          out_file.write(f"{info}:\n")
          out_file.write(f"{lrp_opt_sums}\n\n")

          for key in lrp_opt_sums:
            avg = lrp_opt_sums[key][0]/lrp_opt_sums[key][1]
            out_file.write(f"{key}_avg => {avg:.4f}\n")
          out_file.write("\n")

def draw_slrp_opt_plot(x_points, y_points, axis_labels, title, fig_name):
  plt.scatter(x_points, y_points)
  plt.legend()

  plt.xlim(-1.010, 1.260)
  plt.ylim(-1.010, 1.260)

  plt.xlabel(axis_labels[0], fontsize=10)
  plt.ylabel(axis_labels[1], fontsize=10)

  plt.title(title)
  plt.savefig(fig_name)
  plt.cla()

def obtain_slrp_scatter_plots(iterations):
  for iter in iterations:
    train_lrp_opt_thr_path = path.join(RESULTS_PATH_PREFIX, str(iter), "train", "train_lrp_opt_thr")
    val_lrp_opt_thr_path = path.join(RESULTS_PATH_PREFIX, str(iter), "val", "val_lrp_opt_thr")

    dir_path = path.join(SLRP_STATS_PATH_PREFIX, str(iter))
    if not path.isdir(dir_path):
      mkdir(dir_path)

    with open(train_lrp_opt_thr_path, 'rb') as train_lrp_opt_thr_file, open(val_lrp_opt_thr_path, 'rb') as val_lrp_opt_thr_file:
      train_lrp_opt_thrs = pickle.load(train_lrp_opt_thr_file)
      val_lrp_opt_thrs = pickle.load(val_lrp_opt_thr_file)

      train_points = {"rare":[], "common":[], "frequent":[]}
      val_points = {"rare":[], "common":[], "frequent":[]}
      axis_labels = ["train", "val"]

      count=0
      for cat_id,val in enumerate(train_lrp_opt_thrs):
        if (np.isnan(val)):
          # print("nan")
          count+=1
        train_val = 1.25 if np.isnan(val) else val
        vald_val = 1.25 if np.isnan(val_lrp_opt_thrs[cat_id]) else val_lrp_opt_thrs[cat_id]
        cls_freq_group = index_to_freq_group(cat_id)
        train_points[cls_freq_group].append(train_val)
        val_points[cls_freq_group].append(vald_val)
      for cls_freq_group in train_points.keys():
        fig_name = path.join(dir_path, f"{cls_freq_group}_slrp_opt.png")
        title = f"{cls_freq_group} s* values"
        draw_slrp_opt_plot(train_points[cls_freq_group], val_points[cls_freq_group], axis_labels, title, fig_name)
      print(count)
      count2=0
      for point in train_points["rare"]:
        if point==1.25:
          count2+=1
      print(count2)

def plot_basic(x, y_axises, colors, curve_labels, axis_labels, title, fig_name):
  plt.gca().set_prop_cycle(color=colors)
  for i in range(len(y_axises)):
    plt.scatter(x,y_axises[i],label=curve_labels[i])

  plt.ylim(0, 1.1)

  plt.title(title,fontsize=15)
  plt.xlabel(axis_labels[0],fontsize=13)
  plt.ylabel(axis_labels[1],fontsize=13)
  plt.legend()
  plt.savefig(fig_name)
  plt.cla()

def obtain_score_distribution(iterations, data_keys, dataset_types):
  results_data = read_data(iterations, data_keys, dataset_types)
  val_avg_scores = []
  for iter in iterations:
    values = results_data[iter]

    val_dt_scores_map = values["val_dt_scores"]

    cat_avg_scores_map = list(range(1203))
    count_map = {"rare":0, "common":0, "frequent":0}
    for key in val_dt_scores_map:
      (cat_id,area_id) = key
      # consider "all" area type
      if area_id != 0:
        continue
      scores = val_dt_scores_map[key]

      if scores is not None:
        sum = np.sum(scores)
        for score in scores:
          if score >1:
            count_map[index_to_freq_group(cat_id)]+=1
        avg = sum/len(scores)
      else:
        avg = 0
      cat_avg_scores_map[cat_id] = avg
    print(count_map)
    val_avg_scores.append(cat_avg_scores_map)


  x_points = list(range(1203))
  colors=['red', 'blue']
  plot_basic(x_points, val_avg_scores, colors, ["iter #0", "iter #1"],
            ["cat_id", "avg_score"], "Val set score distribution wrt iterations",
            "val_score_dist.png")

if __name__ == "main":
  iterations = [0]
  # create slrp plots for given iterations. each plot shows train and val slrp curves
  # create_plots(iterations)

  # obtain average s* statistics of given iterations
  # count_settings = (True,True), (True,False), (False,True), (False,False)
  # obtain_slrp_stats(iterations, count_settings)

  # obtain_slrp_scatter_plots(iterations)

  data_keys = ["dt_scores", "tps", "fps"]
  dataset_types = ["val"]
  obtain_score_distribution([0,1], data_keys, dataset_types)