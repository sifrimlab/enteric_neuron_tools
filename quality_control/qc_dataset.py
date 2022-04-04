import os
import random
import numpy as np
from glob import glob
from label_images import labelTrainingSet
from pyxelperfect.diagnose import compareImageStats
import pandas as pd
from sklearn import tree
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, RepeatedStratifiedKFold, GridSearchCV
from sklearn import metrics
from shutil import copyfile
import pickle
from icecream import ic

"""
    This script takes a glob pattern that targets all images, and makes a train/test split, for which the labels you'll be prompted to make yourself
    A classification tree is made based on your labels, and applied to the test set
"""

filtered_out_dir = "./"
filtered_out_bad_dir = os.path.join(filtered_out_dir, "filtered_bad/")
filtered_out_good_dir = os.path.join(filtered_out_dir, "filtered_good/")
os.makedirs(filtered_out_dir, exist_ok=True)
os.makedirs(filtered_out_bad_dir, exist_ok=True)
os.makedirs(filtered_out_good_dir, exist_ok=True)

def newLabelingScheme(glob_pattern: str):
    list_imgs = glob(glob_pattern)
    X_train, x_test = train_test_split(list_imgs, test_size=0.3) # 70% training and 30% test
    training_dir = labelTrainingSet(X_train)
    return X_train, x_test, training_dir

def loadLabelingScheme(path_to_base_dir: str):
    base_list = glob(os.path.join(path_to_base_dir, "*.tif"))
    bad_list =glob( os.path.join(path_to_base_dir, "bad/*"))
    good_list =glob( os.path.join(path_to_base_dir, "good/*"))
    X_train = bad_list + good_list
    basename_X_train = [os.path.basename(el) for el in X_train]
    x_test = [el for el in base_list if os.path.basename(el) not in basename_X_train]
    return X_train, x_test, path_to_base_dir

# X_train, x_test, training_dir = newLabelingScheme("/home/david/.config/nnn/mounts/nacho@10.38.76.144/tool/enteric_neurones/slidescanner_examples/Good/Slide2-2-2_Region0000_Channel647,555,488_Seq0017/Slide2-2-2_Region0000_Channel647,555,488_Seq0017_555_tiles/*tile*.tif")

X_train, x_test, training_dir = loadLabelingScheme("/home/david/.config/nnn/mounts/nacho@10.38.76.144/tool/enteric_neurones/slidescanner_examples/Good/Slide2-2-2_Region0000_Channel647,555,488_Seq0017/Slide2-2-2_Region0000_Channel647,555,488_Seq0017_555_tiles/")

# # get stats of labeled images and combine them into a df for training
good_df = compareImageStats(os.path.join(training_dir, "good/", "*"), result_prefix = "training_good", add_props = {"mode" : "good"})
bad_df = compareImageStats(os.path.join(training_dir, "bad/", "*"), result_prefix = "training_bad", add_props = {"mode" : "bad"})

train_dataset = pd.concat([good_df, bad_df], ignore_index = True)
train_dataset = train_dataset.drop(["Shape", "Dtype range", "Actual range", "Dtype"], axis=1)

train_labels = train_dataset["mode"]
train_dataset = train_dataset.drop(["mode"], axis=1)

sc = StandardScaler()
train_dataset = sc.fit_transform(train_dataset)

# weights = {"good":1.0, "bad":1.0}
balance = [{"good":100, "bad":1}, {"good":10, "bad":1}, {"good":1, "bad":1}, {"good":1, "bad":10}, {"good":1, "bad":100}]
model = tree.DecisionTreeClassifier(class_weight='balanced', max_depth=3)
param_grid = dict(class_weight=balance)
# define evaluation procedure
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
# define grid search
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=cv, scoring='roc_auc')
# evaluate model
scores = cross_val_score(model, train_dataset, train_labels, scoring='roc_auc', cv=cv, n_jobs=-1)
# summarize performance
print('Mean ROC AUC: %.3f' % np.mean(scores))
grid_result = grid.fit(train_dataset, train_labels)
# report the best configuration
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
# report all configurations
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))
# clf = clf.fit(train_dataset, train_labels)
clf = grid_result
train_preds = clf.predict(train_dataset)
print("Train Accuracy:",metrics.accuracy_score(train_labels, train_preds))

with open("tree_model.P", "wb") as f:
    pickle.dump(clf, f)

# Now use the model to classify all "test" tiles 
# prep dataset first
test_df = compareImageStats(x_test, result_prefix = "test_results")
test_dataset = test_df.drop(["Shape", "Dtype range", "Actual range", "Dtype"], axis=1)
test_preds = clf.predict(test_dataset)
ic(test_preds, len(test_preds))

for pred, image_path in zip(test_preds, x_test):
    ic(pred)
    if pred == "good":
        copyfile(image_path, os.path.join(filtered_out_good_dir, os.path.basename(image_path)))
    elif pred == "bad":
        copyfile(image_path, os.path.join(filtered_out_bad_dir, os.path.basename(image_path)))



    
