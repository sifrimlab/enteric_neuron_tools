import pandas as pd
from sklearn import tree
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
import seaborn as sns
import pickle
import matplotlib.pyplot as plt


# Load data
train_dataset = pd.read_csv("./data/self_labeled_train_combined.csv",index_col=0,delimiter=',')
train_dataset = train_dataset.drop(["Shape", "Dtype range", "Actual range", "Dtype"], axis=1)

test_dataset = pd.read_csv("./data/self_labeled_test_combined.csv",index_col=0,delimiter=',')
test_dataset = test_dataset.drop(["Shape", "Dtype range", "Actual range", "Dtype"], axis=1)

clusters = train_dataset["mode"]
train_dataset = train_dataset.drop(["mode"], axis=1)
## X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=101)
y_test =list( test_dataset["mode"])
test_dataset = test_dataset.drop(["mode"], axis=1)

##-------------------------------------------------------------------------------------------------------
sc = StandardScaler()
X = train_dataset
X = sc.fit_transform(X)

y = clusters

# X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=101)

clf = tree.DecisionTreeClassifier()
clf = clf.fit(X, y)

y_pred = clf.predict(test_dataset)
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

with open("tree_model.tif") as f:
    save = pickle.dump(clf, f)

# cm = metrics.confusion_matrix(y_test, preds, labels=["good", "bad"])
# ax= plt.subplot()
# sns.heatmap(cm, annot=True, fmt='g', ax=ax);  #annot=True to annotate cells, ftm='g' to disable scientific notation

# # labels, title and ticks
# ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels'); 
# ax.set_title('Confusion Matrix'); 
# ax.xaxis.set_ticklabels(['Good', 'Bad']); ax.yaxis.set_ticklabels(['Good', 'Bad']);

# plt.show()


# # r = tree.export_text(clf, feature_names = list(dataset.columns))
# # print(r)

# # tree.plot_tree(clf, feature_names=train_dataset.columns, fontsize=10)
# # plt.show()


