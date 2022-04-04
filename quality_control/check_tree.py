from sklearn import tree
import pickle


with open("./tree_model.P", "rb")as f:
    test_tree =pickle.load(f)
    print(tree.export_text(test_tree))

