from math import sqrt 
from collections import defaultdict
from statistics import mode
from sklearn import datasets
from tqdm import tqdm
import numpy as np 

from sklearn import ensemble
from sklearn.datasets import make_classification





# Custom Implementation of a Random Forest Classifier using Python 
# I'll compare this to the sklearn random forest 

class Node: 

    def __init__(self, feat_idx=None, gini=None, cutoff=None, val=None, left=None, right=None, left_rows=None, right_rows=None):
        self.feat_idx = feat_idx
        self.cutoff = cutoff
        self.val = val
        self.left = left
        self.right = right
        self.gini = gini
        self.left_rows = left_rows
        self.right_rows = right_rows


class RandomForestClassifier: 
    '''
    '''

    def __init__(self, num_trees=10, min_samples_split=5):
        self.num_trees = num_trees
        self.min_samples_split = min_samples_split
        self.trees = []

    def _get_feature_importances(self, node, gini_dict):
        gini_dict[node.feat_idx] += node.gini*(node.left_rows+node.right_rows)
        if node.left is not None and node.left.gini is not None: 
            gini_dict[node.feat_idx] -= node.left.gini*(node.left_rows+node.right_rows)
            self._get_feature_importances(node.left, gini_dict)
        if node.right is not None and node.right.gini is not None: 
            gini_dict[node.feat_idx] -= node.right.gini*(node.left_rows+node.right_rows)
            self._get_feature_importances(node.right, gini_dict)

    def get_feature_importances(self): 
        gini_dict = defaultdict(list)
        for root in self.trees: 
            tree_gini_dict = defaultdict(float)
            if root.gini is not None: 
                self._get_feature_importances(root, tree_gini_dict)
                for feat in tree_gini_dict: 
                    gini_dict[feat].append(tree_gini_dict[feat])
            else:
                print("Root is none??")
        
        feature_importances = {}
        for feat in gini_dict: 
            feature_importances[feat] = sum(gini_dict[feat])/self.num_trees
            
        total_gini = sum([feature_importances[i] for i in feature_importances])
        sorted_feats = sorted(list(feature_importances.keys()))
        feature_importances = np.array([feature_importances[feat_idx]/total_gini for feat_idx in sorted_feats])
        return feature_importances 

    def generate_bootstraps(self, X, y): 
        bootstraps = []
        oob_dict = defaultdict(list)
        for b in range(self.num_trees): 
            sample_idxs = np.random.choice(list(range(len(X))), len(X))
            bag_X = X[sample_idxs]
            bag_y = y[sample_idxs]
            oob_mask = np.ones(len(X), dtype=bool)
            oob_mask[sample_idxs] = False
            for oob_idx in np.where(oob_mask==True)[0]:
                oob_dict[oob_idx].append(b)
            bootstraps.append((bag_X, bag_y))
        return bootstraps, oob_dict

    def predict_single(self, X, trees=None):
        if trees is None: 
            trees = self.trees
        results = []
        for root in trees: 
            node = root
            while(True): 
                if node.val is not None: 
                    break
                if X[node.feat_idx] < node.cutoff:
                    node = node.left
                else:
                    node = node.right
            results.append(node.val)
        return mode(results)
            
            
        

    def predict(self, X, trees=None):
        if len(X.shape) != 2: 
            print(f"Wrong shape ({X.shape}) for data!")
            return None
    
        if trees is None: 
            trees = self.trees
        for tree in trees: 
            pass


    def fit(self, X, y):

        bootstrap_sets, oob_dict  = self.generate_bootstraps(X, y)
        print(f"Fitting {self.num_trees} trees on bootstrap set...")
        for boot_X, boot_y in tqdm(bootstrap_sets): 
            tree = self.grow_single_tree(boot_X, boot_y)
            self.trees.append(tree)

        print(f"Trained! Calculating OOB Error...")
        num_correct = 0 
        total_oob = len(X)
        for i, training_example in enumerate(X):
            oob_trees = [self.trees[t] for t in oob_dict[i]]
            if len(oob_trees) == 0: 
                total_oob -= 1 
                continue
            result = self.predict_single(training_example, trees=oob_trees)
            if result == y[i]: 
                num_correct += 1 

        oob_acc = (num_correct/total_oob)
        print(f"OOB Acc: {oob_acc:.2%}")
             
            
        


    def grow_tree_recurse(self, X, y):
        # Creating a terminal node
        if len(X) <= self.min_samples_split: 
            return Node(val=mode(y))

        # Otherwise, grow some internal nodes
        node = Node()
        num_features = int(sqrt(X.shape[1]))
        feat_idxs = np.random.choice(list(range(X.shape[1])), num_features, replace=False)
        min_gini = np.inf
        best_left_X, best_left_y, best_right_X, best_right_y = None, None, None, None
        for feat_idx in feat_idxs: 
            for val in X[:,feat_idx]:
                left_mask = X[:,feat_idx] < val
                left_X = X[left_mask]
                left_y = y[left_mask]
                right_X = X[~left_mask]
                right_y = y[~left_mask]
                groups = (left_y, right_y)
                gini = self.gini_score(groups)
                if gini < min_gini: 
                    node.cutoff = val
                    node.feat_idx = feat_idx
                    node.gini = gini
                    node.left_rows = len(left_X)
                    node.right_rows = len(right_X)
                    best_left_X = left_X
                    best_left_y = left_y
                    best_right_X = right_X
                    best_right_y = right_y
        if len(best_left_y) == 0: 
            return Node(val=mode(best_right_y))
        if len(best_right_y) == 0: 
            return Node(val=mode(best_left_y))
        node.left = self.grow_tree_recurse(best_left_X, best_left_y)
        node.right = self.grow_tree_recurse(best_right_X, best_right_y)
        return node

    def grow_single_tree(self, X, y):
        root = self.grow_tree_recurse(X, y)
        return root

          

    def gini_score(self, groups): 
        '''
            groups: list tuples of (1xm, label)
            Get the gini score of the current groups, where 
            gini score is defined as the sum over all classes of 
            p_hat_mk (1 - p_hat_mk), 
            where p_hat_mk is the proportion of observations
            in region m that are from class k. 

            It is the total variance for K classes.
        ''' 
        gini = 0 
        classes = np.unique(np.concatenate(groups))
        total_obs = float(sum([len(group) for group in groups]))
        for group in groups:
            group_size = len(group) 
            if len(group) == 0: 
                continue
            for k in classes: 
                p_km = len(group[group==k])/group_size
                gini += (p_km * (1-p_km))
        return gini



if __name__ == '__main__': 

    # iris = datasets.load_iris()
    # X = iris.data
    # y = iris.target
    X, y = make_classification(n_samples=1000, n_features=4, n_informative=2, n_redundant=0, random_state=0, shuffle=False)

    my_rf = RandomForestClassifier(num_trees=100, min_samples_split=2)
    my_rf.fit(X, y)
    print(my_rf.get_feature_importances())

    rf = ensemble.RandomForestClassifier(oob_score=True)
    rf.fit(X, y)
    print(f"SKLearn RF OOB Acc: {rf.oob_score_:.2%}")
    print(rf.feature_importances_)
