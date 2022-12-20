import numpy as np 
from tqdm import tqdm
from sklearn.cluster import KMeans
from sklearn.datasets import load_digits

'''
** KMeans Algorithm ** 

We will implement the regular kmeans algorithm as described here: 
https://arxiv.org/pdf/1312.4176.pdf

Start with k random centroids.
Assignment Step: 
    Each obs is assigned to nearest centroid based on squared distance
Refinement Step:
    Centroids are updated as the centroid of each set of obs. 
Iterate until convergence or M iters. 

'''

class RegularKMeans: 

    def __init__(self, n_clusters=2, init='kpp', max_iter=300, tol=1e-4): 
        self.k = n_clusters
        self.init = init 
        self.tol= 1e-4
        self.max_iter = max_iter

        self.cluster_centers_ = None
        self.labels_ = None
        self.inertia_ = None

    def predict(self, X): 
        labels, _ = self._assignment(X, self.cluster_centers_)
        return labels

    
    def fit(self, X):
        centroids = self._init_centroids(X)
        for _ in range(self.max_iter):
            new_labels, self.inertia_ = self._assignment(X, centroids)
            new_centroids = self._refinement(X, new_labels)
            # print('Centroids:', new_centroids, centroids)
            # centroid_diff = np.linalg.norm(centroids - new_centroids)
            centroid_diff = np.abs(centroids - new_centroids)
            if np.sum(centroid_diff > self.tol) == 0: 
                centroids = new_centroids
                break
            centroids = new_centroids
        self.cluster_centers_ = centroids
        self.labels_ = new_labels
        return self

    def _init_centroids(self, X): 
        '''
        Init Methods for Centroids: 
        - Forgy: 
            Randomly choose k observations as points.
            These are the initial means. 
            Spread initial centroids. Good for EM. 
        - Random Partition: 
            Randomly assign each point to a cluster. 
            Calculate mean of each, this is the initial mean. 
            Centroids are tighter together. Good for fuzzy. 
        - KMeans++ (kpp): 
            Choose a center uniformly at random from data points. 
            For each point not chosen, 
                compute distance between it an nearest center. (D(x))
            Choose another point as new center, with weighted probability
                distribution proportional to D(x)^2. 
            Repeat until k centers have been chosen. 
        '''
        centroids = None 
        if self.init == 'forgy': 
            centroids = X[np.random.choice(X.shape[0], self.k, replace=False)]
        elif self.init == 'random': 
            cluster_assigns = np.random.randint(0, self.k, size=(X.shape[0],))
            centroids = []
            for cluster in range(self.k): 
                cluster_mean = np.mean(X[cluster_assigns==cluster], axis=0)
                centroids.append(cluster_mean)
            centroids = np.array(centroids)
        elif self.init == 'kpp': 
            centroids = []
            centroid_idxs = set()            
            probs = np.ones(X.shape[0])/X.shape[0]
            for cluster in range(self.k): 
                centroid_idx = np.random.choice(X.shape[0], p=probs)
                while centroid_idx in centroid_idxs: 
                    centroid_idx = np.random.choice(X.shape[0], p=probs)
                centroid_idxs.add(centroid_idx)
                centroids.append(X[centroid_idx])
                distance_fn = lambda p: self._sq_euc_distance(X[centroid_idx], p)
                centroid_mask = np.ones(X.shape[0])
                centroid_mask[list(centroid_idxs)] = 0 
                distances = np.apply_along_axis(distance_fn, axis=1, arr=X[centroid_mask.astype(bool)])
                probs = distances/np.sum(distances)
                for centroid_idx in centroid_idxs: 
                    if centroid_idx > len(probs): 
                        probs = np.append(probs, 0)
                    else:  
                        probs = np.insert(probs, centroid_idx, 0)
            centroids = np.array(centroids) 
        else: 
            raise Exception(
                f'{self.init_method} is invalid! Please choose forgy, random, or kpp.'
            )
        return centroids            

    def _sq_euc_distance(self, p0, p1): 
        return np.sqrt(np.sum(np.square(p0 - p1)))
        

    def _assignment(self, X, centroids): 
        all_distances = []
        for i in range(centroids.shape[0]):
            centroid = centroids[i]
            distance_fn = lambda p: self._sq_euc_distance(centroid, p)
            distances = np.apply_along_axis(distance_fn, axis=1, arr=X)
            all_distances.append(distances)
        all_distances = np.array(all_distances)
        assignments = np.argmin(all_distances, axis=0)

        min_vals = np.amin(all_distances, axis=0)**2
        inertia = 0 
        for assignment in np.unique(assignments):
            inertia += np.sum(min_vals[assignments == assignment])
        
        # Calculate inertia as well 
        return assignments, inertia
            


    def _refinement(self, X, labels): 
        centroids = []
        for label in np.sort(np.unique(labels)):
            label_mask = (labels == label).astype(bool)
            centroid = np.mean(X[label_mask], axis=0)
            centroids.append(centroid)
        return np.array(centroids)


def baseline(X): 
    print("SKLearn KMeans Implementation")
    kmeans = KMeans(n_clusters=2, random_state=0).fit(X)
    print("Labels:", kmeans.labels_) 
    print("Intertia:", kmeans.inertia_) 
    print('Cluster Centers:',kmeans.cluster_centers_)
    print('Preds:', kmeans.predict(np.array([[0, 0], [12, 3]])))


if __name__ == '__main__': 

    X = np.array([[1, 2], [1, 4], [1, 0], [10, 2], [10, 4], [10, 0]])
    baseline(X)

    # kmeans = RegularKMeans(init='forgy').fit(X)
    # kmeans = RegularKMeans(init='random').fit(X)
    print("Kpp Custom KMeans Implementation")
    kmeans = RegularKMeans(init='kpp')
    kmeans.fit(X)
    print("Labels:", kmeans.labels_) 
    print('Cluster Centers:',kmeans.cluster_centers_)
    print('Preds:', kmeans.predict(np.array([[0, 0], [12, 3]])))
    print("Intertia:", kmeans.inertia_) 
