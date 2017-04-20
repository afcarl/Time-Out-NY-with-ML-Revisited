import fs_util
from sklearn.neighbors import NearestNeighbors
from sklearn.externals import joblib

def main():
    # checkins = fs_util.clean_checkins(min_venues_per_user=2)
    # fs_util.train_test_split(checkins, thresh=0.8, holdout=0.5)
    # initialize a NN search object to recommend to unseen venues
    X = fs_util.generate_features_tr(n_category=400)
    neigh = NearestNeighbors(n_neighbors=20, algorithm="brute", metric="cosine", n_jobs=-1)
    neigh.fit(X)
    joblib.dump(neigh, "neigh.pkl")

if __name__=="__main__":
    main()