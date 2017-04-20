import numpy as np, pandas as pd
from collections import defaultdict

def get_user_venue_pairs(_tr, _vv, _vp):
    train_pairs, valid_pairs = defaultdict(set), defaultdict(set)
    mapping = list(pd.read_csv(_tr).groupby(["user_id", "venue_id"]).count().index)
    for user, venue in mapping:
        if user in train_pairs:
            train_pairs[user].add(venue)
        else:
            train_pairs[user] = set([venue])
    # during training, we merge vv and vp to get all the positive examples in our validation set
    mapping_vv = list(pd.read_csv(_vv).groupby(["user_id", "venue_id"]).count().index)
    mapping_vp = list(pd.read_csv(_vp).groupby(["user_id", "venue_id"]).count().index)
    for user, venue in mapping_vv:
        if user in valid_pairs:
            valid_pairs[user].add(venue)
        else:
            valid_pairs[user] = set([venue])
    for user, venue in mapping_vp:
        if user in valid_pairs:
            valid_pairs[user].add(venue)
        else:
            valid_pairs[user] = set([venue])
    return train_pairs, valid_pairs

def get_user_venue_pairs_eval(_tv, _tp):
    testv_pairs, testp_pairs = defaultdict(set), defaultdict(set)
    # during testing, we separate tv and tp to get mAP score
    mapping_tv = list(pd.read_csv(_tv).groupby(["user_id", "venue_id"]).count().index)
    mapping_tp = list(pd.read_csv(_tp).groupby(["user_id", "venue_id"]).count().index)
    for user, venue in mapping_tv:
        if user in testv_pairs:
            testv_pairs[user].add(venue)
        else:
            testv_pairs[user] = set([venue])
    for user, venue in mapping_tp:
        if user in testp_pairs:
            testp_pairs[user].add(venue)
        else:
            testp_pairs[user] = set([venue])
    return testv_pairs, testp_pairs

def get_unique(datapath, users, venues):
    u, v = set(), set()
    if users:
        u = set(list(pd.read_csv(datapath, usecols=["user_id"])["user_id"]))
    if venues:
        v = set(list(pd.read_csv(datapath, usecols=["venue_id"])["venue_id"]))
    return u, v