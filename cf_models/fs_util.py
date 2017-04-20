import pandas as pd, numpy as np
from collections import Counter, OrderedDict
from datetime import datetime
from sklearn.feature_extraction import FeatureHasher
import random, json

categories_to_drop = ['Airport', 
                      'Bus Stop',
                      'Bus Station', 
                      'Train Station', 
                      "Doctor's Office", 
                      'Border Crossing', 
                      'Gas Station', 
                      'Bank',
                      'Church',
                      'Residential Building (Apartment / Condo)',
                      'Office',
                      'Tech Startup',
                      'Hostel',
                      'Hotel']

def convert_time(t):
    try:
        dt = datetime.strptime(t, "%Y-%m-%d %H:%M:%S.0")
    except ValueError:
        dt = datetime.strptime(t, "%Y-%m-%d %H:%M:%S")
    return dt

def parse_time_to_weekend(dt):
    if dt.strftime('%A') in ['Saturday', 'Sunday']:
        return True
    return False

def parse_time_to_hour(dt):
    return dt.hour

def clean_checkins(min_venues_per_user):
    checkins = pd.read_csv("../data/checkins.csv", converters={"time": convert_time})
    print "Before cleaning, there are %d checkins in total" % (checkins.shape[0])
    checkins = checkins.loc[~checkins["venue_category"].isin(categories_to_drop)]
    print "After dropping unfit categories, there are %d checkins in total" % (checkins.shape[0])
    checkins_by_u_v = checkins.groupby(["user_id", "venue_id"], as_index=False).count()[["user_id", "venue_id", "venue_idx"]]
    # number of venues checked in by each user
    checkin_counts = Counter(list(checkins_by_u_v["user_id"]))
    ids_to_keep = [i[0] for i in checkin_counts.most_common() if i[1]>=min_venues_per_user]
    checkins = checkins.loc[checkins["user_id"].isin(ids_to_keep)]
    print "After dropping infrequent users, there are %d checkins in total" % (checkins.shape[0])
    checkins.drop(["label", "venue_idx"], axis=1, inplace=True)
    checkins.rename(columns={"message": "text"}, inplace=True)
    return checkins

def train_test_split(checkins, thresh, holdout):
    # since we are predicting venues for the "future"
    checkins.sort_values(by="time", inplace=True)
    user_ids = list(checkins["user_id"].unique())
    random.shuffle(user_ids)
    t_v_ids, test_ids = user_ids[:int(thresh*len(user_ids))], user_ids[int(thresh*len(user_ids)):]
    random.shuffle(t_v_ids)
    train_ids, valid_ids = t_v_ids[:int(thresh*len(t_v_ids))], t_v_ids[int(thresh*len(t_v_ids)):]
    print "Number of users in training set: %d" % (len(train_ids))
    print "Number of users in validation set: %d" % (len(valid_ids))
    print "Number of users in testing set: %d" % (len(test_ids))
    train = checkins[checkins["user_id"].isin(train_ids)]
    valid = checkins[checkins["user_id"].isin(valid_ids)]
    test = checkins[checkins["user_id"].isin(test_ids)]
    valid_visible, valid_predict, test_visible, test_predict = pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    for user in valid_ids:
        subset = valid[valid["user_id"]==user]
        n_holdout = int(holdout*subset.shape[0])
        visible = pd.DataFrame(subset.values[:-n_holdout], columns=checkins.columns)
        predict = pd.DataFrame(subset.values[-n_holdout:], columns=checkins.columns)
        valid_visible = valid_visible.append(visible, ignore_index=True)
        valid_predict = valid_predict.append(predict, ignore_index=True)
    for user in test_ids:
        subset = test[test["user_id"]==user]
        n_holdout = int(holdout*subset.shape[0])
        visible = pd.DataFrame(subset.values[:-n_holdout], columns=checkins.columns)
        predict = pd.DataFrame(subset.values[-n_holdout:], columns=checkins.columns)
        test_visible = test_visible.append(visible, ignore_index=True)
        test_predict = test_predict.append(predict, ignore_index=True)
    print train.shape[0]
    print valid_visible.shape[0], valid_predict.shape[0]
    print test_visible.shape[0], test_predict.shape[0]
    train.to_csv("../data/train_data.csv", index=False)
    valid_visible.to_csv("../data/valid_visible.csv", index=False)
    valid_predict.to_csv("../data/valid_predict.csv", index=False)
    test_visible.to_csv("../data/test_visible.csv", index=False)
    test_predict.to_csv("../data/test_predict.csv", index=False)

def sort_venues(datapath, binary):
    if not binary:
        venues = list(pd.read_csv(datapath, usecols=["venue_id"])["venue_id"])
    else:
        venues = list(pd.read_csv(datapath).groupby(["venue_id", "user_id"], as_index=False).count()["venue_id"])
    return [i[0] for i in Counter(venues).most_common()]

def sort_venue_categories(datapath, binary):
    if not binary:
        categories = list(pd.read_csv(datapath, usecols=["venue_category"])["venue_category"])
    else:
        categories = list(pd.read_csv(datapath).groupby(["venue_category", "user_id"], as_index=False).count()["venue_category"])
    return [i[0] for i in Counter(categories).most_common()]

def user_to_venues(datapath):
    u2v = dict()
    mapping = list(pd.read_csv(datapath).groupby(["user_id", "venue_id"]).count().index)
    for user,venue in mapping:
        if user in u2v:
            u2v[user].add(venue)
        else:
            u2v[user] = set([venue])
    return u2v

def venue_to_users(datapath):
    v2u = dict()
    mapping = list(pd.read_csv(datapath).groupby(["venue_id", "user_id"]).count().index)
    for venue,user in mapping:
        if venue in v2u:
            v2u[venue].add(user)
        else:
            v2u[venue] = set([user])
    return v2u

def unique_venues(datapath):
    v = set(list(pd.read_csv(datapath, usecols=["venue_id"])["venue_id"]))
    return v

def sort_dict_dec(d):
    return sorted(d.keys(), key=lambda s:d[s], reverse=True)

def keep_useful_features(df):
    df['weekend'] = df['time'].apply(parse_time_to_weekend)
    df['hour'] = df['time'].apply(parse_time_to_hour)
    df.drop(['county', 'text', 'time', 'city', 'venue_name'], axis=1, inplace=True)

'''
FEATURE ENGINEERING FOR UNSEEN VENUES ON TRAINING SET
'''

def generate_venue_info_dict():
    train = pd.read_csv("../data/train_data.csv", converters={"time": convert_time})
    keep_useful_features(train)

def generate_categorical_features_tr(n_features, train):
    venue_dict, all_category_pairs = dict(), dict()
    for user in train["user_id"].unique():
        subset = train[train["user_id"]==user]
        venues_visited = list(subset.groupby(["venue_id", "venue_category"]).count()["hour"].index)
        for (v1, c1) in venues_visited:
            category_dict = dict()
            for (v2, c2) in venues_visited:
                if v1 != v2 and c1 != c2:
                    if "%s_%s" % (c1, c2) in category_dict:
                        category_dict["%s_%s" % (c1, c2)] += 1
                    else:
                        category_dict["%s_%s" % (c1, c2)] = 1
                    if "%s_%s" % (c1, c2) in all_category_pairs:
                        all_category_pairs["%s_%s" % (c1, c2)] += 1
                    else:
                        all_category_pairs["%s_%s" % (c1, c2)] = 1
            if v1 in venue_dict:
                for (pair, val) in category_dict.iteritems():
                    if pair in venue_dict[v1]:
                        venue_dict[v1][pair] += val
                    else:
                        venue_dict[v1][pair] = val
            else:
                venue_dict[v1] = category_dict

    svenue_dict = OrderedDict(sorted(venue_dict.items()))
    del venue_dict

    hasher = FeatureHasher(n_features)
    category_features = hasher.transform([i[1] for i in svenue_dict.items()])
    return category_features.toarray(), all_category_pairs

def generate_temporal_features_tr(train):
    all_venues_hours, all_venues_weekends = list(), list()
    for user in train["user_id"].unique():
        subset = train[train["user_id"]==user]
        venues_hours = list(subset.groupby(["venue_id", "hour"]).count()["venue_category"].index)
        venues_weekends = list(subset.groupby(["venue_id", "weekend"]).count()["venue_category"].index)
        all_venues_hours.extend(venues_hours)
        all_venues_weekends.extend(venues_weekends)

    vh_dict, vw_dict = dict(), dict()
    for (v, h) in all_venues_hours:
        if v in vh_dict:
            if h in vh_dict[v]:
                vh_dict[v][str(h)] += 1
            else:
                vh_dict[v][str(h)] = 1
        else:
            vh_dict[v] = {str(h): 1}
    del all_venues_hours
    hasher = FeatureHasher(n_features=24)
    hour_features = hasher.transform([i[1] for i in OrderedDict(sorted(vh_dict.items())).items()])

    for (v, w) in all_venues_weekends:
        if v in vw_dict:
            vw_dict[v]['total'] += 1
        else:
            vw_dict[v] = {'total': 1, 'weekends': 0}
        if w:
            vw_dict[v]['weekends'] += 1
    del all_venues_weekends
    for v in vw_dict:
        percentage = float(vw_dict[v]['weekends'])/vw_dict[v]['total']
        del vw_dict[v]['weekends'], vw_dict[v]['total']
        vw_dict[v] = percentage
    weekend_feature = np.array([val for key, val in sorted(vw_dict.items())]).reshape(-1,1)

    return np.hstack((hour_features.toarray(), weekend_feature))

def generate_spatial_features_tr(train):
    train.drop_duplicates("venue_id", inplace=True)
    train.set_index(keys="venue_id", drop=True, inplace=True)
    train.sort_index(inplace=True)
    spatial_features = np.array(train[["lat", "lon"]].values)
    return spatial_features, {c:i for c,i in enumerate(train.index)}

def generate_features_tr(n_category):
    train = pd.read_csv("../data/train_data.csv", converters={"time": convert_time})
    keep_useful_features(train)
    categorical_features, category_pairs = generate_categorical_features_tr(n_category, train)
    temporal_features = generate_temporal_features_tr(train)
    spatial_features, index_venue = generate_spatial_features_tr(train)
    with open("../data/index_venue.json", "w") as f:
        json.dump(index_venue, f)
    with open("../data/category_pairs.json", "w") as f:
        json.dump(category_pairs, f)
    return np.hstack((spatial_features, temporal_features, categorical_features))

'''
FEATURE ENGINEERING FOR UNSEEN VENUES ON VALIDATION / TEST SET
'''

def generate_categorical_features_vt(category):
    with open("../data/category_pairs.json", "r") as f:
        category_pairs = json.load(f)
    subset = {key: val for key, val in category_pairs.iteritems() if key.startswith(category)}
    hasher = FeatureHasher(n_features=400)
    hasher.transform([subset]).toarray()
    return hasher.transform([subset]).toarray()

def generate_temporal_features_vt(hours, weekends):
    hour_dict = {str(h): c for h, c in Counter(hours).items()}
    hasher = FeatureHasher(n_features=24)
    hour_features = hasher.transform([hour_dict]).toarray()
    weekend_feature = np.array([[np.sum(weekends)/float(len(weekends))]])
    return np.hstack((hour_features, weekend_feature))

def generate_features_vt(venue, datapath):
    checkins = pd.read_csv(datapath, converters={"time": convert_time})
    checkins = checkins[checkins["venue_id"]==venue]
    keep_useful_features(checkins)
    category = checkins["venue_category"].values[0]
    categorical_features = generate_categorical_features_vt(category)
    hours, weekends = list(checkins["hour"].values), list(checkins["weekend"].values)
    temporal_features = generate_temporal_features_vt(hours, weekends)
    lat, lon = checkins["lat"].values[0], checkins["lon"].values[0]
    spatial_features = np.array([[lat, lon]])
    return np.hstack((spatial_features, temporal_features, categorical_features))
