import numpy as np, scipy.sparse as sp
from sklearn.externals import joblib
from sklearn.feature_extraction import FeatureHasher
import random
import data_helper

# paths to data
_tr = "../data/train_data.csv"
_tv = "../data/test_visible.csv"
_tp = "../data/test_predict.csv"

# l_rec: list of recommended venues
# vMu: actually visited venues by user
# tau: 20
def AP(l_rec, vMu, tau):
	np = len(vMu)
	nc = 0.0
	mapr_user = 0.0
	for j,s in enumerate(l_rec):
		if j >= tau:
			break
		if s in vMu:
			nc += 1.0
		mapr_user += nc/(j+1)
	mapr_user /= min(np, tau)
	return mapr_user

# l_users: list of users
# l_rec_venues: list of lists, recommended venues for users
# u2v: mapping users to venues
# tau: 20
def mAP(l_users, l_rec_venues, u2v, tau):
	mapr = 0
	n_users = len(l_users)
	for user, l_rec in zip(l_users, l_rec_venues):
		mapr += AP(l_rec, u2v[user], tau)
	return mapr/n_users

def generate_interaction(_tr, _tv, _tp):
	print "Creating user venue-interaction lists"
	_, all_venues = data_helper.get_unique(_tr, users=False, venues=True)
	testv_pairs, testp_pairs = data_helper.get_user_venue_pairs_eval(_tv, _tp)
	return all_venues, testv_pairs, testp_pairs

def generate_features_tr(all_venues, yay_venues_v):
	yay_pairs, nay_pairs, venues_to_catch = [], [], []

	# positive examples
	for venue1 in yay_venues_v:
		venue_pairs = dict()
		for venue2 in yay_venues_v:
			# skip itself to avoid overfitting
			if venue1 != venue2:
				venue_pairs["%s_%s" % (venue1, venue2)] = 1.
		yay_pairs.append(venue_pairs)

	# negative examples: at this point, we don't know what venues will be visited
	nay_venues = all_venues - yay_venues_v
	for venue1 in random.sample(nay_venues, len(yay_venues_v)):
		venue_pairs = dict()
		for venue2 in yay_venues_v:
			venue_pairs["%s_%s" % (venue1, venue2)] = 1.
		nay_pairs.append(venue_pairs)

	labels = np.hstack([np.ones(len(yay_pairs)), np.zeros(len(nay_pairs))])
	return labels, yay_pairs, nay_pairs

def generate_features_te(all_venues, yay_venues_p, yay_venues_v):
	yay_pairs, nay_pairs, venues_to_predict = [], [], []

	# positive examples
	for venue1 in yay_venues_p:
		venue_pairs = dict()
		for venue2 in yay_venues_v:
			venue_pairs["%s_%s" % (venue1, venue2)] = 1.
		yay_pairs.append(venue_pairs)
		venues_to_predict.append(venue1)

	# negative examples
	nay_venues = all_venues - yay_venues_v - yay_venues_p
	for venue1 in nay_venues:
		venue_pairs = dict()
		for venue2 in yay_venues_v:
			venue_pairs["%s_%s" % (venue1, venue2)] = 1.
		nay_pairs.append(venue_pairs)
		venues_to_predict.append(venue1)

	labels = np.hstack([np.ones(len(yay_pairs)), np.zeros(len(nay_pairs))])
	return labels, yay_pairs, nay_pairs, venues_to_predict

def test(tau):
	all_venues, testv_pairs, testp_pairs = generate_interaction(_tr, _tv, _tp)

	extractor = FeatureHasher(n_features=2**20)
	model = joblib.load('model_logit_size20.pkl')

	print "Training"
	for i, (user, yay_venues) in enumerate(testv_pairs.iteritems()):
		print "Training on user", i, user
		labels, yay_pairs, nay_pairs = generate_features_tr(all_venues, yay_venues)
		yay_features, nay_features = extractor.transform(yay_pairs), extractor.transform(nay_pairs)
		features = sp.vstack([yay_features, nay_features])
		model.partial_fit(features, labels, classes=[0, 1])

	print "Testing"
	l_users, l_rec_venues = [], []
	for i, (user, yay_venues) in enumerate(testp_pairs.iteritems()):
		print "Testing on user", i, user
		l_users.append(user)
		labels, yay_pairs, nay_pairs, v_to_p = generate_features_te(all_venues, yay_venues, testv_pairs[user])
		yay_features, nay_features = extractor.transform(yay_pairs), extractor.transform(nay_pairs)
		features = sp.vstack([yay_features, nay_features])
		probas = model.predict_proba(features)[:, 1]
		rec_venues = [v_to_p[i] for i in np.argsort(probas)[::-1][:20]]
		l_users.append(user), l_rec_venues.append(rec_venues)

	print "\nmAP(%d): %f" % (tau, mAP(l_users, l_rec_venues, testp_pairs, tau))

if __name__=="__main__":
	test(20)