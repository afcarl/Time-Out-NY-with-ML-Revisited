# http://rnowling.github.io/data/science/2016/10/20/lr-hashing-recsys.html
# https://github.com/rnowling/rec-sys-experiments

import numpy as np, scipy.sparse as sp
import random, sys
from sklearn.feature_extraction import FeatureHasher
from sklearn.linear_model import SGDClassifier
# from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix
from sklearn.externals import joblib
import data_helper
import matplotlib.pyplot as plt
plt.style.use("fivethirtyeight")

# paths to data
_tr = "../data/train_data.csv"
_vv = "../data/valid_visible.csv"
_vp = "../data/valid_predict.csv"

def generate_interaction(_tr, _vv, _vp):
	print "Creating user venue-interaction lists"
	_, all_venues = data_helper.get_unique(_tr, users=False, venues=True)
	train_pairs, valid_pairs = data_helper.get_user_venue_pairs(_tr, _vv, _vp)
	return all_venues, train_pairs, valid_pairs

def generate_features(all_venues, yay_venues):
	yay_pairs, nay_pairs = [], []

	# positive examples
	for venue1 in yay_venues:
		venue_pairs = dict()
		for venue2 in yay_venues:
			# skip itself to avoid overfitting
			if venue1 != venue2:
				venue_pairs["%s_%s" % (venue1, venue2)] = 1.
		yay_pairs.append(venue_pairs)

	# negative examples
	nay_venues = all_venues - yay_venues
	for venue1 in random.sample(nay_venues, len(yay_venues)):
		venue_pairs = dict()
		for venue2 in yay_venues:
			venue_pairs["%s_%s" % (venue1, venue2)] = 1.
		nay_pairs.append(venue_pairs)

	labels = np.hstack([np.ones(len(yay_pairs)), np.zeros(len(nay_pairs))])
	return labels, yay_pairs, nay_pairs


def train_and_score(_tr, _vv, _vp, model_sizes, colors=None):
	all_venues, train_pairs, valid_pairs = generate_interaction(_tr, _vv, _vp)
	
	print "Creating models"

	plt.figure(figsize=(10,10)); lw = 2
	roc_aucs = []
	for size, color in zip(model_sizes, colors):
		extractor = FeatureHasher(n_features=2**size)
		model = SGDClassifier(loss="log", penalty="l2", alpha=0.001, n_jobs=-1)
		# model = BernoulliNB()

		print "Training"
		for i, (user, yay_venues) in enumerate(train_pairs.iteritems()):
			print "Training on user", i, user
			labels, yay_pairs, nay_pairs = generate_features(all_venues, yay_venues)
			yay_features, nay_features = extractor.transform(yay_pairs), extractor.transform(nay_pairs)
			features = sp.vstack([yay_features, nay_features])
			model.partial_fit(features, labels, classes=[0, 1])

		print "Testing"
		all_labels, all_preds, all_probas = [], [], []
		for i, (user, yay_venues) in enumerate(valid_pairs.iteritems()):
			print "Testing on user", i, user
			labels, yay_pairs, nay_pairs = generate_features(all_venues, yay_venues)
			all_labels.extend(labels)
			yay_features, nay_features = extractor.transform(yay_pairs), extractor.transform(nay_pairs)
			features = sp.vstack([yay_features, nay_features])
			preds, probas = model.predict(features), model.predict_proba(features)
			all_preds.extend(preds), all_probas.extend(probas[:, 1])

		print "Scoring"
		roc_auc = roc_auc_score(all_labels, all_probas)
		cm = confusion_matrix(all_labels, all_preds)
		print "Model size", size, "AUC", roc_auc
		print cm
		roc_aucs.append(roc_auc)
		fpr, tpr, _ = roc_curve(all_labels, all_probas)
		plt.plot(fpr, tpr, color=color,
			lw=lw, label='Model %d (area = %0.2f)' % (size, roc_auc))

	joblib.dump(model, 'model_logit_size%d.pkl' % size)
	np.save("labels_logit_size%d.npy" % size, all_labels)
	np.save("probas_logit_size%d.npy" % size, all_probas)

	plt.plot([0, 1], [0, 1], color='navy', lw=lw, ls='--', label='Luck')
	plt.xlim([-.05, 1.0])
	plt.ylim([0.0, 1.05])
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.title('Receiver operating characteristic for different model sizes')
	plt.legend(loc="lower right")
	# plt.savefig('../plots/model_nb.png')
	plt.tight_layout()
	plt.show()
	
	'''
	plt.figure(figsize=(15,9))
	plt.plot(model_sizes, roc_aucs, c='gray', ls='dashed', lw=1)
	for model_size, roc_auc in zip(model_sizes, roc_aucs):
		plt.plot(model_size, roc_auc, "*", markersize=12)
	plt.xlim([model_sizes[0]-.1, model_sizes[-1]+.1])
	plt.ylim([0.5, 0.85])
	plt.xlabel("Model Size")
	plt.ylabel("ROC AUC Score")
	plt.title('ROC AUC score for different model sizes')
	plt.savefig('../plots/auc_by_model_size_nb.png')
	plt.tight_layout()
	plt.show()
	'''

def main():
	# model size by number of bits
	# model_size = int(sys.argv[1])
	# model_sizes = range(10, 22)
	model_sizes = [20]
	colors = ['forestgreen']
	# colors = ['darkorange', 'skyblue', 'forestgreen', 'darkslategray', 'firebrick']
	train_and_score(_tr, _vv, _vp, model_sizes, colors)

if __name__=="__main__":
	main()