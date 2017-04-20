import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score
plt.style.use("fivethirtyeight")

labels_probas_names = [('labels_gbm_size14.npy', 'probas_gbm_size14.npy', 'Gradient Boosting', 'darkorange'),
	('labels_rf_size20.npy', 'probas_rf_size20.npy', 'Random Forest', 'forestgreen'),
	('labels_nb_size19.npy', 'probas_nb_size19.npy', 'Naive Bayes', 'skyblue'),
	('labels_logit_size20.npy', 'probas_logit_size20.npy', 'Logit', 'firebrick')]

plt.figure(figsize=(10,10))
lw = 2

for (label_, proba_, name, color) in labels_probas_names:
	labels, probas = np.load(label_), np.load(proba_)
	roc_auc = roc_auc_score(labels, probas)
	fpr, tpr, _ = roc_curve(labels, probas)
	plt.plot(fpr, tpr, color=color,
		lw=lw, label='%s (area = %0.2f)' % (name, roc_auc))

plt.plot([0, 1], [0, 1], color='navy', lw=lw, ls='--', label='Luck')
plt.xlim([-.05, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic for different model sizes')
plt.legend(loc="lower right")
plt.savefig('../plots/roc_auc_all.png')
plt.tight_layout()
plt.show()