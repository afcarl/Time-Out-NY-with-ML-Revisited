import numpy as np
import fs_rec, fs_util

# paths to data
_tr = "../data/train_data.csv"
_tv = "../data/test_visible.csv"
_tp = "../data/test_predict.csv"

# parameters
_A_optSI = 0.09
_Q_optSI = 0.9
_A_optSU = 0
_Q_optSU = 0.6
_tau = 20
_Gamma_opt = [0.1, 0.9]

print 'default ordering by popularity'
venues_ordered = fs_util.sort_venues(_tr, binary=True)

print 'venue to users on %s' % (_tr)
v2u_tr = fs_util.venue_to_users(_tr)

# print 'user to venues on %s' % (_tr)
# u2v_tr = fs_util.user_to_venues(_tr)

print 'user to venues on %s' % (_tv)
u2v_tv = fs_util.user_to_venues(_tv)
print 'user to venues on %s' % (_tp)
u2v_tp = fs_util.user_to_venues(_tp)

# coef_range = list(np.arange(11)/10.)
# Gammas = zip(coef_range, [1-c for c in coef_range])

print 'Creating predictors...'
predictorSI = fs_rec.PredSI(v2u_tr, _A_optSI, _Q_optSI)
predictorSI.printati()
predictorSU = fs_rec.PredSU(u2v_tr, _A_optSU, _Q_optSU)
predictorSU.printati()

print 'Creating recommender...'
recommender = fs_rec.Reco(venues_ordered, _tau, _Gamma_opt, _tv)
recommender.Add(predictorSIc)
recommender.Add(predictorSU)
print _Gamma_opt

recommender.Valid(_tau, u2v_tv.keys(), u2v_tv, u2v_tp)