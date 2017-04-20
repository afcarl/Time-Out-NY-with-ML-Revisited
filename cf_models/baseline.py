import fs_util, fs_rec

# paths to data
_tr = "../data/train_data.csv"
_vv = "../data/valid_visible.csv"
_vp = "../data/valid_predict.csv"

# parameters
_tau = 10

print 'default ordering by popularity'
venues_ordered = fs_util.sort_venues(_tr, binary=False)
# venues_ordered = fs_util.sort_venues(_tr, binary=True)

print 'user to venues on %s' % (_vv)
u2v_vv = fs_util.user_to_venues(_vv)
print 'user to venues on %s' % (_vp)
u2v_vp = fs_util.user_to_venues(_vp)

# recommend top N most popular songs (extremely unpersonalized :|)
all_recs = []
for u in u2v_vv:
    recs_ = set(venues_ordered[:_tau])-u2v_vv[u]
    recs4u = list(recs_)
    if len(recs4u)<_tau:
        n_more = _tau-len(recs4u)
        recs4u += venues_ordered[_tau:_tau+n_more]
    all_recs.append(recs4u)

map_all = fs_rec.mAP(u2v_vv.keys(), all_recs, u2v_vp, _tau)
print
print "mAP for %d users in the validation set: %f"%(len(u2v_vv), map_all)
# binary: 0.098924
# frequency-based: 0.062301