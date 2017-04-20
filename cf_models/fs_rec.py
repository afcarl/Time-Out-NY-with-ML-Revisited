import os, sys, random, math, time, json
import numpy as np
import fs_util
from sklearn.externals import joblib
from pprint import pprint

def fl():
    sys.stdout.flush()

# l_rec: list of recommended venues
# vMu: actually visited venues by user
# tau: 20
def AP(l_rec, vMu, tau):
    np = len(vMu)
    nc = 0.0
    mapr_user = 0.0
    for j,s in enumerate(l_rec):
        if j>=tau:
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

###
### PREDICTORS
###

class Pred:
    '''Implement generic predictor'''        
    
    def __init__(self):
        pass

    def Score(self, user_venues, all_venues):
        return {}

class PredSI(Pred):
    '''Implement venue-similarity based predictor'''

    def __init__(self, _v2u_tr, _A=0, _Q=1):
        Pred.__init__(self)
        self.v2u_tr = _v2u_tr
        self.Q = _Q
        self.A = _A

    def printati(self):
        print "PredSI(A=%f, Q=%f)" % (self.A, self.Q)
        
    def Match(self, v, u_venue):
        l1 = len(self.v2u_tr[v])
        l2 = len(self.v2u_tr[u_venue])
        up = float(len(self.v2u_tr[v] & self.v2u_tr[u_venue]))
        if up > 0:
            dn = math.pow(l1, self.A) * math.pow(l2, (1.0-self.A))
            return up/dn
        return 0.0

    def Score(self, user_venues, all_venues):
        v_scores = {}
        for v in all_venues:
            v_scores[v] = 0.0
            if not (v in self.v2u_tr):
                continue
            for u_venue in user_venues:
                if not (u_venue in self.v2u_tr):
                    continue
                v_match = self.Match(v, u_venue)
                v_scores[v] += math.pow(v_match, self.Q)
        return v_scores

class PredSIc(PredSI):
    '''Implement calibrated venue-similarity based predictor''' 

    def __init__(self, _s2u_tr, _A=0, _Q=1, f_hvenues=""):
        PredSI.__init__(self, _s2u_tr, _A, _Q)
        self.hvenues = {}
        # with open(f_hvenues,"r") as f:
        #     for line in f:
        #         s, v = line.strip().split()
        #         self.hvenues[s] = float(v)
        # self.THETA = 0.5
    
    def calibrate(self, sco, venue):
        h = self.hvenues[venue]
        theta = self.THETA
        prob = sco
        if sco < h:
            prob = theta*sco/h
        elif sco > h:
            prob = theta+(1.0-theta)*(sco-h)/(1.0-h)
        return prob

    def Score(self, user_venues, all_venues):
        np = len(user_venues)
        v_scores, cs_venues = {}, set()
        for v in all_venues:
            v_scores[v] = 0.0
            for u_venue in user_venues:
                if u_venue not in self.v2u_tr:
                    if u_venue not in cs_venues:
                        cs_venues.add(u_venue)
                    continue
                v_match = self.Match(v, u_venue)
                v_scores[v] += math.pow(v_match, self.Q)/np
        # for v in all_venues:
        #     if v in self.hvenues:
        #         v_scores[v] = self.calibrate(v_scores[v], v)
        #     else:
        #         v_scores[v] = 0.0
        # return v_scores
        return v_scores, cs_venues

    
class PredSU(Pred):

    '''Implement user-similarity based predictor'''
    
    def __init__(self, _u2v_tr, _A=0, _Q=1):
        Pred.__init__(self)
        self.u2v_tr = _u2v_tr
        self.Q = _Q
        self.A = _A
    
    def printati(self):
        print "PredSU(A=%f, Q=%f)" % (self.A, self.Q)

    def Score(self, user_venues, all_venues):
        v_scores = {}
        for u_tr in self.u2v_tr:
            w = float(len(self.u2v_tr[u_tr] & user_venues))
            if w > 0:
                l1 = len(user_venues)
                l2 = len(self.u2v_tr[u_tr])
                w /= (math.pow(l1, self.A) * (math.pow(l2, (1.0-self.A))))
                w = math.pow(w, self.Q)
            for v in self.u2v_tr[u_tr]:
                if v in v_scores:
                    v_scores[v] += w
                else:
                    v_scores[v] = w
        return v_scores

###
### RECOMMENDERS
###

class Reco:

    '''Implements Recommender'''

    def __init__(self, _all_venues, _tau, _Gamma, _vpath=""):
        self.predictors=[]
        self.all_venues = _all_venues
        self.tau = _tau
        self.Gamma = _Gamma
        self.vpath = _vpath

    def Add(self, p):
        self.predictors.append(p)

    def GetStocIndex(self, n, distr):
        r = random.random()
        for i in range(n):
            if r < distr[i]:
                return i
            r -= distr[i]
        return 0
        
    def GetStochasticRec(self, venues_sorted, distr):
        nPreds = len(self.predictors)
        r = []
        ii = [0]*nPreds
        while len(r) < self.tau:
            pi = self.GetStocIndex(nPreds, distr)
            v = venues_sorted[pi][ii[pi]]
            if not v in r:
                r.append(v)
            ii[pi] += 1
        return r

    def Valid(self, T, users_te, u2v_v, u2v_h):
        ave_AP = 0.0
        rec = []
        start = time.clock()
        for i,ru in enumerate(users_te):
            '''
            if ru in u2v_v:
                print "%d] scoring user %s with %d venues" % (i, ru, len(u2v_v[ru]))
            fl()
            '''
            venues_sorted = []
            for p in self.predictors:
                svenues = []
                if ru in u2v_v:
                    try:
                        v_scores, cs_venues = p.Score(u2v_v[ru], self.all_venues)
                    except:
                        v_scores = p.Score(u2v_v[ru], self.all_venues)
                        cs_venues = None
                else:
                    svenues = list(self.all_venues)

                if cs_venues:
                    clf = joblib.load("neigh.pkl")
                    with open("../data/index_venue.json", "r") as f:
                        index_venue = json.load(f)
                    venues_ordered = fs_util.sort_venues("../data/train_data.csv", binary=True)
                    n_neighbors = 14
                    n_populars = self.tau - n_neighbors
                    for v in cs_venues:
                        X = fs_util.generate_features_vt(v, self.vpath)
                        indices = clf.kneighbors(X, n_neighbors=n_neighbors, return_distance=False).flatten()
                        for idx in indices:
                            v = index_venue[str(idx)]
                            v_scores[v] += .01
                        for v in venues_ordered[:n_populars]:
                            v_scores[v] += .02

                svenues = fs_util.sort_dict_dec(v_scores)

                cleaned_venues = []
                for x in svenues:
                    if len(cleaned_venues) >= self.tau: 
                        break
                    if ru not in u2v_v or x not in u2v_v[ru]:
                         cleaned_venues.append(x)
                                        
                venues_sorted += [cleaned_venues]
                
            rec += [self.GetStochasticRec(venues_sorted, self.Gamma)]

        cti = time.clock()-start
        print "Processed in %f secs" % (cti)
        fl()
        map_ = mAP(users_te, rec, u2v_h, self.tau)
        print "MAP(%d): %f" % (T, map_)
        fl()
    
        print "Done!\n"

    def RecommendToUser(self, user, u2v_v):
        venues_sorted = []
        for p in self.predictors:
            svenues = []
            if user in u2v_v:
                svenues = fs_util.sort_dict_dec(p.Score(u2s_v[user], self.all_venues))
            else:
                svenues = list(self.all_venues)

            cleaned_venues = []
            for x in svenues:
                if len(cleaned_venues) >= self.tau:
                    break
                if x not in u2v_v[user]:
                    cleaned_venues.append(x)

            venues_sorted += [cleaned_venues]

        return self.GetStochasticRec(venues_sorted, self.Gamma)

    def RecommendToUsers(self, l_users, u2v_v):
        sti = time.clock()
        rec4users = []
        for i,u in enumerate(l_users):
            if not (i+1)%10:
                if u in u2s_v:
                    print "%d] %s w/ %d venues" % (i+1, l_users[i], len(u2v_v[u]))
                fl()
            rec4users.append(self.RecommendToUser(u, u2v_v))
            cti = time.clock()-sti
            if not (i+1)%10:
                print " tot secs: %f (%f)" % (cti, cti/(i+1))
            fl()
        return rec4users
