"""
run_mc.py  -  Monte Carlo (200 samples, +-20 % on ten parameters), ideal state feedback.
Adds key 'mc2' to results.json and writes mats.json (A, B, G). Run after run_base.py.
"""
import numpy as np, json
from hc import *
R=json.load(open('results.json'));KB=np.array(R['K_B']);P=NOM;L=P['Lf']+P['Lr'];S=3.5;t=np.arange(0,20,0.002)
rng=np.random.default_rng(7);keys=['mb','Ib','mwf','mwr','kbf','kbr','kwf','kwr','cbf','cbr'];mc=[];ev_all=[]
for i in range(200):
    p=dict(P)
    for k in keys:p[k]=P[k]*rng.uniform(0.8,1.2)
    pt=dict(p);pt['cbf']*=S;pt['cbr']*=S
    Ad,Bd,E,_,_,_,_=defmodel(p);ev=np.linalg.eigvals(Ad-Bd@KB).real.max()
    rd=road_iso(64e-6,20.0,L/20.0,t,np.random.default_rng(5000+i))
    mt=metrics(pt,simulate(pt,None,rd,t));ma=metrics(p,simulate(p,('lqr',KB),rd,t));m0=metrics(p,simulate(p,None,rd,t))
    mc.append(dict(ab_red=100*(1-ma['ab_rms']/mt['ab_rms']),dtl_chg=100*(ma['dtl_rms']/mt['dtl_rms']-1),s_chg=100*(ma['s_rms']/mt['s_rms']-1),ab_red0=100*(1-ma['ab_rms']/m0['ab_rms']),u_rms=ma['u_rms'],maxeig=float(ev)))
R['mc2']=mc;json.dump(R,open('results.json','w'))
for k in mc[0]:
    a=np.array([m[k] for m in mc]);print(k,round(a.mean(),1),round(a.std(ddof=1),1),round(a.min(),1),round(a.max(),1))
A,B,G=absmodel(P)
json.dump(dict(A=A.tolist(),B=B.tolist(),G=G.tolist()),open('mats.json','w'))