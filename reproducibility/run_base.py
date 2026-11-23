"""
run_base.py  -  Ideal-chain results: bumps, ISO 8608 random roads (20 realizations), trade-off sweep.
Writes results.json and figs/base.npz. Run first (needed by all other scripts).
"""
import numpy as np, json
from hc import *
from scipy.signal import welch
P=NOM;L=P['Lf']+P['Lr'];S=3.5
pp=dict(P);pp['cbf']=P['cbf']*S;pp['cbr']=P['cbr']*S
PRE={'C':dict(beta=0.01,ru=0.1),'B':dict(beta=0.3,ru=0.1),'H':dict(beta=3.0,ru=0.1)}
K={k:lqr(P,alpha=1,**v) for k,v in PRE.items()}
CS=12000.0
ctrls={'passive':(P,None),'passive-tuned':(pp,None),'skyhook':(P,('sky',CS,CS)),'LQR-C':(P,('lqr',K['C'])),'LQR-B':(P,('lqr',K['B'])),'LQR-H':(P,('lqr',K['H']))}
out={'K_B':K['B'].tolist(),'K_C':K['C'].tolist(),'K_H':K['H'].tolist(),'presets':PRE,'csky':CS}
tb=np.arange(0,15,0.001)
cases={'h3_v36':(0.03,10.0),'h5_v36':(0.05,10.0),'h8_v36':(0.08,10.0),'h5_v20':(0.05,20/3.6),'h5_v40':(0.05,40/3.6),'h5_v60':(0.05,60/3.6)}
out['bump']={cn:{n:metrics(p,simulate(p,c,road_bump(h,1.0,v,L/v,tb),tb),True) for n,(p,c) in ctrls.items()} for cn,(h,v) in cases.items()}
base={}
for n in ('passive','passive-tuned','skyhook','LQR-B'):
    p,c=ctrls[n];r=simulate(p,c,road_bump(0.05,1.0,10.0,L/10.0,tb),tb)
    base.update({f'{n}_{k}':r[k] for k in ('xb','th','ab','u')})
np.savez('figs/base.npz',t=tb,**base)
dt=0.002;t=np.arange(0,20,dt);NR=20
rc={'B_v10':(64e-6,10.0),'B_v20':(64e-6,20.0),'B_v30':(64e-6,30.0),'C_v20':(256e-6,20.0)}
rr={};psd={}
for cn,(Gd,v) in rc.items():
    runs={n:[] for n in ctrls}
    for i in range(NR):
        rd=road_iso(Gd,v,L/v,t,np.random.default_rng(2000+i))
        for n,(p,c) in ctrls.items():
            r=simulate(p,c,rd,t);runs[n].append(metrics(p,r))
            if cn=='B_v20' and i==0:
                f,Pxx=welch(r['ab'],fs=1/dt,nperseg=4096);psd[n]=(f.tolist(),Pxx.tolist())
    rr[cn]={n:{k:(float(np.mean([m[k] for m in ms])),float(np.std([m[k] for m in ms],ddof=1))) for k in ms[0]} for n,ms in runs.items()}
out['random']=rr;out['psd']=psd
RB=[road_iso(64e-6,20.0,L/20.0,t,np.random.default_rng(1000+i)) for i in range(8)]
def avg(p,c):
    ms=[metrics(p,simulate(p,c,r,t)) for r in RB];return {k:float(np.mean([m[k] for m in ms])) for k in ('ab_rms','dtl_rms','s_rms','u_rms','u_pk')}
par={'passive':avg(P,None),'lqr':[],'sky':[],'popt':[]}
for ru in (1.0,0.1,0.01):
    for b in np.logspace(-2,2,11):
        m=avg(P,('lqr',lqr(P,alpha=1,beta=b,ru=ru)));m.update(ru=ru,beta=float(b));par['lqr'].append(m)
for c in np.linspace(0,16000,17):
    m=avg(P,('sky',c,c));m['c']=float(c);par['sky'].append(m)
for s in [1,1.5,2,2.5,3,3.5,4,5,6]:
    q=dict(P);q['cbf']*=s;q['cbr']*=s;m=avg(q,None);m['s']=s;par['popt'].append(m)
out['pareto']=par
json.dump(out,open('results.json','w'))
print('base done')