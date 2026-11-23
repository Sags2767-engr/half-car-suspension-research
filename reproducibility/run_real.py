"""
run_real.py  -  ISO-weighted RMS, realism ladder, actuator sensitivity, Monte Carlo of the realistic chain.
Reads results.json, writes results2.json. Run after run_base.py and run_mc.py.
"""
import numpy as np, json, sys
from scipy.signal import butter, filtfilt
from real import *
R=json.load(open('results.json'));KB=np.array(R['K_B']);P=NOM;L=P['Lf']+P['Lr'];S=3.5
pp=dict(P);pp['cbf']*=S;pp['cbr']*=S
KC,KH=np.array(R['K_C']),np.array(R['K_H']);CS=R['csky']
bl,al=butter(2,5.0/(0.5/0.001));
def ts_filt(t,xb): return settle(t,filtfilt(bl,al,xb))
out={}
# ---------- A. weighted RMS for all controllers on random roads (continuous model, dt=2 ms, 20 realizations)
dt=0.002;t=np.arange(0,20,dt)
ctr={'passive':(P,None),'passive-tuned':(pp,None),'skyhook':(P,('sky',CS,CS)),'LQR-C':(P,('lqr',KC)),'LQR-B':(P,('lqr',KB)),'LQR-H':(P,('lqr',KH))}
rc={'B_v10':(64e-6,10.0),'B_v20':(64e-6,20.0),'B_v30':(64e-6,30.0),'C_v20':(256e-6,20.0)}
W={}
for cn,(Gd,v) in rc.items():
    W[cn]={n:[] for n in ctr}
    for i in range(20):
        rd=road_iso(Gd,v,L/v,t,np.random.default_rng(2000+i))
        for n,(p,c) in ctr.items():
            r=simulate(p,c,rd,t);W[cn][n].append(wrms(r['ab'],dt,skip=1000))
    W[cn]={n:(float(np.mean(a)),float(np.std(a,ddof=1))) for n,a in W[cn].items()}
out['wrms']=W;print('A done',flush=True)
# ---------- B. realism ladder (LQR-B): ideal / actuator limits / actuator + observer + noise
dt=0.001
def ladder_bump(h,v):
    t=np.arange(0,15,dt);rd=road_bump(h,1.0,v,L/v,t);res={}
    r=simulate(P,('lqr',KB),rd,t);m=metrics(P,r);m['ts']=ts_filt(t,r['xb']);m['sat']=0.0;m['tf_err']=0.0;res['ideal']=m
    D=design(P)
    for mode,key in (('act','act'),('full','full')):
        Xs,Xh,Uc=run(P,KB,rd,t,D,mode=mode,seed=1);o=outputs(P,Xs,Xh,rd,t);m=metrics(P,dict(o,wf=rd[0],wr=rd[1]))
        Xn,Xhn,Ucn=run(P,KB,rd,t,D,mode=mode,seed=1,noise=False);on=outputs(P,Xn,Xhn,rd,t)
        m['ts']=ts_filt(t,on['xb']);m['jit']=rms(o['xb'][-5000:]-on['xb'][-5000:]);m['sat']=float(np.mean(np.abs(Uc)>=D['act']['fmax']-1e-6));m['tf_err']=rms(Xs[:,2]-Xh[:,2]);res[key]=m
    return res
out['ladder_bump']={'h5':ladder_bump(0.05,10.0),'h8':ladder_bump(0.08,10.0)}
def ladder_rand(Gd,v,n=20):
    t=np.arange(0,20,dt);D=design(P);acc={k:[] for k in ('ideal','act','full')}
    for i in range(n):
        rd=road_iso(Gd,v,L/v,t,np.random.default_rng(2000+i))
        r=simulate(P,('lqr',KB),rd,t);m=metrics(P,r);m['wab']=wrms(r['ab'],dt,skip=2000);m['sat']=0.0;m['tf_err']=0.0;m['tf_rms']=rms(r['tf']);acc['ideal'].append(m)
        for mode in ('act','full'):
            Xs,Xh,Uc=run(P,KB,rd,t,D,mode=mode,seed=10+i);o=outputs(P,Xs,Xh,rd,t);m=metrics(P,dict(o,wf=rd[0],wr=rd[1]))
            m['wab']=wrms(o['ab'],dt,skip=2000);m['sat']=float(np.mean(np.abs(Uc)>=D['act']['fmax']-1e-6));m['tf_err']=rms(Xs[:,2]-Xh[:,2]);m['tf_rms']=rms(Xs[:,2]);acc[mode].append(m)
    return {k:{q:(float(np.mean([m[q] for m in ms])),float(np.std([m[q] for m in ms],ddof=1))) for q in ms[0]} for k,ms in acc.items()}
out['ladder_rand']={cn:ladder_rand(*rc[cn]) for cn in ('B_v20','C_v20')};print('B done',flush=True)
# ---------- C. actuator bandwidth and force-limit sensitivity (state feedback, LQR-B)
def sens_case(fbw,fmax):
    act=dict(fbw=fbw,fmax=fmax);D=design(P,act=act);res={}
    # stability of the lagged loop without saturation
    Ad,Bd,E=D['Ad'],D['Bd'],D['E'];tau=D['tau'];A=np.zeros((10,10));A[:8,:8]=Ad;A[:8,8:]=Bd;A[8:,:8]=-KB/tau;A[8:,8:]=-np.eye(2)/tau
    res['stable']=bool(np.linalg.eigvals(A).real.max()<0);res['maxeig']=float(np.linalg.eigvals(A).real.max())
    t=np.arange(0,20,dt);ms=[]
    for i in range(20):
        rd=road_iso(64e-6,20.0,L/20.0,t,np.random.default_rng(2000+i));Xs,Xh,Uc=run(P,KB,rd,t,D,mode='act');o=outputs(P,Xs,Xh,rd,t);ms.append(metrics(P,dict(o,wf=rd[0],wr=rd[1])))
    res['ab_rms']=float(np.mean([m['ab_rms'] for m in ms]));res['dtl_rms']=float(np.mean([m['dtl_rms'] for m in ms]))
    tb=np.arange(0,15,dt);rd=road_bump(0.05,1.0,10.0,L/10.0,tb);Xs,Xh,Uc=run(P,KB,rd,tb,D,mode='act');o=outputs(P,Xs,Xh,rd,tb);mb=metrics(P,dict(o,wf=rd[0],wr=rd[1]))
    res['ts']=ts_filt(tb,o['xb']);res['ab_pk5']=mb['ab_pk']
    rd=road_bump(0.08,1.0,10.0,L/10.0,tb);Xs,Xh,Uc=run(P,KB,rd,tb,D,mode='act');o=outputs(P,Xs,Xh,rd,tb);mb=metrics(P,dict(o,wf=rd[0],wr=rd[1]));res['dtl_pk8']=mb['dtl_pk']
    return res
out['sens']={f'{b}_{f}':sens_case(b,f) for b,f in [(5,2000),(10,2000),(20,2000),(50,2000),(20,1000),(20,1500),(20,3000)]};print('C done',flush=True)
# ---------- D. Monte Carlo with realistic chain (+-20 % plant errors, nominal LQR-B and nominal Kalman filter)
rng=np.random.default_rng(7);keys=['mb','Ib','mwf','mwr','kbf','kbr','kwf','kwr','cbf','cbr'];mc=[];Dn=design(P);dtm=0.002;tm=np.arange(0,20,dtm)
for i in range(100):
    p=dict(P)
    for k in keys:p[k]=P[k]*rng.uniform(0.8,1.2)
    pt=dict(p);pt['cbf']*=S;pt['cbr']*=S
    Dp=design(p);D2=dict(Dn);D2['plant']=dict(Phi=Dp['Phi'],Gam=Dp['Gam'],C=Dp['C'])
    # closed-loop spectral radius (no noise, no saturation)
    Phi,Gam,C,Lk=Dn['Phi'],Dn['Gam'],Dn['C'],Dn['L'];Pp,Gp,Cp=Dp['Phi'],Dp['Gam'],Dp['C'];K10=np.hstack([KB,np.zeros((2,2))])
    Gu=Gp[:,:2];Gun=Gam[:,:2];wi=WASH['idx'];aw=np.exp(-2*np.pi*WASH['fc']*Dn['dt']);nW=len(wi);Sel=np.zeros((nW,10))
    for j,q in enumerate(wi):Sel[j,q]=1
    Kz=-(K10@Sel.T);Acl=np.zeros((20+nW,20+nW));Acl[:10,:10]=Pp;Acl[:10,10:20]=-Gu@K10;Acl[:10,20:]=-Gu@Kz
    Acl[10:20,:10]=Lk@Cp;Acl[10:20,10:20]=Phi-Gun@K10-Lk@C;Acl[10:20,20:]=-Gun@Kz;Acl[20:,10:20]=(1-aw)*Sel;Acl[20:,20:]=aw*np.eye(nW)
    rho=float(np.abs(np.linalg.eigvals(Acl)).max())
    rd=road_iso(64e-6,20.0,L/20.0,tm,np.random.default_rng(5000+i))
    t1=np.arange(0,20,0.001);rd1=road_iso(64e-6,20.0,L/20.0,t1,np.random.default_rng(5000+i))
    Xs,Xh,Uc=run(p,KB,rd1,t1,D2,mode='full',seed=100+i);o=outputs(p,Xs,Xh,rd1,t1);ma=metrics(p,dict(o,wf=rd1[0],wr=rd1[1]))
    mt=metrics(pt,simulate(pt,None,rd,tm));
    mc.append(dict(ab_red=100*(1-ma['ab_rms']/mt['ab_rms']),dtl_chg=100*(ma['dtl_rms']/mt['dtl_rms']-1),s_chg=100*(ma['s_rms']/mt['s_rms']-1),u_rms=ma['u_rms'],rho=rho))
out['mc_real']=mc;print('D done',flush=True)
json.dump(out,open('results2.json','w'))