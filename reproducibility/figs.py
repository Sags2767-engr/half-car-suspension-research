"""
figs.py  -  Generates all manuscript figures into figs/. Needs results.json and results2.json.
"""
import numpy as np, json, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle,FancyBboxPatch
from real import *
plt.rcParams.update({'font.family':'serif','font.size':9,'axes.grid':True,'grid.alpha':.3,'axes.spines.top':False,'axes.spines.right':False})
R=json.load(open('results.json'));Q=json.load(open('results2.json'));KB=np.array(R['K_B']);P=NOM;L=P['Lf']+P['Lr']
col={'passive':'#555555','passive-tuned':'#2a9d8f','skyhook':'#e08a00','LQR-B':'#c1121f','real':'#1d4e89'}
# ---- Fig 1 schematic
fig,ax=plt.subplots(figsize=(6.2,3.4));ax.set_xlim(0,10);ax.set_ylim(0,6.4);ax.axis('off')
ax.add_patch(Rectangle((1,4.2),8,0.7,fc='#d9d9d9',ec='k'));ax.text(3.2,4.55,'Body  $m_b$, $I_b$',ha='center',va='center');ax.plot([5],[4.55],'ko',ms=3);ax.text(5.1,5.2,'$x_b,\\ \\theta$',fontsize=9)
def spring(x,y0,y1,n=6,w=.18):
    ys=np.linspace(y0,y1,2*n+1);xs=[x]+[x+(w if i%2 else -w) for i in range(1,2*n)]+[x];ax.plot(xs,ys,'k',lw=1)
def damper(x,y0,y1):
    m=(y0+y1)/2;ax.plot([x,x],[y0,m-.15],'k',lw=1);ax.plot([x-.18,x-.18,x+.18,x+.18],[m+.15,m-.15,m-.15,m+.15],'k',lw=1);ax.plot([x,x],[m,y1],'k',lw=1)
for x,sub in ((2,'r'),(8,'f')):
    spring(x-.45,3.0,4.2);damper(x+.45,3.0,4.2);ax.add_patch(Rectangle((x-1,2.4),2,.6,fc='#bdbdbd',ec='k'));ax.text(x,2.7,f'$m_{{w{sub}}}$',ha='center',va='center')
    spring(x,1.0,2.4,n=5);ax.plot([x-1.2,x+1.2],[1.0,1.0],'k',lw=.8)
    ax.text(x-1.05,3.6,f'$k_{{b{sub}}}$',ha='right');ax.text(x+.65,3.6,f'$c_{{b{sub}}}$',ha='left');ax.text(x+.3,1.7,f'$k_{{w{sub}}}$',ha='left')
    ax.annotate('',xy=(x,3.55),xytext=(x,3.0),arrowprops=dict(arrowstyle='->',color='#c1121f'));ax.text(x,3.68,f'$u_{sub}$',ha='center',color='#c1121f',fontsize=8)
    ax.text(x,.55,f'$w_{sub}$',ha='center');ax.text(x-1.05,2.7,f'$x_{{w{sub}}}$',ha='right');ax.text(x+.2,4.95,f'$x_{{b{sub}}}$',ha='left')
ax.annotate('',xy=(8,6.0),xytext=(5,6.0),arrowprops=dict(arrowstyle='<->'));ax.text(6.5,6.1,'$L_f$',ha='center')
ax.annotate('',xy=(2,6.0),xytext=(5,6.0),arrowprops=dict(arrowstyle='<->'));ax.text(3.5,6.1,'$L_r$',ha='center')
plt.tight_layout();plt.savefig('figs/fig_model.png',dpi=300);plt.close()
# ---- Fig 2 block diagram
fig,ax=plt.subplots(figsize=(7.2,2.3));ax.set_xlim(0,14);ax.set_ylim(0,4.4);ax.axis('off')
def box(x,y,w,h,t,fc='#eeeeee'):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.02',fc=fc,ec='k',lw=.9));ax.text(x+w/2,y+h/2,t,ha='center',va='center',fontsize=8)
def arr(x0,y0,x1,y1):ax.annotate('',xy=(x1,y1),xytext=(x0,y0),arrowprops=dict(arrowstyle='->',lw=.9))
box(0.1,2.7,1.6,1.0,'Road\n$w_f,\\ w_r$');box(2.5,2.7,2.6,1.0,'Half-car plant\n(Eq. 7)',fc='#dbe7f3');box(6.0,2.7,2.0,1.0,'Sensors + noise\n$s_f,s_r,\\ \\ddot{x}_b,\\ \\ddot{x}_w$')
box(8.9,2.7,2.3,1.0,'Kalman filter\n$\\hat{s}$',fc='#f6e0e0');box(8.9,0.6,2.3,1.0,'Washout 0.5 Hz\n(body velocity)');box(5.8,0.6,2.4,1.0,'LQR gain  $-K$');box(2.9,0.6,2.2,1.0,'Limit $\\pm F_{max}$');box(0.1,0.6,2.0,1.0,'Actuator lag\n$f_{bw}$')
arr(1.7,3.2,2.5,3.2);arr(5.1,3.2,6.0,3.2);arr(8.0,3.2,8.9,3.2);arr(10.05,2.7,10.05,1.6);arr(8.9,1.1,8.2,1.1);arr(5.8,1.1,5.1,1.1);arr(2.9,1.1,2.1,1.1);arr(1.1,1.6,1.1,2.35);ax.plot([1.1,3.8],[2.35,2.35],'k',lw=.9);arr(3.8,2.35,3.8,2.7)
ax.text(1.15,2.0,'$u$',fontsize=8);plt.tight_layout();plt.savefig('figs/fig_chain.png',dpi=300);plt.close()
# ---- Fig 3 bump histories
B=np.load('figs/base_all.npz') if False else None
S=3.5;pp=dict(P);pp['cbf']*=S;pp['cbr']*=S;tb=np.arange(0,15,0.001);rd=road_bump(0.05,1.0,10.0,L/10.0,tb)
H={}
for n,(p,c) in {'passive':(P,None),'passive-tuned':(pp,None),'LQR-B':(P,('lqr',KB))}.items():
    r=simulate(p,c,rd,tb);H[n]=dict(xb=r['xb'],th=r['th'],ab=r['ab'],u=r['u'][:,0])
D=design(P);Xs,Xh,Uc=run(P,KB,rd,tb,D,mode='full',seed=1);o=outputs(P,Xs,Xh,rd,tb);H['real']=dict(xb=o['xb'],th=o['th'],ab=o['ab'],u=Xs[:,8])
fig,ax=plt.subplots(2,2,figsize=(7.2,5.2),sharex=True)
lab={'passive':'passive (original)','passive-tuned':'passive (tuned damping)','LQR-B':'LQR-B ideal','real':'LQR-B realistic chain'};ls={'passive':'-','passive-tuned':'--','LQR-B':'-','real':'-'}
for a,(k,tl,sc) in zip(ax.ravel(),[('xb','Body displacement $x_b$ (mm)',1e3),('th','Pitch angle $\\theta$ (mrad)',1e3),('ab','Body acceleration (m/s$^2$)',1),('u','Front actuator force $u_f$ (N)',1)]):
    for n in ('passive','passive-tuned','LQR-B','real'):
        if k=='u' and n.startswith('passive'):continue
        a.plot(tb,H[n][k]*sc,color=col[n],ls=ls[n],lw=1.0 if n=='real' else 1.2,alpha=.85 if n=='real' else 1,label=lab[n])
    a.set_ylabel(tl);a.set_xlim(0,6)
for a in ax[1]:a.set_xlabel('Time (s)')
ax[0,0].legend(frameon=False,fontsize=7);plt.tight_layout();plt.savefig('figs/fig_bump.png',dpi=300);plt.close()
# ---- Fig 4 Pareto
P_=R['pareto'];fig,ax=plt.subplots(figsize=(4.6,3.6));mk={1.0:'o',0.1:'s',0.01:'^'}
for ru in (1.0,0.1,0.01):
    pts=[p for p in P_['lqr'] if p['ru']==ru];ax.plot([p['ab_rms'] for p in pts],[p['dtl_rms'] for p in pts],marker=mk[ru],ms=3.5,lw=.8,color='#c1121f',alpha={1.0:.45,0.1:1,0.01:.7}[ru],label=f'LQR, $r_u$={ru}')
ax.plot([p['ab_rms'] for p in P_['sky']],[p['dtl_rms'] for p in P_['sky']],'-.',color=col['skyhook'],marker='d',ms=3.5,lw=1,label='Active skyhook')
ax.plot([p['ab_rms'] for p in P_['popt']],[p['dtl_rms'] for p in P_['popt']],'--',color=col['passive-tuned'],marker='v',ms=3.5,lw=1,label='Passive, damping $\\times$1 to 6')
ax.plot(P_['passive']['ab_rms'],P_['passive']['dtl_rms'],'k*',ms=9,label='Passive (original)')
for nm,(b,ru) in {'C':(0.01,0.1),'B':(0.3,0.1),'H':(3,0.1)}.items():
    q=min([p for p in P_['lqr'] if p['ru']==ru],key=lambda p:abs(np.log(p['beta']/b)));ax.annotate(nm,(q['ab_rms'],q['dtl_rms']),textcoords='offset points',xytext=(-9,5),fontsize=8,fontweight='bold')
ax.set_xlabel('RMS body acceleration (m/s$^2$)');ax.set_ylabel('RMS dynamic tire load / static load');ax.set_xlim(0,0.42);ax.set_ylim(0.017,0.062);ax.legend(frameon=False,fontsize=6.5,loc='lower left')
plt.tight_layout();plt.savefig('figs/fig_pareto.png',dpi=300);plt.close()
# ---- Fig 5 PSD
fig,ax=plt.subplots(figsize=(4.6,3.2))
for n in ('passive','passive-tuned','skyhook','LQR-B'):
    f,Pxx=R['psd'][n];ax.loglog(f[1:],np.array(Pxx[1:]),color=col[n],ls={'passive':'-','passive-tuned':'--','skyhook':'-.','LQR-B':'-'}[n],lw=1.1,label=n)
ax.set_xlim(0.3,30);ax.set_ylim(1e-6,1e1);ax.set_xlabel('Frequency (Hz)');ax.set_ylabel('PSD of body acceleration ((m/s$^2$)$^2$/Hz)');ax.legend(frameon=False,fontsize=7.5)
plt.tight_layout();plt.savefig('figs/fig_psd.png',dpi=300);plt.close()
# ---- Fig 6 MC overlay
mi=R['mc2'];mr=Q['mc_real'];fig,ax=plt.subplots(1,3,figsize=(7.2,2.5))
for a,(ki,kr,lb) in zip(ax,[('ab_red','ab_red','Reduction in RMS body acc. (%)'),('dtl_chg','dtl_chg','Change in RMS tire load (%)'),('s_chg','s_chg','Change in RMS susp. deflection (%)')]):
    di=[m[ki] for m in mi];dr=[m[kr] for m in mr];bins=np.linspace(min(min(di),min(dr)),max(max(di),max(dr)),22)
    a.hist(di,bins=bins,color='#888888',alpha=.65,label='ideal state feedback',density=True);a.hist(dr,bins=bins,color='#c1121f',alpha=.65,label='realistic chain',density=True);a.set_xlabel(lb,fontsize=8);a.set_yticks([])
ax[0].legend(frameon=False,fontsize=6.5,loc='upper left');ax[0].set_ylabel('Density')
plt.tight_layout();plt.savefig('figs/fig_mc.png',dpi=300);plt.close()
print('figs ok')