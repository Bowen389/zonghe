from pathlib import Path
import gzip
import json
import os
import numpy as np
import pandas as pd
ROOT=Path(os.environ.get('ZONGHE_DATA','market'))
def index(sym):
    record=json.loads((ROOT/'data'/f'{sym}.json').read_text())['data'][sym]
    return pd.Series({r[0]:float(r[2]) for r in record.get('day',record.get('qfqday'))}).sort_index()
def dev(x,n): return x/x.rolling(n).mean()-1
def build(days, meta):
    parts=[]
    for f in (ROOT/'data/stock').glob('*.json.gz'):
        with gzip.open(f,'rt') as stream: j=json.load(stream)
        c=j['code']
        if c not in meta.index or not j['raw'] or not j['qfq']: continue
        def frame(rows,cols,fields):
            a=pd.DataFrame([[row[i] if i<len(row) else None for i in fields] for row in rows],columns=['date',*cols])
            return a.drop_duplicates('date',keep='last').set_index('date').apply(pd.to_numeric,errors='coerce')
        a=frame(j['raw'],['o','c','h','l','v','to','amount'],[0,1,2,3,4,5,7,8])
        q=frame(j['qfq'],['qo','qc','qh','ql'],[0,1,2,3,4])
        a=a.join(q).reindex(days)
        for o,c_,h,l in [('o','c','h','l'),('qo','qc','qh','ql')]:
            bad=(a[h]<a[[o,c_,l]].max(axis=1)-1e-7)|(a[l]>a[[o,c_,h]].min(axis=1)+1e-7)|(a[l]<=0)
            a.loc[bad,:]=np.nan
        a.loc[a.qc.pct_change(fill_method=None).abs()>.25,:]=np.nan
        pc=a.c.shift()
        qp=a.qc.shift()
        a['ret']=a.qc/qp-1
        a['os']=a.qo/qp-1
        a['amp']=(a.qh-a.ql)/qp
        a['amount']*=10000
        a['s20']=a.amount.shift().rolling(20).mean()
        age=np.minimum(np.arange(len(days))-days.searchsorted(str(meta.loc[c,'ipoDate'])),a.c.notna().cumsum())
        no_st=not any(x in str(j.get('current_name','')).upper() for x in ['ST','退'])
        five=((a.c/pc-1).between(.047,.053))&((a.c-a.h).abs()<.006)
        st=five.rolling(120,min_periods=20).max().eq(1)&~a.ret.abs().gt(.075).rolling(120,min_periods=20).max().eq(1)
        a['base']=(a.v>0)&(age>=60)&(a.s20>=3e7)&(a.qc>=3)&no_st&a.to.notna()
        a['hbase']=(a.v>0)&(age>=60)&(a.s20>=3e7)&(a.c>=3)&~st&no_st
        a['pbase']=(a.v>0)&(age>=60)&~st&no_st
        upper=np.floor(pc*110+.50000001)/100
        lower=np.floor(pc*90+.50000001)/100
        lu=(a.c-upper).abs().lt(.006)&a.ret.ge(.08)
        broken=(a.h-upper).abs().lt(.006)&~lu
        first=lu&~lu.shift(fill_value=False)
        a['p2']=a.pbase&first&(a.h>a.l+.005)&a.to.between(3,15)
        n=lu.shift(2).rolling(5).sum()
        prevbroken=broken.shift(fill_value=False)
        bf=a.hbase&first&((pc<a.o.shift())|prevbroken)&(n>=1)&(a.c>np.maximum(pc,a.o.shift()))
        a['bfscore']=(n==1).astype(int)+(a.s20>=2.5e8).astype(int)+(a.o>=pc).astype(int)
        a['bf']=bf&(a.s20>=2e8)&~prevbroken&(a.bfscore>=2)
        a['fb']=a.hbase&first&(a.o<upper-.005)&(age>=120)&(a.amount>=1e8)
        a['pos']=qp/a.qc.shift().rolling(60).median()
        a['gap']=a.o.shift(-1)/a.c-1
        a['qgap']=a.qo.shift(-1)/a.qc-1
        a['buyable']=(a.o.shift(-1)>np.floor(a.c*90+.50000001)/100+.005)&(a.o.shift(-1)<np.floor(a.c*110+.50000001)/100-.005)&a.v.shift(-1).gt(0)
        a['lock']=a.c.le(lower+.005)&a.ret.le(-.075)
        a['trendlock']=a.ret.le(-.0985)
        a['lu']=a.ret.ge(.099).groupby((~a.ret.ge(.099)).cumsum()).cumsum()
        ma5,ma10,ma20=[a.qc.rolling(k).mean() for k in (5,10,20)]
        for k in (60,120):
            a[f'p{k}']=.4*((ma5>ma10)&(ma10>ma20))+.6*a.qc/a.qh.rolling(k).max()
        for k in (3,5,10):
            a[f'v{k}']=a.v/a.v.rolling(k).mean()
        for k in (20,40):
            t=a.to.rolling(k).mean()
            a[f't{k}']=np.nan
            for low,high in ((0,1),(1,3),(3,7),(7,np.inf)):
                s=t[t.ge(low)&t.lt(high)]
                a.loc[s.index,f't{k}']=-(s-s.cummin())/(s.cummax()-s.cummin()+1e-12)
        a['vr3']=a['v3']
        a['v3']=(a.ret.gt(.02)&a.v.gt(2*a.v.shift().rolling(5).mean()))|(a.qc.ge(a.qc.shift().rolling(60).max())&a.v.gt(1.2*a.v.shift().rolling(60).mean()))
        a['name']=j.get('current_name',c)
        a['code']=c
        a.index.name='date'
        parts.append(a.drop(columns=['h','l','qh','ql','to','pbase','hbase']).loc[a.c.notna()& (a.index>='2025-05-01')].reset_index())
    out=pd.concat(parts,ignore_index=True)
    return out

