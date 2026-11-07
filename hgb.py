import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor


# Freeze definitions and hyperparameters before evaluating the account.
features = [
    'os', 'amp', 'ret', 'vr3', 'v5', 'v10', 't20', 't40',
    'p60', 'p120', 'lu', 'mom5', 'mom10', 'mom20', 'mom60',
    'ma5', 'ma10', 'ma20', 'ma60', 'vol10', 'vol20', 's20', 'break60',
]
parameters = dict(max_iter=100, learning_rate=.05, max_leaf_nodes=7,
                  min_samples_leaf=150, l2_regularization=20,
                  early_stopping=False, random_state=71)

def features_and_labels(p, days, universe):
    data = p[p.code.isin(universe)]
    chunks = []
    # Missing sessions remain missing, including suspensions in forward labels.
    calendar = days[days >= data.date.min()]
    label_date = pd.Series(calendar, index=calendar).shift(-4)
    for code, q in data.groupby('code', sort=True):
        g = q.set_index('date').reindex(calendar)
        c = g.qc
        for k in [5,10,20,60]:
            g[f'mom{k}'] = c/c.shift(k)-1
            g[f'ma{k}'] = c/c.rolling(k).mean()-1
        for k in [10,20]:
            g[f'vol{k}'] = g.ret.rolling(k).std()
        g['break60'] = c/c.shift().rolling(60).max()-1
        raw_buy = g.o.shift(-1)*1.001
        raw_sell = g.c.shift(-4)*.999
        qty = np.floor((200000/12)/raw_buy/100)*100
        # Avoid accidentally allocating a full extra lot beyond the budget after fees.
        cost = raw_buy*qty
        qty = qty.where(cost+np.maximum(cost*.0001,5)+cost*.00001<=200000/12, qty-100)
        cost = raw_buy*qty
        proceeds = raw_sell*qty
        buy_fee = np.maximum(cost*.0001,5)+cost*.00001
        sell_fee = np.maximum(proceeds*.0001,5)+proceeds*.00051
        adj_ratio = g.qc.shift(-4)/g.qo.shift(-1)
        target = (adj_ratio*(.999/1.001)*cost-sell_fee)/(cost+buy_fee)-1
        observed = g.buyable.eq(True)&g.lock.shift(-4).eq(False)&qty.ge(100)
        g['target'] = target.where(observed)
        g['label_end'] = label_date
        g['code'] = code
        eligible = (g.v>0)&(g.qc>0)&(g[features].notna().sum(axis=1)>=18)
        chunks.append(g.loc[eligible, features+['code','gap','buyable','target','label_end']].reset_index(names='date'))
    if not chunks:
        raise ValueError('HGB 股票池缺少有效行情')
    f = pd.concat(chunks, ignore_index=True)
    f[features+['target']] = f[features+['target']].replace([np.inf,-np.inf],np.nan)
    # Percentile transforms use only that signal day's visible cross-section.
    f[features] = f.groupby('date')[features].rank(pct=True, method='average').astype('float32')-.5
    return f

