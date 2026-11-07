"""Publish independent next-session candidate lists for seven strategies."""
import argparse
from datetime import datetime
import json
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from features import build, dev, index
from hgb import features, features_and_labels, parameters
from market_data import exchange, refresh

BASE = Path(__file__).resolve().parent
BACKTEST_OVERVIEW = """## 近一年回测总览

回测区间 **2025-11-07 至 2026-11-06** · 各策略原始仓位口径 · 单笔为已平仓交易净收益均值。

| 策略 | 起始资金 | 账户收益 | 最大回撤 | 夏普 | 平仓笔数 | 胜率 | 单笔 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| P2 首板低开 | 20 万元 | +76.20% | -28.84% | 1.40 | 198 | 48.48% | +0.69% |
| H1 BF 主线／FB 补位 | 4 万元 | +77.32% | -8.28% | 3.75 | 167 | 64.67% | +2.24% |
| 趋势 1 双门控稳健 | 6 万元 | +26.76% | -3.77% | 2.29 | 78 | 60.26% | +2.25% |
| 趋势 2 无门控稳健 | 6 万元 | +33.42% | -6.65% | 2.04 | 181 | 53.59% | +1.12% |
| 趋势 3 双门控三核心 | 6 万元 | +23.65% | -4.62% | 2.15 | 73 | 58.90% | +2.12% |
| 趋势 4 双门控自适应退出 | 6 万元 | +37.26% | -4.77% | 2.44 | 78 | 47.44% | +3.06% |
| HGB 稳健 Top3（23 因子重建） | 20 万元 | +22.54% | -13.42% | 1.52 | 178 | 56.18% | +1.60% |

"""

TITLES = {
    'P2':'P2 首板低开', 'H1':'H1 BF主线 / FB补位',
    'S2_1':'趋势1 双门控稳健', 'S2_2':'趋势2 无门控稳健',
    'S2_3':'趋势3 双门控三核心', 'S2_4':'趋势4 双门控自适应退出',
    'HGB':'HGB 稳健 Top3（23因子重建）',
}

def universe(name):
    return set((BASE/'universe'/f'{name}.txt').read_text().split())

def candidate(row, route='', score=None):
    return dict(code=row.code, name=getattr(row,'name',row.code),
                route=route, close=round(float(row.c),2),
                score=None if score is None else round(float(score),6))

def trend_scores(today):
    pool=today[today.base].copy()
    fields=['vr3','v5','v10','t20','t40','os','lu','p60','p120','amp']
    data=pool[fields].replace([np.inf,-np.inf],np.nan)
    z=((data-data.mean())/(data.std()+1e-12)).clip(-3,3).fillna(0)
    shared=1.375*z.os+1.233*z.lu-.598*z.amp
    pool['stable']=shared+1.695*z.v5+1.687*z.t20+1.205*z.p120
    champ=shared+1.695*z.v10+1.687*z.t40+1.205*z.p120
    wild=shared+1.695*z.vr3+1.687*z.t20+1.205*z.p60
    pool['mean']=(pool.stable+champ+wild)/3
    return pool[(pool.v3|pool['mean'].ge(pool['mean'].quantile(.8)))&pool.p120.notna()]

def scan(p, hs, zz, asof, next_day):
    today=p[p.date.eq(asof)].copy()
    if 'name' not in today:
        today['name']=today.code
    hs=hs.loc[:asof]
    zz=zz.loc[:asof]
    if today.empty or hs.index[-1]!=asof or zz.index[-1]!=asof:
        raise ValueError('缺少当日有效股票或指数收盘')
    out={}
    def section(key, status, note, rows):
        out[key]=dict(title=TITLES[key],status=status,note=note,candidates=rows)

    p2=today[today.p2].sort_values(['amount','code'] if 'amount' in today else ['s20','code'],ascending=[False,True])
    section('P2','待开盘' if len(p2) else '无候选',
            '展示顺序为成交额参考。E开盘确认 −5% < 缺口 ≤ −2%，缺口最深优先，同缺口按代码，正式只取1只。',
            [candidate(x,'待缺口') for x in p2.itertuples(index=False)])

    ew=p[p.code.isin(universe('bf'))&p.date.le(asof)].groupby('date').ret.mean()
    bf_open=bool(dev((1+ew.fillna(0)).cumprod(),20).get(asof,-1)>=0)
    fb_open=bool(dev(zz,20).iloc[-1]<0)
    bf=today[today.bf] if bf_open else today.iloc[:0]
    if len(bf):
        bf=bf[bf.bfscore.eq(bf.bfscore.max())].sort_values(['s20','code'],ascending=[False,True])
    fb=today[today.fb] if fb_open else today.iloc[:0]
    c2=fb[fb.pos.lt(.7)].sort_values('code')
    c1=fb[fb.pos.between(.7,1,inclusive='left')].sort_values('code')
    h1=[candidate(x,'BF A' if x.bfscore==3 else 'BF B',x.bfscore) for x in bf.itertuples(index=False)]
    for label, rows in [('FB C2',c2),('FB C1',c1)]:
        h1.extend(candidate(x,label) for x in rows.itertuples(index=False))
    section('H1','待开盘' if h1 else '无候选',
            f'BF门控{"通过" if bf_open else "关闭"}，FB门控{"通过" if fb_open else "关闭"}。BF先按档位和20日均额，开盘缺口 (−10.1%, +8%]。BF无可执行票才看C2 [−2.5%, +5%]、C1 [−5%, −2%]，同类按缺口升序，合计只取1只。FB盘后按代码展示，次日才定最终名次。', h1)

    ma20=bool(dev(hs,20).iloc[-1]>0)
    ma120=bool(dev(hs,120).iloc[-1]>0)
    pool=trend_scores(today)
    for key, field in [('S2_1','stable'),('S2_2','stable'),('S2_3','mean'),('S2_4','stable')]:
        active=key=='S2_2' or (ma20 and ma120)
        rows=pool.sort_values([field,'code'],ascending=[False,True]).head(5)
        status=('待开盘' if len(rows) else '无候选') if active else '门控关闭'
        section(key,status,
                ('门控关闭，下面仅作观察。' if not active else '')+
                '按收盘分展示Top5，执行口径只看Top1，次日开盘缺口须在 [−5%, +6%]。'+
                ('与趋势1同入口，退出规则不同。' if key=='S2_4' else ''),
                [candidate(x,'可复核' if active else '观察',getattr(x,field)) for x in rows.itertuples(index=False)])

    try:
        if not ma20:
            section('HGB','门控关闭','沪深300收盘未高于MA20，本日无执行候选。',[])
        else:
            frozen=universe('v3')
            coverage=today.code.isin(frozen).sum()
            if coverage<.90*len(frozen):
                raise ValueError(f'冻结股票池覆盖 {coverage}/{len(frozen)}，低于90%')
            f=features_and_labels(p[p.date.le(asof)],hs.loc['2024-02-01':].index,frozen)
            next_month=next_day[:7]
            sessions=hs.index.to_series()
            first_entry=sessions[sessions.str[:7].eq(next_month)]
            first_signal=asof if first_entry.empty else hs.index[hs.index.get_loc(first_entry.iloc[0])-1]
            train=f[f.label_end.lt(first_signal)&f.target.notna()]
            current=f[f.date.eq(asof)].copy()
            if len(train)<10000 or current.empty:
                raise ValueError(f'HGB训练或当日评分样本不足：{len(train)} / {len(current)}')
            model=HistGradientBoostingRegressor(**parameters).fit(train[features],train.target)
            current['score']=model.predict(current[features])
            top=current.sort_values(['score','code'],ascending=[False,True]).head(3)
            top=top[['code','score']].merge(today[['code','name','c']],on='code',validate='one_to_one')
            section('HGB','待开盘',f'沪深300 MA20通过，冻结V3池，逐月训练至 {train.label_end.max()}，分数为预期净收益。E开盘复核可成交性。', [candidate(x,'Top3',x.score) for x in top.itertuples(index=False)])
    except (ValueError,RuntimeError) as error:
        section('HGB','数据异常',str(error),[])
    return out

def publish(asof, next_day, sections, quality):
    folder=BASE/'reports'
    folder.mkdir(exist_ok=True)
    payload=dict(date=asof,next_session=next_day,updated_at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(timespec='seconds'),
                 quality=quality,strategies=sections)
    lines=[f'# {asof} 综合策略候选', '',f'下一交易日 **{next_day}** · 主板日线覆盖 **{quality["coverage"]}** 只。', '',
           '每路展示前5名或前5个观察候选，HGB展示Top3。完整候选见同日期JSON。', '']
    lines[2:2] = BACKTEST_OVERVIEW.splitlines()
    for key,s in sections.items():
        lines.extend([f'## {s["title"]} · {s["status"]}', '', s['note'], ''])
        if s['candidates']:
            lines.extend(['| 展示序 | 代码 | 名称 | 路线 | 收盘价 | 分数 |','|---:|---|---|---|---:|---:|'])
            for i,row in enumerate(s['candidates'][:5],1):
                name=str(row['name']).replace('|','\\|').replace('\n',' ')
                score='—' if row['score'] is None else f'{row["score"]:.6f}'
                lines.append(f'| {i} | {row["code"]} | {name} | {row["route"]} | {row["close"]:.2f} | {score} |')
            lines.append('')
    lines.extend(['---','日线候选供研究观察。实际涨跌停及次日开盘可成交性须在交易时点核验。',''])
    report='\n'.join(lines)
    for stem in (asof,'latest'):
        existing=folder/f'{stem}.json'
        if stem=='latest' and existing.exists() and json.loads(existing.read_text())['date']>asof:
            continue
        (folder/f'{stem}.md').write_text(report,encoding='utf-8')
        existing.write_text(json.dumps(payload,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--date',default=datetime.now(ZoneInfo('Asia/Shanghai')).date().isoformat())
    parser.add_argument('--retry',action='store_true')
    args=parser.parse_args()
    path=BASE/'reports'/f'{args.date}.json'
    if args.retry and path.exists():
        previous=json.loads(path.read_text())
        if set(previous.get('strategies',{}))==set(TITLES) and all(s['status']!='数据异常' for s in previous['strategies'].values()) and previous.get('quality',{}).get('source')=='daily':
            print(f'{args.date} 已有完整日更报告')
            return
    next_day, names=exchange(args.date)
    if not next_day:
        print(f'{args.date} 休市')
        return
    meta, errors=refresh(args.date,names)
    hs,zz=(index(code).loc[:args.date] for code in ('sh000300','sh000852'))
    p=build(hs.loc['2024-02-01':].index,meta)
    result=scan(p,hs,zz,args.date,next_day)
    publish(args.date,next_day,result,dict(coverage=int(p.date.eq(args.date).sum()),missing=len(errors),source='daily'))
    if any(s['status']=='数据异常' for s in result.values()):
        raise SystemExit('报告已生成，部分策略数据异常')

if __name__=='__main__':
    main()
