"""Refresh Tencent bars and verify the session against the exchange calendar."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timedelta
from pathlib import Path
from threading import Lock
from time import monotonic, sleep
from urllib.parse import urlencode
from urllib.request import Request, urlopen
import gzip
import json
import re

import pandas as pd

from features import ROOT, index

MAIN = re.compile(r'^(?:600|601|603|605|000|001|002|003)\d{3}$')
HOSTS = ('https://web.ifzq.gtimg.cn', 'https://web.sqt.gtimg.cn')
lock = Lock()
next_request = 0.0

def exchange(asof):
    import baostock as bs

    login = bs.login()
    if login.error_code != '0':
        raise RuntimeError(f'交易日历登录失败: {login.error_msg}')
    try:
        end = (date.fromisoformat(asof)+timedelta(days=20)).isoformat()
        rows = bs.query_trade_dates(start_date=asof, end_date=end)
        if rows.error_code != '0':
            raise RuntimeError(rows.error_msg)
        sessions = rows.get_data()
        today = sessions[sessions.calendar_date.eq(asof)]
        if today.empty:
            raise RuntimeError(f'交易日历缺少 {asof}')
        if today.iloc[0].is_trading_day != '1':
            return None, None
        future = sessions[sessions.calendar_date.gt(asof)&sessions.is_trading_day.eq('1')]
        if future.empty:
            raise RuntimeError('交易日历缺少下一交易日')
        stocks = bs.query_all_stock(day=asof)
        if stocks.error_code != '0':
            raise RuntimeError(stocks.error_msg)
        data = stocks.get_data()
        data['code'] = data.code.str.split('.').str[-1]
        data = data[data.code.str.fullmatch(MAIN)&data.tradeStatus.eq('1')]
        if len(data)<2500:
            raise RuntimeError(f'主板交易股票池异常: {len(data)}')
        return future.iloc[0].calendar_date, data.set_index('code').code_name.to_dict()
    finally:
        bs.logout()

def fetch(endpoint, params):
    global next_request
    for attempt in range(2):
        with lock:
            sleep(max(0,next_request-monotonic()))
            next_request=monotonic()+.10
        try:
            url=f'{HOSTS[attempt]}/appstock/app/{endpoint}/get?'+urlencode({'param':params})
            with urlopen(Request(url,headers={'User-Agent':'Mozilla/5.0'}),timeout=9) as response:
                result=json.load(response)
            if not result.get('data'):
                raise ValueError('行情返回为空')
            return result
        except (OSError,ValueError):
            if attempt==1:
                raise

def refresh(asof, names):
    target=ROOT/'data'/'stock'
    target.mkdir(parents=True,exist_ok=True)
    for sym in ('sh000300','sh000852'):
        result=fetch('fqkline',f'{sym},day,,{asof},680,qfq')
        (ROOT/'data'/f'{sym}.json').write_text(json.dumps(result),encoding='utf-8')
        if asof not in index(sym).index:
            raise RuntimeError(f'{sym} 未更新到 {asof}')
    since=(date.fromisoformat(asof)-timedelta(days=980)).isoformat()
    def stock(code,name):
        symbol=('sh' if code.startswith('6') else 'sz')+code
        output={'code':code,'current_name':name}
        for key,adj in [('raw',''),('qfq','qfq')]:
            payload=fetch('newfqkline',f'{symbol},day,{since},{asof},640,{adj}')['data'][symbol]
            bars=(payload.get('qfqday') if adj else payload.get('day')) or payload.get('day') or []
            output[key]=[r for r in bars if r[0]<=asof and len(r)>=(6 if adj else 9)]
            if not output[key] or output[key][-1][0]!=asof:
                raise ValueError(f'{key} 缺少当日日线')
        with gzip.open(target/f'{code}.json.gz','wt',encoding='utf-8') as f:
            json.dump(output,f,ensure_ascii=False)
        return output['raw'][0][0]
    errors={}
    ipo={}
    with ThreadPoolExecutor(max_workers=24) as executor:
        tasks={executor.submit(stock,c,name):c for c,name in names.items()}
        for task in as_completed(tasks):
            code=tasks[task]
            try:
                ipo[code]=task.result()
            except (OSError,ValueError,KeyError,TypeError) as error:
                errors[code]=str(error)
    print(f'行情覆盖 {len(ipo)}/{len(names)}，缺失 {len(errors)}',flush=True)
    if len(ipo)<len(names)*.95:
        raise RuntimeError(f'当日行情覆盖不足 95%，缺失 {len(errors)} 只')
    return pd.DataFrame({'ipoDate':ipo}), errors
