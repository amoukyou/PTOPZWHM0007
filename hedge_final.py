"""
葡萄牙基金对冲方案 — 完整报告 v18
v18: 审稿修正 — 标题动态化/24张推导补全/覆盖率描述修正/skew警告/滚仓成本量化
v17: 条件Beta实证分析 + A/B方案切换 + 跌回买入点场景
v16: 删除Collar方案，精简为两个纯Put方案(混合+纯OTM)
v15: 混合行权价策略(ATM×8+OTM×20)，Event #11检测
v10: 删除循环论证覆盖率，动态滚仓触发，Beta链独立回归说明
"""

import os, sys, math
import numpy as np
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go

sys.stdout.reconfigure(encoding='utf-8')
DATA_DIR = os.path.dirname(os.path.abspath(__file__))

# ═══ 固定参数 ══════════════════════════════════
INITIAL_INV      = 507_000
FUND_ENTRY_PRICE = 13.3534
FUND_UNITS       = round(INITIAL_INV / FUND_ENTRY_PRICE)
PSI20_ENTRY_ACT  = 6_711
BETA_FUND_IBEX   = 0.4230
BETA_FUND_PSI    = 0.6271
BETA_IBEX_PSI    = 0.6897
IBEX_IMPLIED_VOL = 0.185
ECB_RATE         = 0.026
N_CONTRACTS      = 16          # Mini IBEX, 乘数 €1/点
ENTRY_DATE       = '2024-07-16'
# ═══════════════════════════════════════════════

def fetch_live_prices():
    """获取基金NAV(FT Markets) + 三大指数(yfinance)"""
    import re, subprocess
    prices = {}
    # 1) 基金NAV从FT Markets抓取（比Yahoo更新更快）
    try:
        r = subprocess.run(['curl', '-s', '-H', 'User-Agent: Mozilla/5.0',
            'https://markets.ft.com/data/funds/tearsheet/summary?s=PTOPZWHM0007:EUR'],
            capture_output=True, text=True, timeout=15)
        html = r.stdout
        # Price: <span class="mod-ui-data-list__value">16.97</span>
        m = re.search(r'Price \(EUR\)</span><span class="mod-ui-data-list__value">([0-9.]+)', html)
        if m:
            prices['fund'] = float(m.group(1))
        # Date: "as of Mar 03 2026"
        m2 = re.search(r'as of ([A-Z][a-z]{2} \d{2} \d{4})', html)
        if m2:
            from datetime import datetime as _dt
            prices['fund_date'] = _dt.strptime(m2.group(1), '%b %d %Y').strftime('%Y-%m-%d')
        prices['fund_src'] = 'FT Markets'
    except Exception:
        pass
    # 2) 三大指数从yfinance
    for key, sym in [('psi','PSI20.LS'), ('ibex','^IBEX'), ('estx','^STOXX50E')]:
        try:
            t = yf.Ticker(sym)
            h = t.history(period='5d')
            if len(h) > 0:
                prices[key] = float(h['Close'].iloc[-1])
                prices[key+'_date'] = h.index[-1].strftime('%Y-%m-%d')
        except Exception:
            pass
    # 3) Fallback: 如果FT没抓到，用Yahoo
    if 'fund' not in prices:
        try:
            t = yf.Ticker('0P0001O8MU.F')
            h = t.history(period='5d')
            if len(h) > 0:
                prices['fund'] = float(h['Close'].iloc[-1])
                prices['fund_date'] = h.index[-1].strftime('%Y-%m-%d')
                prices['fund_src'] = 'Yahoo Finance'
        except Exception:
            pass
    return prices

def fetch_meff_iv():
    """从MEFF每日公报抓取Mini IBEX Put期权的实盘IV数据"""
    import re, subprocess
    from datetime import datetime, timedelta
    day_names = ['mon', 'tue', 'wed', 'thu', 'fri', 'sat', 'sun']
    # 从今天开始逐日回退，最多试5个工作日（处理假期、未发布等情况）
    for days_back in range(7):
        try:
            d = datetime.now() - timedelta(days=days_back)
            wd = d.weekday()
            if wd >= 5:  # 跳过周末
                continue
            day_suffix = day_names[wd]
            url = f'https://www.meff.es/docs/Ficheros/boletin/ing/boletiip{day_suffix}.htm'
            r = subprocess.run(['curl', '-sL', '-H', 'User-Agent: Mozilla/5.0', url],
                capture_output=True, text=True, timeout=20)
            html = r.stdout
            idx = html.find('OPTIONS (PUT)')
            if idx < 0:
                continue  # 这天没数据，试前一天
            chunk = html[idx:idx+80000]
            lines = re.sub(r'<[^>]+>', '|', chunk)
            lines = re.sub(r'\|+', '|', lines).split('|')
            data = {}
            i = 0
            while i < len(lines):
                s = lines[i].strip()
                m = re.match(r'((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)-\d{2})(?:\s+w\d)?\s*(?:&nbsp;)*\s*([\d,]+(?:\.\d+)?)', s)
                if m:
                    exp = m.group(1)
                    strike_s = m.group(2).replace(',', '')
                    strike = float(strike_s)
                    if strike < 100:
                        i += 1
                        continue
                    fields = []
                    j = i + 1
                    while j < len(lines) and len(fields) < 8:
                        f = lines[j].strip()
                        if f and f != '&nbsp;':
                            fields.append(f)
                        j += 1
                    close = float(fields[0].replace(',', '')) if fields and fields[0] != '-' else None
                    iv = float(fields[4]) / 100 if len(fields) > 4 and fields[4] != '-' else None
                    delta = float(fields[5]) if len(fields) > 5 and fields[5] != '-' else None
                    oi = int(fields[7].replace(',', '')) if len(fields) > 7 and fields[7] != '-' else 0
                    if close and iv and strike > 5000:
                        data.setdefault(exp, []).append(dict(strike=strike, close=close, iv=iv, delta=delta, oi=oi))
                i += 1
            if data:
                if days_back > 0:
                    print(f'  MEFF: 今日公报无数据，使用{d.strftime("%A")}({d.strftime("%m-%d")})的公报')
                return data
        except Exception:
            continue
    return None

def pick_expiry(meff_data):
    """选择最接近1年的到期月"""
    if not meff_data:
        return None, []
    from datetime import datetime
    now = datetime.now()
    best_exp, best_diff, best_points = None, 999, []
    for exp_str, points in meff_data.items():
        if len(points) < 2:
            continue
        try:
            exp_date = datetime.strptime('15-' + exp_str, '%d-%b-%y')
            diff = abs((exp_date - now).days - 365)
            if diff < best_diff:
                best_exp, best_diff, best_points = exp_str, diff, points
        except Exception:
            continue
    return best_exp, sorted(best_points, key=lambda p: p['strike'])

def interp_iv(strike, iv_points, fallback=IBEX_IMPLIED_VOL):
    """根据MEFF实盘数据插值IV。线性插值，边界外线性外推（带上下限）"""
    if not iv_points or len(iv_points) < 2:
        return fallback
    pts = sorted(iv_points, key=lambda p: p['strike'])
    if strike <= pts[0]['strike']:
        # 左侧外推（更深OTM → IV更高）
        if len(pts) >= 2:
            slope = (pts[1]['iv'] - pts[0]['iv']) / (pts[1]['strike'] - pts[0]['strike'])
            iv = pts[0]['iv'] + slope * (strike - pts[0]['strike'])
            return max(min(iv, 0.50), 0.10)  # 限制在10%-50%
        return pts[0]['iv']
    if strike >= pts[-1]['strike']:
        # 右侧外推（更深ITM → IV更低）
        if len(pts) >= 2:
            slope = (pts[-1]['iv'] - pts[-2]['iv']) / (pts[-1]['strike'] - pts[-2]['strike'])
            iv = pts[-1]['iv'] + slope * (strike - pts[-1]['strike'])
            return max(min(iv, 0.50), 0.08)
        return pts[-1]['iv']
    # 线性插值
    for j in range(len(pts) - 1):
        if pts[j]['strike'] <= strike <= pts[j+1]['strike']:
            t = (strike - pts[j]['strike']) / (pts[j+1]['strike'] - pts[j]['strike'])
            return pts[j]['iv'] * (1 - t) + pts[j+1]['iv'] * t
    return fallback

def fetch_meff_live_chain(ibex_now):
    """从MEFF官网抓取Mini IBEX Put期权链实时bid/ask报价（15分钟延迟）"""
    import re, subprocess, urllib.parse
    url = 'https://www.meff.es/ing/Financial-Derivatives/Spreadsheet/FIEM_MiniIbex_35'
    try:
        # Step 1: GET获取VIEWSTATE
        r = subprocess.run(['curl', '-sL', '-c', '/tmp/meff_cookies.txt',
            '-H', 'User-Agent: Mozilla/5.0', url],
            capture_output=True, text=True, timeout=20)
        vs = re.search(r'__VIEWSTATE" value="([^"]*)"', r.stdout)
        vsg = re.search(r'__VIEWSTATEGENERATOR" value="([^"]*)"', r.stdout)
        if not vs:
            return {}
        # Step 2: POST with full VIEWSTATE to get all expiration data
        post_data = urllib.parse.urlencode({
            '__VIEWSTATE': vs.group(1),
            '__VIEWSTATEGENERATOR': vsg.group(1) if vsg else '',
            'OpStyle': 'E', 'OpType': 'P', 'OpStrike': 'OPE20260918'
        })
        r2 = subprocess.run(['curl', '-sL', '-b', '/tmp/meff_cookies.txt',
            '-H', 'User-Agent: Mozilla/5.0',
            '-H', 'Content-Type: application/x-www-form-urlencoded',
            '-d', post_data, url], capture_output=True, text=True, timeout=20)
        html = r2.stdout
        # 提取可用到期月
        avail_exps = re.findall(r'value="(OPE\d+)"[^>]*>([^<]+)<', html)
        # 解析所有行 (data-tipo标识到期月)
        all_rows = re.findall(r'<tr[^>]*data-tipo="(OPE\d+)"[^>]*>(.*?)</tr>', html, re.DOTALL)
        result = {}  # {exp_code: [rows...]}
        for tipo, data in all_rows:
            cells = re.findall(r'<td[^>]*>(.*?)</td>', data)
            if len(cells) < 13:
                continue
            def cl(c):
                c = c.replace('&#160;','').replace('&nbsp;','').replace(',','').strip()
                return None if c.startswith('-') or c=='-' or not c else c
            try:
                strike = float(cl(cells[0]).replace(',','')) if cl(cells[0]) else None
                if not strike or strike < 5000:
                    continue
                bid = float(cl(cells[3])) if cl(cells[3]) else None
                ask = float(cl(cells[4])) if cl(cells[4]) else None
                last = float(cl(cells[7])) if cl(cells[7]) else None
                vol = int(float(cl(cells[8]))) if cl(cells[8]) else 0
                prev = float(cl(cells[12])) if cl(cells[12]) else None
                result.setdefault(tipo, []).append(dict(
                    strike=strike, bid=bid, ask=ask, last=last, vol=vol, prev=prev,
                    otm_pct=round((1 - strike/ibex_now) * 100, 1) if ibex_now else None
                ))
            except (ValueError, TypeError):
                continue
        # 排序
        for k in result:
            result[k] = sorted(result[k], key=lambda x: x['strike'])
        # 附上到期月标签
        exp_labels = {code: label for code, label in avail_exps}
        return {'chains': result, 'labels': exp_labels}
    except Exception as e:
        print(f'  MEFF实时期权链抓取失败: {e}')
        return {}

# 全局MEFF数据（运行时填充）
_meff_iv_points = []
_meff_expiry = None
_meff_source = 'BS'  # 'MEFF' or 'BS'
_meff_live_chain = {}  # 实时期权链数据

def norm_cdf(x):
    return (1 + math.erf(x / math.sqrt(2))) / 2

def bs_put(S, K, T, r=ECB_RATE, sigma=None):
    if sigma is None:
        sigma = interp_iv(K, _meff_iv_points, IBEX_IMPLIED_VOL)
    if T <= 0: return max(K - S, 0)
    d1 = (math.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*math.sqrt(T))
    d2 = d1 - sigma*math.sqrt(T)
    return K*math.exp(-r*T)*norm_cdf(-d2) - S*norm_cdf(-d1)


def update_csv_with_live(live):
    """将 live NAV 数据追加到 CSV 文件，避免数据缺口"""
    if not live or 'fund_nav' not in live or 'fund_date' not in live:
        return
    # 只有真正从数据源抓到的NAV才写CSV，fallback值不写（避免污染数据）
    if live.get('fund_date') == '?' or live.get('_fund_is_fallback'):
        return
    csv_path = os.path.join(DATA_DIR, 'PTOPZWHM0007_daily_2022-2026.csv')
    fund = pd.read_csv(csv_path, parse_dates=['Date'])
    live_dt = pd.Timestamp(live['fund_date'])
    # 只追加CSV中还没有的日期
    if live_dt not in fund['Date'].values:
        nav = live['fund_nav']
        new_row = pd.DataFrame({'Date': [live_dt], 'Open': [nav], 'High': [nav],
                                'Low': [nav], 'Close': [nav], 'Volume': [0]})
        fund = pd.concat([fund, new_row], ignore_index=True)
        fund = fund.sort_values('Date').reset_index(drop=True)
        fund.to_csv(csv_path, index=False)
        print(f"  CSV已追加 {live['fund_date']} NAV={nav}")
    # 同步更新 2024-2026 版本
    csv2 = os.path.join(DATA_DIR, 'PTOPZWHM0007_daily_2024-2026.csv')
    if os.path.exists(csv2):
        f2 = pd.read_csv(csv2, parse_dates=['Date'])
        if live_dt not in f2['Date'].values and live_dt >= pd.Timestamp('2024-01-01'):
            nav = live['fund_nav']
            new_row = pd.DataFrame({'Date': [live_dt], 'Open': [nav], 'High': [nav],
                                    'Low': [nav], 'Close': [nav], 'Volume': [0]})
            f2 = pd.concat([f2, new_row], ignore_index=True)
            f2 = f2.sort_values('Date').reset_index(drop=True)
            f2.to_csv(csv2, index=False)

def load_data(live=None):
    fund = pd.read_csv(os.path.join(DATA_DIR, 'PTOPZWHM0007_daily_2022-2026.csv'), parse_dates=['Date'])
    fund = fund.sort_values('Date').reset_index(drop=True)
    from datetime import datetime, timedelta
    end_date = (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')
    ibex = yf.download('^IBEX', start='2022-01-01', end=end_date, progress=False)
    ibex.columns = [c[0] for c in ibex.columns]
    ibex = ibex.reset_index()[['Date','Close']].rename(columns={'Close':'ibex'})
    ibex['Date'] = ibex['Date'].dt.normalize()
    psi = yf.download('PSI20.LS', start='2022-01-01', end=end_date, progress=False)
    psi.columns = [c[0] for c in psi.columns]
    psi = psi.reset_index()[['Date','Close']].rename(columns={'Close':'psi'})
    psi['Date'] = psi['Date'].dt.normalize()
    # 追加live数据到fund_df（内存中），确保事件检测覆盖最新交易日
    if live and 'fund_nav' in live and 'fund_date' in live:
        live_dt = pd.Timestamp(live['fund_date'])
        if fund['Date'].iloc[-1] < live_dt:
            live_row = pd.DataFrame({'Date': [live_dt], 'Close': [live['fund_nav']]})
            for col in fund.columns:
                if col not in live_row.columns:
                    live_row[col] = np.nan
            fund = pd.concat([fund, live_row], ignore_index=True)
    return fund, ibex, psi

def find_weekly_drops(df, threshold=-3.0):
    df = df.copy()
    df['fund_5d'] = df['fund'].pct_change(5) * 100
    drops = df[df['fund_5d'] < threshold]
    events = []
    prev_end = None
    for _, row in drops.iterrows():
        d = row['Date']
        if prev_end is not None and (d - prev_end).days < 10:
            if row['fund_5d'] < events[-1]['fund_chg']:
                events[-1]['fund_chg'] = row['fund_5d']
                events[-1]['worst_date'] = d
            events[-1]['end'] = d
        else:
            idx = df.index[df['Date'] == d][0]
            events.append(dict(start=df['Date'].iloc[max(0,idx-5)], end=d, worst_date=d, fund_chg=row['fund_5d']))
        prev_end = d
    for ev in events:
        idx_s = df.index[df['Date'] == ev['start']][0]
        idx_e = df.index[df['Date'] == ev['end']][0]
        lo, hi = max(0, idx_s-5), min(len(df)-1, idx_e+10)
        ibex_peak = df['ibex'].iloc[lo:hi+1].max()
        ibex_trough = df['ibex'].iloc[lo:hi+1].min()
        ev.update(
            ibex_chg=(ibex_trough/ibex_peak-1)*100, ibex_level=ibex_trough, ibex_peak=ibex_peak,
            sync=((ibex_trough/ibex_peak-1)*100) < -1.5,
            start_str=ev['start'].strftime('%Y-%m-%d'), end_str=ev['worst_date'].strftime('%Y-%m-%d'),
            in_hold=ev['worst_date'] >= pd.Timestamp(ENTRY_DATE), window_lo=lo, window_hi=hi,
        )
    return events

def simulate_strategy(df_h, months):
    roll_days, T = months*21, months/12
    positions, total_prem, i = [], 0, 0
    while i < len(df_h):
        ib = df_h['ibex'].iloc[i]
        K = ib  # ATM
        prem = bs_put(ib, K, T) * N_CONTRACTS
        total_prem += prem
        exp = min(i + roll_days, len(df_h)-1)
        positions.append(dict(buy_idx=i, strike=K, expiry_idx=exp, premium=prem))
        i = exp + 1
    return dict(positions=positions, total_premium=total_prem, months=months)

def get_put_mtm(positions, ibex_val, day_idx):
    for pos in positions:
        if pos['buy_idx'] <= day_idx <= pos['expiry_idx']:
            rem = max((pos['expiry_idx'] - day_idx)/252, 0.001)
            return bs_put(ibex_val, pos['strike'], rem) * N_CONTRACTS, pos['strike']
    return 0, 0

def analyze(fund_df, ibex_df, psi_df, live):
    fv = live['fund_value']
    ibex_now = live['ibex']
    psi_now = live['psi']
    # 使用MEFF实际最长Put到期期限，没有实时数据时fallback到1年
    T_put = live.get('best_put_T', 1.0)

    df = pd.merge(fund_df[['Date','Close']].rename(columns={'Close':'fund'}),
                  ibex_df, on='Date', how='inner').sort_values('Date').reset_index(drop=True)
    events = find_weekly_drops(df)
    df_h = df[df['Date'] >= ENTRY_DATE].reset_index(drop=True)
    strats = {}
    for key, months in [('3M',3),('6M',6),('12M',12)]:
        strats[key] = simulate_strategy(df_h, months)
    # Per-event strategy results (holding period only)
    for ev in events:
        ev['strat'] = {}
        if not ev['in_hold']: continue
        loss = abs(fv * ev['fund_chg'] / 100)
        ev['fund_loss'] = loss
        idx = (df_h['ibex'] - ev['ibex_level']).abs().idxmin()
        for k, s in strats.items():
            mtm, strike = get_put_mtm(s['positions'], ev['ibex_level'], idx)
            ev['strat'][k] = dict(mtm=mtm, strike=strike, cov=mtm/loss*100 if loss>0 else 0)
    # Empirical crash ratio: fund_drop / ibex_drop during each event
    crash_ratios = []
    for ev in events:
        if ev['ibex_chg'] < -1.5 and ev['fund_chg'] < -1.5:
            ratio = ev['fund_chg'] / ev['ibex_chg']
            ev['crash_ratio'] = round(ratio, 3)
            crash_ratios.append(ratio)
        else:
            ev['crash_ratio'] = None
    avg_crash_ratio = round(np.mean(crash_ratios), 3) if crash_ratios else BETA_FUND_IBEX
    # Recommendation (base: 16 contracts)
    K = round(ibex_now / 50) * 50  # MEFF标准行权价间距50点
    p1 = bs_put(ibex_now, K, T_put)
    rec = dict(K=K, price=round(p1,1), total=round(p1*N_CONTRACTS), annual=round(p1*N_CONTRACTS/fv*100,2), T=T_put)
    # Mixed-strike configurations
    K_90 = round(ibex_now * 0.90 / 50) * 50
    K_85 = round(ibex_now * 0.85 / 50) * 50
    p_90 = bs_put(ibex_now, K_90, T_put)
    p_85 = bs_put(ibex_now, K_85, T_put)
    options = []
    configs = [
        (t('纯ATM ×16','Pure ATM ×16'), t('现方案：全部平值','Current: all ATM'), [(16, K, p1)]),
        (f'ATM ×8 + 90%OTM ×20', t('混合：小跌有底+大跌加倍','Mixed: floor for small drops + doubled crash protection'), [(8, K, p1), (20, K_90, p_90)]),
        (f'ATM ×4 + 90%OTM ×30', t('进取：重注大跌保护','Aggressive: heavy crash protection'), [(4, K, p1), (30, K_90, p_90)]),
        (t('纯90%OTM ×24','Pure 90%OTM ×24'), t('省钱：放弃小跌，只防崩盘','Budget: skip small drops, crash-only'), [(24, K_90, p_90)]),
    ]
    for label, desc, legs in configs:
        prem = round(sum(p * n for n, k, p in legs))
        scenarios = {}
        for drop_pct in [5, 10, 15, 20, 30]:
            ibex_drop = ibex_now * (1 - drop_pct/100)
            put_payoff = sum(max(k - ibex_drop, 0) * n for n, k, p in legs)
            scenarios[drop_pct] = dict(payoff=round(put_payoff))
        options.append(dict(label=label, desc=desc, legs=legs, prem=prem,
                            annual_pct=round(prem/fv*100, 2), five_yr=prem*5, scenarios=scenarios))
    # PSI20 scenario table for section 六 — Plan A: recommended mixed config
    rec_legs = configs[1][2]  # ATM×8 + 90%OTM×20
    rec_prem = round(sum(p * n for n, k, p in rec_legs))
    psi_scenarios = []
    for psi_target in [8500, 8000, 7500, 7000, 6000]:
        psi_drop_pct = (psi_target - psi_now) / psi_now  # negative
        ibex_est = ibex_now * (1 + BETA_IBEX_PSI * psi_drop_pct)
        fund_est = fv * (1 + BETA_FUND_PSI * psi_drop_pct)
        fund_loss = fv - fund_est
        put_pay = sum(max(k - ibex_est, 0) * n for n, k, p in rec_legs)
        net = fund_est + put_pay - rec_prem
        psi_scenarios.append(dict(
            psi=psi_target, fund_est=round(fund_est), fund_loss=round(fund_loss),
            put_pay=round(put_pay), net=round(net), ibex_est=round(ibex_est),
            cov=round(put_pay/fund_loss*100) if fund_loss > 0 else 0))
    # Plan B: pure 90%OTM×24
    planb_legs = configs[3][2]  # 纯90%OTM ×24
    planb_prem = round(sum(p * n for n, k, p in planb_legs))
    psi_scenarios_b = []
    for psi_target in [8500, 8000, 7500, 7000, 6000]:
        psi_drop_pct = (psi_target - psi_now) / psi_now
        ibex_est = ibex_now * (1 + BETA_IBEX_PSI * psi_drop_pct)
        fund_est = fv * (1 + BETA_FUND_PSI * psi_drop_pct)
        fund_loss = fv - fund_est
        put_pay = sum(max(k - ibex_est, 0) * n for n, k, p in planb_legs)
        net = fund_est + put_pay - planb_prem
        psi_scenarios_b.append(dict(
            psi=psi_target, fund_est=round(fund_est), fund_loss=round(fund_loss),
            put_pay=round(put_pay), net=round(net), ibex_est=round(ibex_est),
            cov=round(put_pay/fund_loss*100) if fund_loss > 0 else 0))
    # "跌回买入点"场景：PSI20回到入场时水平
    psi_drop_entry = (PSI20_ENTRY_ACT - psi_now) / psi_now
    ibex_entry_est = ibex_now * (1 + BETA_IBEX_PSI * psi_drop_entry)
    fund_entry_est = fv * (1 + BETA_FUND_PSI * psi_drop_entry)
    fund_entry_loss = fv - fund_entry_est
    gain_now = fv - INITIAL_INV  # 当前浮盈
    # Plan A payoff
    pa_pay = sum(max(k - ibex_entry_est, 0) * n for n, k, p in rec_legs)
    pa_net = fund_entry_est + pa_pay - rec_prem
    # Plan B payoff
    pb_pay = sum(max(k - ibex_entry_est, 0) * n for n, k, p in planb_legs)
    pb_net = fund_entry_est + pb_pay - planb_prem
    entry_scenario = dict(
        psi_target=PSI20_ENTRY_ACT, psi_drop_pct=round(psi_drop_entry*100, 1),
        ibex_est=round(ibex_entry_est), fund_est=round(fund_entry_est),
        fund_loss=round(fund_entry_loss), gain_now=gain_now,
        pa_pay=round(pa_pay), pa_net=round(pa_net),
        pa_kept=round((pa_net - INITIAL_INV) / gain_now * 100) if gain_now > 0 else 0,
        pb_pay=round(pb_pay), pb_net=round(pb_net),
        pb_kept=round((pb_net - INITIAL_INV) / gain_now * 100) if gain_now > 0 else 0,
        no_hedge_kept=round((fund_entry_est - INITIAL_INV) / gain_now * 100) if gain_now > 0 else 0,
    )
    return dict(df=df, df_h=df_h, events=events, strats=strats, rec=rec,
                options=options, psi_scenarios=psi_scenarios,
                psi_scenarios_b=psi_scenarios_b,
                K_90=K_90, rec_prem=rec_prem, planb_prem=planb_prem,
                avg_crash_ratio=avg_crash_ratio, crash_ratios=crash_ratios,
                entry_scenario=entry_scenario)

# ─── Charts ──────────────────────────────
def chart_fund_psi_ibex(fund_df, psi_df, ibex_df, live):
    e = pd.Timestamp(ENTRY_DATE)
    m = pd.merge(fund_df[fund_df['Date']>=e][['Date','Close']], psi_df[psi_df['Date']>=e], on='Date', how='inner')
    m = pd.merge(m, ibex_df[ibex_df['Date']>=e], on='Date', how='inner')
    rf, rp, ri = m['Close'].iloc[0], m['psi'].iloc[0], m['ibex'].iloc[0]
    # Append live prices as latest data point (may be newer than CSV)
    live_date = pd.Timestamp(live.get('fund_date', live.get('psi_date', m['Date'].iloc[-1])))
    if live_date > m['Date'].iloc[-1]:
        live_row = pd.DataFrame([{'Date': live_date, 'Close': live['fund_nav'], 'psi': live['psi'], 'ibex': live['ibex']}])
        m = pd.concat([m, live_row], ignore_index=True)
    fund_pct = (m['Close']/rf-1)*100
    psi_pct = (m['psi']/rp-1)*100
    ibex_pct = (m['ibex']/ri-1)*100
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=m['Date'], y=fund_pct, name='基金', line=dict(color='#1565c0',width=2.5)))
    fig.add_trace(go.Scatter(x=m['Date'], y=psi_pct, name='PSI20', line=dict(color='#ff7f0e',width=2,dash='dot')))
    fig.add_trace(go.Scatter(x=m['Date'], y=ibex_pct, name='IBEX35', line=dict(color='#e65100',width=2,dash='dash')))
    fig.add_hline(y=0, line_dash='dot', line_color='gray', opacity=0.3)
    # Endpoint annotations with live prices (use actual computed %)
    last_date = m['Date'].iloc[-1]
    for val, color, price_str in [
        (fund_pct.iloc[-1], '#1565c0', f'NAV €{live["fund_nav"]:.2f}'),
        (psi_pct.iloc[-1], '#ff7f0e', f'{live["psi"]:,.0f}'),
        (ibex_pct.iloc[-1], '#e65100', f'{live["ibex"]:,.0f}'),
    ]:
        fig.add_annotation(x=last_date, y=val, xanchor='left', text=f' {price_str} ({val:+.1f}%)',
            showarrow=False, font=dict(size=10, color=color, weight='bold'),
            xshift=5, bgcolor='rgba(255,255,255,0.85)', borderpad=2)
    fig.update_layout(template='plotly_white', height=340, yaxis_title='相对买入日涨跌(%)',
        legend=dict(x=0.01,y=0.99), margin=dict(t=10,b=30,l=60,r=100), hovermode='x unified')
    return fig.to_json()

def chart_fund_ibex(df, events):
    rf, ri = df['fund'].iloc[0], df['ibex'].iloc[0]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['Date'], y=(df['fund']/rf-1)*100, name='基金', line=dict(color='#1565c0',width=2.5)))
    fig.add_trace(go.Scatter(x=df['Date'], y=(df['ibex']/ri-1)*100, name='IBEX35', line=dict(color='#e65100',width=2.5)))
    fig.add_hline(y=0, line_dash='dot', line_color='gray', opacity=0.3)
    for i, ev in enumerate(events):
        fig.add_vrect(x0=ev['start'], x1=ev['end'], fillcolor='#c8e6c9', opacity=0.2)
        fig.add_annotation(x=ev['worst_date'], y=1.0, yref='paper', text=f'#{i+1}', showarrow=False,
            font=dict(size=9, color='#2e7d32'), bgcolor='#e8f5e9', bordercolor='#2e7d32', borderwidth=1)
    fig.add_vline(x=ENTRY_DATE, line_dash='dot', line_color='#c62828', opacity=0.5)
    fig.add_annotation(x=ENTRY_DATE, y=0.02, yref='paper', text='买入日', showarrow=False,
        font=dict(size=10, color='#c62828'))
    fig.update_layout(template='plotly_white', height=460, yaxis_title='相对2022年初涨跌(%)',
        legend=dict(x=0.01,y=0.99), margin=dict(t=10,b=30,l=60,r=20), hovermode='x unified')
    return fig.to_json()

def make_zoom_charts(df, events, strats, fv):
    """为每个事件生成±20天放大图"""
    charts = []
    df_h = df[df['Date'] >= ENTRY_DATE].reset_index(drop=True)
    for ev in events:
        # use full df for pre-hold events, df_h for hold events
        src = df_h if ev['in_hold'] else df
        center_mask = src['Date'] == ev['worst_date']
        if center_mask.sum() == 0:
            charts.append(None); continue
        ci = src.index[center_mask][0]
        lo, hi = max(0, ci-20), min(len(src)-1, ci+20)
        w = src.iloc[lo:hi+1]
        rf, ri = w['fund'].iloc[0], w['ibex'].iloc[0]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=w['Date'],y=(w['fund']/rf-1)*100,name='基金',
            line=dict(color='#1565c0',width=2.5),hovertemplate='%{x|%m-%d} 基金:%{y:+.1f}%<extra></extra>'))
        fig.add_trace(go.Scatter(x=w['Date'],y=(w['ibex']/ri-1)*100,name='IBEX35',
            line=dict(color='#e65100',width=2.5),hovertemplate='%{x|%m-%d} IBEX:%{y:+.1f}%<extra></extra>'))
        # 12M ATM portfolio for holding-period events
        if ev['in_hold']:
            s12 = strats['12M']
            pv = []
            for t in range(lo, hi+1):
                mtm, _ = get_put_mtm(s12['positions'], df_h['ibex'].iloc[t], t)
                pv.append((df_h['fund'].iloc[t]/rf-1)*100 + mtm/fv*100)
            fig.add_trace(go.Scatter(x=w['Date'],y=pv,name='基金+Put',
                line=dict(color='#2e7d32',width=3),hovertemplate='%{x|%m-%d} 组合:%{y:+.1f}%<extra></extra>'))
        fig.add_hline(y=0,line_dash='dot',line_color='gray',opacity=0.3)
        fig.add_vline(x=ev['worst_date'],line_dash='dash',line_color='#c62828',opacity=0.5)
        fig.update_layout(template='plotly_white',height=260,yaxis_title='涨跌(%)',
            legend=dict(x=0.01,y=0.99,font=dict(size=10)),margin=dict(t=5,b=25,l=50,r=15),hovermode='x unified')
        charts.append(fig.to_json())
    return charts

def chart_payoff(rec, live):
    """损益图：x轴=PSI20点位, y轴=基金市值(EUR), 显示混合配置"""
    fv, psi_now, ibex_now = live['fund_value'], live['psi'], live['ibex']
    T_put = live.get('best_put_T', 1.0)
    K_atm = rec['K']
    K_90 = round(ibex_now * 0.90 / 50) * 50
    p1 = bs_put(ibex_now, K_atm, T_put)
    p_90 = bs_put(ibex_now, K_90, T_put)
    # 推荐混合配置
    mix_prem = round(p1*8 + p_90*20)
    atm_prem = round(p1*16)
    psi_x = np.linspace(5000, 11000, 500)
    psi_ret = (psi_x - psi_now) / psi_now
    ibex_est = ibex_now * (1 + BETA_IBEX_PSI * psi_ret)
    fund_val = fv * (1 + BETA_FUND_PSI * psi_ret)
    # 纯ATM×16
    atm_pay = np.maximum(K_atm - ibex_est, 0) * 16
    atm_hedged = fund_val + atm_pay - atm_prem
    # 混合 ATM×8 + 90%OTM×20
    mix_pay = np.maximum(K_atm - ibex_est, 0) * 8 + np.maximum(K_90 - ibex_est, 0) * 20
    mix_hedged = fund_val + mix_pay - mix_prem
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=psi_x, y=fund_val, name='不对冲',
        line=dict(color='#c62828',width=2.5,dash='dot'),
        hovertemplate='PSI20:%{x:,.0f}<br>基金:€%{y:,.0f}<extra></extra>'))
    fig.add_trace(go.Scatter(x=psi_x, y=atm_hedged, name=f'纯ATM×16 (€{atm_prem:,}/年)',
        line=dict(color='#888',width=2,dash='dash'),
        hovertemplate='PSI20:%{x:,.0f}<br>纯ATM:€%{y:,.0f}<extra></extra>'))
    fig.add_trace(go.Scatter(x=psi_x, y=mix_hedged, name=f'ATM×8+OTM×20 [推荐] (€{mix_prem:,}/年)',
        line=dict(color='#2e7d32',width=3),
        hovertemplate='PSI20:%{x:,.0f}<br>混合:€%{y:,.0f}<extra></extra>'))
    fig.add_hline(y=fv, line_dash='dot', line_color='gray', opacity=0.3,
        annotation_text=f'当前€{fv:,}', annotation_position='top left', annotation_font=dict(size=10,color='gray'))
    fig.add_vline(x=psi_now, line_dash='dot', line_color='gray', opacity=0.4,
        annotation_text=f'当前PSI20 {psi_now:,.0f}', annotation_position='top right', annotation_font=dict(size=10,color='gray'))
    for level in [8000, 7000]:
        fig.add_vline(x=level, line_dash='dash', line_color='#e65100', opacity=0.3,
            annotation_text=f'{level:,}', annotation_position='bottom left', annotation_font=dict(size=9,color='#e65100'))
    fig.update_layout(template='plotly_white', height=360, xaxis_title='PSI20点位',
        yaxis_title='基金市值 (EUR)', yaxis_tickformat=',',
        legend=dict(x=0.01,y=0.01,bgcolor='rgba(255,255,255,0.9)'),
        margin=dict(t=10,b=50,l=80,r=20), hovermode='x unified')
    return fig.to_json()

def chart_payoff_planb(rec, live):
    """损益图：Plan B 纯OTM×24"""
    fv, psi_now, ibex_now = live['fund_value'], live['psi'], live['ibex']
    T_put = live.get('best_put_T', 1.0)
    K_90 = round(ibex_now * 0.90 / 50) * 50
    p_90 = bs_put(ibex_now, K_90, T_put)
    otm_prem = round(p_90*24)
    psi_x = np.linspace(5000, 11000, 500)
    psi_ret = (psi_x - psi_now) / psi_now
    ibex_est = ibex_now * (1 + BETA_IBEX_PSI * psi_ret)
    fund_val = fv * (1 + BETA_FUND_PSI * psi_ret)
    otm_pay = np.maximum(K_90 - ibex_est, 0) * 24
    otm_hedged = fund_val + otm_pay - otm_prem
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=psi_x, y=fund_val, name='不对冲',
        line=dict(color='#c62828',width=2.5,dash='dot'),
        hovertemplate='PSI20:%{x:,.0f}<br>基金:€%{y:,.0f}<extra></extra>'))
    fig.add_trace(go.Scatter(x=psi_x, y=otm_hedged, name=f'纯OTM×24 (€{otm_prem:,}/年)',
        line=dict(color='#e65100',width=3),
        hovertemplate='PSI20:%{x:,.0f}<br>OTM:€%{y:,.0f}<extra></extra>'))
    fig.add_hline(y=fv, line_dash='dot', line_color='gray', opacity=0.3,
        annotation_text=f'当前€{fv:,}', annotation_position='top left', annotation_font=dict(size=10,color='gray'))
    fig.add_vline(x=psi_now, line_dash='dot', line_color='gray', opacity=0.4,
        annotation_text=f'当前PSI20 {psi_now:,.0f}', annotation_position='top right', annotation_font=dict(size=10,color='gray'))
    for level in [8000, 7000]:
        fig.add_vline(x=level, line_dash='dash', line_color='#e65100', opacity=0.3,
            annotation_text=f'{level:,}', annotation_position='bottom left', annotation_font=dict(size=9,color='#e65100'))
    fig.update_layout(template='plotly_white', height=360, xaxis_title='PSI20点位',
        yaxis_title='基金市值 (EUR)', yaxis_tickformat=',',
        legend=dict(x=0.01,y=0.01,bgcolor='rgba(255,255,255,0.9)'),
        margin=dict(t=10,b=50,l=80,r=20), hovermode='x unified')
    return fig.to_json()

def t(zh, en):
    """Bilingual text wrapper"""
    return f'<span class="zh">{zh}</span><span class="en">{en}</span>'

def _build_live_chain_html(live, ibex_now, K_atm, K_otm, best_exp_code=None):
    """生成MEFF实时期权链HTML表格"""
    chain_data = live.get('meff_chain', {})
    chains = chain_data.get('chains', {})
    labels = chain_data.get('labels', {})
    if not chains:
        return ''
    # 选择所有Put到期月（排除近期周度期权，只保留月度以上）
    from datetime import datetime
    now = datetime.now()
    put_exps = []
    for code in sorted(chains.keys()):
        if not code.startswith('OPE'):
            continue
        # 解析到期日: OPE20260918 → 2026-09-18
        try:
            exp_date = datetime.strptime(code[3:], '%Y%m%d')
            days_to_exp = (exp_date - now).days
            if days_to_exp > 30:  # 只显示1个月以上的
                put_exps.append((code, exp_date, days_to_exp))
        except ValueError:
            continue

    if not put_exps:
        return ''

    # 为每个到期月生成一个可折叠的表格
    panels_html = ''
    for idx, (code, exp_date, days) in enumerate(put_exps):
        pts = chains.get(code, [])
        if not pts:
            continue
        exp_label = labels.get(code, exp_date.strftime('%d/%m/%Y'))
        months = days / 30.4
        # 只过滤合理范围的行权价
        filtered = [p for p in pts if ibex_now * 0.7 <= p['strike'] <= ibex_now * 1.05]
        if not filtered:
            continue
        # 找到推荐行的摘要（用于折叠标题）
        atm_ask = next((p['ask'] for p in filtered if abs(p['strike'] - K_atm) < 51 and p.get('ask')), None)
        otm_ask = next((p['ask'] for p in filtered if abs(p['strike'] - K_otm) < 51 and p.get('ask')), None)
        summary_parts = []
        if atm_ask:
            summary_parts.append(f'ATM Ask €{atm_ask:,.0f}')
        if otm_ask:
            summary_parts.append(f'90%OTM Ask €{otm_ask:,.0f}')
        summary = ' | '.join(summary_parts) if summary_parts else f'{len(filtered)} {t("个行权价","strikes")}'

        tbl_rows = ''
        for p in filtered:
            otm = (1 - p['strike']/ibex_now) * 100
            style = ''
            tag = ''
            if abs(p['strike'] - K_atm) < 51:
                style = 'background:#e8f5e9;font-weight:600'
                tag = f' <b style="color:#2e7d32">← ATM</b>'
            elif abs(p['strike'] - K_otm) < 51:
                style = 'background:#fff3e0;font-weight:600'
                tag = f' <b style="color:#e65100">← 90%OTM</b>'
            bid_s = f'€{p["bid"]:,.0f}' if p['bid'] else '-'
            ask_s = f'€{p["ask"]:,.0f}' if p['ask'] else '-'
            spread_s = f'€{p["ask"]-p["bid"]:,.0f}' if p['bid'] and p['ask'] else '-'
            last_s = f'€{p["last"]:,.0f}' if p['last'] else '-'
            liq = '✓' if p['bid'] and p['ask'] else (t('卖','Ask') if p['ask'] else (t('买','Bid') if p['bid'] else ''))
            tbl_rows += f'''<tr style="{style}">
                <td style="text-align:right">{p["strike"]:,.0f}</td>
                <td style="text-align:right">{otm:.1f}%</td>
                <td style="text-align:right;color:#2e7d32">{bid_s}</td>
                <td style="text-align:right;color:#c62828">{ask_s}</td>
                <td style="text-align:right;color:#888">{spread_s}</td>
                <td style="text-align:right">{last_s}</td>
                <td style="text-align:center">{liq}{tag}</td></tr>'''

        is_rec = (code == best_exp_code)
        open_attr = ' open' if is_rec else ''
        border_color = '#1a237e' if is_rec else '#ccc'
        rec_badge = f' <span style="background:#1a237e;color:white;padding:2px 8px;border-radius:4px;font-size:11px;margin-left:8px">{t("推荐","REC")}</span>' if is_rec else ''
        panels_html += f'''
<details style="margin-bottom:8px;border:2px solid {border_color};border-radius:8px;overflow:hidden"{open_attr}>
  <summary style="padding:10px 14px;background:{'#e8eaf6' if is_rec else '#f5f5f5'};cursor:pointer;font-weight:700;font-size:13px;display:flex;justify-content:space-between;align-items:center">
    <span>{t('到期','Expiry')} <b>{exp_label}</b> ({t(f'约{months:.0f}个月',f'~{months:.0f} months')}) — {len(filtered)} {t('个行权价','strikes')}{rec_badge}</span>
    <span style="color:#1a237e;font-size:12px">{summary}</span>
  </summary>
  <div style="overflow-x:auto;padding:0">
  <table style="width:100%;border-collapse:collapse;font-size:12px">
  <thead><tr style="background:#1a237e;color:white">
    <th style="padding:5px;text-align:right">Strike</th>
    <th style="padding:5px;text-align:right">OTM%</th>
    <th style="padding:5px;text-align:right;color:#a5d6a7">Bid</th>
    <th style="padding:5px;text-align:right;color:#ef9a9a">Ask</th>
    <th style="padding:5px;text-align:right">Spread</th>
    <th style="padding:5px;text-align:right">Last</th>
    <th style="padding:5px;text-align:center">{t('流动性','Liq.')}</th>
  </tr></thead>
  <tbody>{tbl_rows}</tbody>
  </table></div>
</details>'''

    return f'''
<div class="section">
<h2>{t('六B、MEFF 实时期权链','Section 6B: MEFF Live Option Chain')}</h2>
<div class="alert a-warn" style="font-size:13px;margin-bottom:12px">
  <b>{t('以下为MEFF交易所真实报价','Below are actual MEFF exchange quotes')}</b>({t('延迟约15分钟','~15 min delay')})。
  IBEX ≈ <b>{ibex_now:,.0f}</b> | ATM Strike ≈ <b>{K_atm:,}</b> | 90%OTM Strike ≈ <b>{K_otm:,}</b><br>
  {t('Bid=买入价(你卖出价)，Ask=卖出价(你买入价)。下单时以IBKR终端实时报价为准。','Bid = buy price (your sell), Ask = sell price (your buy). Use IBKR terminal live quotes when placing orders.')}
</div>
{panels_html}
<p style="font-size:11px;color:#888;margin-top:8px">{t('数据来源: MEFF官网实时行情页 | Mini IBEX | 乘数 €1/点 | 欧式期权','Source: MEFF live market page | Mini IBEX | Multiplier €1/pt | European style')}</p>
</div>'''

# ─── HTML ─────────────────────────────
def generate_html(fund_df, psi_df, res, live):
    df, events, strats, rec, options = res['df'], res['events'], res['strats'], res['rec'], res['options']
    psi_scenarios = res['psi_scenarios']
    psi_scenarios_b = res['psi_scenarios_b']
    K_90 = res['K_90']
    rec_prem = res['rec_prem']
    planb_prem = res['planb_prem']
    avg_crash_ratio = res['avg_crash_ratio']
    crash_ratios = res['crash_ratios']
    es = res['entry_scenario']
    K = rec['K']
    T_put = rec.get('T', 1.0)
    put_expiry_label = live.get('best_put_expiry_label', '2027-03')
    fv = live['fund_value']
    fund_nav = live['fund_nav']
    psi_now = live['psi']
    ibex_now = live['ibex']
    estx_now = live['estx']
    gen_time = live['gen_time']

    # 从MEFF实时期权链中查找真实Ask价格（如有）
    _real_atm_ask = None
    _real_otm_ask = None
    chain_data = live.get('meff_chain', {})
    if chain_data.get('chains'):
        # 找最接近1年的到期月
        from datetime import datetime as _dt
        now = _dt.now()
        best_code, best_diff = None, 9999
        for code in chain_data['chains']:
            if not code.startswith('OPE'):
                continue
            try:
                exp = _dt.strptime(code[3:], '%Y%m%d')
                diff = abs((exp - now).days - 365)
                if diff < best_diff:
                    best_code, best_diff = code, diff
            except ValueError:
                continue
        if best_code:
            pts = chain_data['chains'][best_code]
            for p in pts:
                if abs(p['strike'] - K) < 51 and p.get('ask'):
                    _real_atm_ask = p['ask']
                if abs(p['strike'] - K_90) < 51 and p.get('ask'):
                    _real_otm_ask = p['ask']

    c1 = chart_fund_psi_ibex(fund_df, psi_df, res['df'][['Date','ibex']], live)
    c2 = chart_fund_ibex(df, events)
    c5 = chart_payoff(rec, live)
    zooms = make_zoom_charts(df, events, strats, fv)

    fg = (fund_nav / FUND_ENTRY_PRICE - 1) * 100
    nt = len(events)
    ns = sum(1 for e in events if e['sync'])
    ibex_avg_chg = f'{np.mean([e["ibex_chg"] for e in events]):.1f}'
    fund_avg_chg = f'{np.mean([e["fund_chg"] for e in events]):.1f}'
    ncr = len(crash_ratios)
    cr_min = f'{min(crash_ratios):.3f}'
    cr_max = f'{max(crash_ratios):.3f}'
    cr_pct_above = round((avg_crash_ratio/BETA_FUND_IBEX-1)*100)
    cov_pct_actual = round(BETA_FUND_IBEX/avg_crash_ratio*100)

    # Event table rows + tab buttons + panels
    hist_rows, tab_btns, tab_panels = '', '', ''
    for i, ev in enumerate(events):
        act = ' active' if i==0 else ''
        hold = '' if ev['in_hold'] else f' <span style="font-size:10px;color:#aaa">{t("(买入前)","(Pre-entry)")}</span>'
        cr = ev.get('crash_ratio')
        cr_str = f'{cr:.3f}' if cr else '-'
        cr_color = '#c62828' if cr and cr > BETA_FUND_IBEX else '#2e7d32'
        hist_rows += f"""<tr style="background:#f0fff0;cursor:pointer" onclick="showTab({i})">
          <td>{i+1}</td><td>{ev['start_str']}~{ev['end_str']}{hold}</td>
          <td style="color:#c62828;font-weight:600">{ev['fund_chg']:.1f}%</td>
          <td style="color:#2e7d32;font-weight:600">{ev['ibex_chg']:+.1f}%</td>
          <td>{ev['ibex_peak']:,.0f}&rarr;{ev['ibex_level']:,.0f}</td>
          <td style="color:{cr_color};font-weight:600">{cr_str}</td></tr>"""

        tab_btns += f'<button class="tab-btn{act}" onclick="showTab({i})">#{i+1} {ev["end_str"][5:]}</button>'

        loss = ev.get('fund_loss', abs(fv*ev['fund_chg']/100))
        detail = ''
        if ev['in_hold'] and ev['strat']:
            detail = f'<table style="font-size:13px;margin-top:12px"><tr><th style="text-align:left">{t("策略","Strategy")}</th><th>{t("行权价","Strike")}<br><span style="font-weight:400;font-size:10px;color:#888">{t("（模拟中的历史值）","(historical backtest value)")}</span></th><th>{t("Put赚了","Put Gained")}</th><th>{t("净亏","Net Loss")}</th><th>{t("覆盖率","Coverage")}</th></tr>'
            for k, lab in [('12M',t('12月ATM年滚','12M ATM Annual Roll')),('6M',t('6月ATM半年滚','6M ATM Semi-annual Roll')),('3M',t('3月ATM季滚','3M ATM Quarterly Roll'))]:
                r = ev['strat'].get(k,{})
                m, st, c = r.get('mtm',0), r.get('strike',0), r.get('cov',0)
                cc = '#2e7d32' if c>=20 else ('#e65100' if c>=5 else '#c62828')
                rm = f' <b style="color:#2e7d32">[{t("推荐","Rec.")}]</b>' if k=='12M' else ''
                detail += f'<tr><td style="text-align:left">{lab}{rm}</td><td style="font-size:12px;color:#888">{st:,.0f}</td><td style="color:#2e7d32;font-weight:700">+&euro;{m:,.0f}</td><td style="color:#1565c0;font-weight:700">-&euro;{loss-m:,.0f}</td><td style="font-weight:700;color:{cc}">{c:.0f}%</td></tr>'
            detail += '</table>'
        elif not ev['in_hold']:
            detail = f'<p style="font-size:13px;color:#888;margin-top:8px">{t("此事件发生在你买入基金之前，仅作为IBEX同步性的历史参考。","This event occurred before your fund purchase, included only as historical reference for IBEX synchronization.")}</p>'

        tab_panels += f"""<div class="tab-panel{act}" id="panel_{i}">
          <div class="two-col">
            <div>
              <div class="alert a-good" style="margin-bottom:8px">
                <b>#{i+1} {ev['start_str']} ~ {ev['end_str']}</b><br>
                {t('基金1周跌','Fund 1-wk drop')} <b style="color:#c62828">{ev['fund_chg']:.1f}%</b> ({t('约','~')}&euro;{loss:,.0f})<br>
                IBEX&plusmn;{t('2周跌','2-wk drop')} <b style="color:#2e7d32">{ev['ibex_chg']:+.1f}%</b> ({ev['ibex_peak']:,.0f}&rarr;{ev['ibex_level']:,.0f})
              </div>{detail}
            </div>
            <div class="chart-box" style="padding:8px"><div id="zoom_{i}" style="height:260px"></div></div>
          </div></div>"""

    # Contract options table — mixed-strike configs
    opt_rows = ''
    for i, opt in enumerate(options):
        is_rec = (i == 1)  # ATM×8 + 90%OTM×20 推荐
        st = ' style="background:#f0fff0;font-weight:600"' if is_rec else ''
        tag = f' <span style="color:#2e7d32;font-size:11px">[{t("推荐","Rec.")}]</span>' if is_rec else ''
        s5, s10, s20, s30 = opt['scenarios'][5], opt['scenarios'][10], opt['scenarios'][20], opt['scenarios'][30]
        opt_rows += f'''<tr{st}>
          <td style="text-align:left">{opt['label']}{tag}<br><span style="font-size:10px;color:#888">{opt['desc']}</span></td>
          <td>&euro;{opt['prem']:,}<br><span style="font-size:10px;color:#888">{opt['annual_pct']:.2f}%/{t('年','yr')}</span></td>
          <td>&euro;{opt['five_yr']:,}<br><span style="font-size:10px;color:#888">{opt['five_yr']/fv*100:.1f}%</span></td>
          <td style="color:{'#2e7d32' if s5['payoff']>0 else '#ccc'}">&euro;{s5['payoff']:,}</td>
          <td style="color:#2e7d32;font-weight:600">&euro;{s10['payoff']:,}</td>
          <td style="color:#2e7d32;font-weight:600">&euro;{s20['payoff']:,}</td>
          <td style="color:#2e7d32;font-weight:700">&euro;{s30['payoff']:,}</td>
        </tr>'''

    # Cost table (rolling frequency)
    cost_rows = ''
    for k, lab, freq in [('12M',t('12个月ATM 年滚','12M ATM Annual Roll'),1),('6M',t('6个月ATM 半年滚','6M ATM Semi-annual Roll'),2),('3M',t('3个月ATM 季滚','3M ATM Quarterly Roll'),4)]:
        s = strats[k]; T = s['months']/12
        pe = bs_put(ibex_now, ibex_now, T)*N_CONTRACTS
        ae = pe * (12/s['months']); fy = ae*5
        is_r = k=='12M'
        st = ' style="background:#f0fff0;font-weight:600"' if is_r else ''
        tg = f' <span style="color:#2e7d32;font-size:11px">[{t("推荐","Rec.")}]</span>' if is_r else ''
        cost_rows += f'<tr{st}><td style="text-align:left">{lab}{tg}</td><td>{freq}x/{t("年","yr")}</td><td>&euro;{ae:,.0f} ({ae/fv*100:.2f}%)</td><td>&euro;{fy:,.0f} ({fy/fv*100:.1f}%)</td></tr>'

    # PSI20 scenario rows for section 六 Plan A
    psi_rows = ''
    for sc in psi_scenarios:
        cc = '#2e7d32' if sc['cov']>=80 else ('#e65100' if sc['cov']>=40 else '#c62828')
        psi_drop = (sc['psi'] - psi_now) / psi_now * 100
        psi_rows += f'''<tr>
          <td style="font-weight:700">{sc['psi']:,} <span style="font-size:10px;color:#888">({psi_drop:+.0f}%)</span></td>
          <td>&euro;{sc['fund_est']:,}</td>
          <td style="color:#c62828;font-weight:600">-&euro;{sc['fund_loss']:,}</td>
          <td style="color:#2e7d32;font-weight:600">+&euro;{sc['put_pay']:,}</td>
          <td style="font-weight:700">&euro;{sc['net']:,}</td>
          <td style="color:{cc};font-weight:700">{sc['cov']}%</td>
        </tr>'''

    # PSI20 scenario rows for section 六 Plan B
    psi_rows_b = ''
    for sc in psi_scenarios_b:
        cc = '#2e7d32' if sc['cov']>=80 else ('#e65100' if sc['cov']>=40 else '#c62828')
        psi_drop = (sc['psi'] - psi_now) / psi_now * 100
        psi_rows_b += f'''<tr>
          <td style="font-weight:700">{sc['psi']:,} <span style="font-size:10px;color:#888">({psi_drop:+.0f}%)</span></td>
          <td>&euro;{sc['fund_est']:,}</td>
          <td style="color:#c62828;font-weight:600">-&euro;{sc['fund_loss']:,}</td>
          <td style="color:#2e7d32;font-weight:600">+&euro;{sc['put_pay']:,}</td>
          <td style="font-weight:700">&euro;{sc['net']:,}</td>
          <td style="color:{cc};font-weight:700">{sc['cov']}%</td>
        </tr>'''

    # Plan B payoff chart
    planb_chart = chart_payoff_planb(rec, live)

    html = f"""<!DOCTYPE html><html lang="zh-CN"><head><meta charset="utf-8">
<title>{t('葡萄牙基金对冲方案','Portuguese Fund Hedge Analysis')}</title>
<script src="https://cdn.plot.ly/plotly-3.4.0.min.js" crossorigin="anonymous"></script>
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;background:#f4f6fb;color:#1a1a2e;font-size:15px;line-height:1.65}}
.page{{max-width:1100px;margin:0 auto;padding:32px 20px 70px}}
h1{{font-size:26px;font-weight:800;color:#1a237e;margin-bottom:6px}}
.meta{{color:#777;font-size:13px;margin-bottom:36px}}
h2{{font-size:17px;font-weight:700;color:#1a237e;margin-bottom:16px;padding:6px 12px;border-left:4px solid #1a237e;background:#eef2ff;border-radius:0 6px 6px 0}}
.section{{margin-bottom:40px}}
.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:12px;margin-bottom:16px}}
.card{{background:white;border-radius:12px;padding:16px 14px;box-shadow:0 2px 10px rgba(0,0,0,0.07)}}
.card .lbl{{font-size:11px;color:#999;letter-spacing:.5px;margin-bottom:3px}}
.card .val{{font-size:22px;font-weight:800;margin-bottom:2px}}
.card .sub{{font-size:12px;color:#bbb}}
.green .val{{color:#2e7d32}}.purple .val{{color:#6a1b9a}}.orange .val{{color:#e65100}}
.alert{{border-radius:10px;padding:14px 16px;margin-bottom:14px;font-size:14px;line-height:1.8}}
.a-warn{{background:#fff8e1;border-left:5px solid #f9a825}}
.a-info{{background:#e3f2fd;border-left:5px solid #1565c0}}
.a-good{{background:#e8f5e9;border-left:5px solid #388e3c}}
.a-bad{{background:#fdecea;border-left:5px solid #c62828}}
.a-note{{background:#f3e5f5;border-left:5px solid #7b1fa2}}
.chart-box{{background:white;border-radius:12px;padding:16px;box-shadow:0 2px 10px rgba(0,0,0,0.07);margin-bottom:16px}}
table{{width:100%;border-collapse:collapse;background:white;border-radius:12px;overflow:hidden;box-shadow:0 2px 10px rgba(0,0,0,0.07);font-size:13px;margin-bottom:16px}}
th{{background:#1a237e;color:white;padding:10px 8px;text-align:center;font-size:11px}}
td{{padding:9px 8px;text-align:center;border-bottom:1px solid #eee}}
tr:last-child td{{border:none}} tr:hover td{{background:#f5f5ff}}
.note-sm{{font-size:12px;color:#aaa;margin-top:6px}}
.rec{{background:#e8f5e9;border:2px solid #388e3c;border-radius:12px;padding:20px 24px;margin-bottom:16px}}
.rec h3{{color:#1b5e20;font-size:16px;margin-bottom:12px}}
.rec-grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px}}
.rec-item{{background:white;border-radius:8px;padding:10px 12px}}
.rec-item .rl{{font-size:11px;color:#888;margin-bottom:2px}}
.rec-item .rv{{font-size:20px;font-weight:800;color:#1b5e20}}
.steps ol{{padding-left:20px}}.steps li{{margin-bottom:10px;font-size:14px;line-height:1.7}}
.steps li b{{color:#1a237e}}
.chain{{display:flex;align-items:center;gap:0;margin:16px 0;flex-wrap:wrap}}
.chain-node{{background:white;border-radius:10px;padding:12px 16px;box-shadow:0 2px 8px rgba(0,0,0,0.07);text-align:center;min-width:130px}}
.chain-arrow{{font-size:22px;color:#1a237e;padding:0 6px;font-weight:800}}
.two-col{{display:grid;grid-template-columns:1fr 1fr;gap:14px;margin-bottom:14px}}
.tab-bar{{display:flex;gap:3px;margin-bottom:0;flex-wrap:wrap}}
.tab-btn{{padding:7px 12px;border:none;background:#e0e0e0;border-radius:8px 8px 0 0;cursor:pointer;font-size:11px;font-weight:600;color:#555}}
.tab-btn.active{{background:#1a237e;color:white}}
.tab-btn:hover{{background:#c5cae9}}
.tab-panel{{display:none;background:white;border-radius:0 12px 12px 12px;padding:18px;box-shadow:0 2px 10px rgba(0,0,0,0.07);margin-bottom:16px}}
.tab-panel.active{{display:block}}
@media(max-width:700px){{.cards{{grid-template-columns:1fr 1fr}}.two-col{{grid-template-columns:1fr}}
.chain{{flex-direction:column}}.chain-arrow{{transform:rotate(90deg)}}}}
.data-bar{{background:white;border-radius:10px;padding:12px 16px;margin-bottom:20px;box-shadow:0 2px 10px rgba(0,0,0,0.07);display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:8px;font-size:13px}}
.data-bar .src{{color:#888}}.data-bar .src b{{color:#1a237e}}
.data-bar .timer{{color:#e65100;font-weight:600}}
.plan-nav{{position:sticky;top:0;z-index:100;background:rgba(244,246,251,0.95);backdrop-filter:blur(8px);padding:10px 0;margin-bottom:16px;border-bottom:2px solid #e0e0e0}}
.plan-nav .inner{{max-width:1100px;margin:0 auto;display:flex;align-items:center;gap:12px;padding:0 20px}}
.plan-nav .label{{font-size:13px;color:#888;font-weight:600}}
.plan-btn{{padding:10px 24px;border:2px solid #ccc;border-radius:10px;cursor:pointer;font-size:14px;font-weight:700;background:white;transition:all .2s;line-height:1.3}}
.plan-btn:hover{{border-color:#1a237e;background:#f5f5ff}}
.plan-btn.active-a{{border-color:#2e7d32;background:#e8f5e9;color:#1b5e20}}
.plan-btn.active-b{{border-color:#e65100;background:#fff3e0;color:#e65100}}
.plan-panel-a,.plan-panel-b{{display:none}}
.plan-panel-a.show,.plan-panel-b.show{{display:block}}
body.lang-en .zh{{display:none}}
body.lang-zh .en{{display:none}}
</style></head><body class="lang-zh">

<div class="plan-nav">
  <div class="inner">
    <span class="label">{t('方案切换：','Switch Plan:')}</span>
    <button class="plan-btn active-a" id="btn-a" onclick="switchPlan('a')">
      A: {t('混合 ATM+OTM','Mixed ATM+OTM')}<br><span style="font-size:11px;font-weight:400">{t('小跌+大跌都保护','Protects both small &amp; large drops')}</span>
    </button>
    <button class="plan-btn" id="btn-b" onclick="switchPlan('b')">
      B: {t('纯OTM 省钱版','Pure OTM Budget')}<br><span style="font-size:11px;font-weight:400">{t('只防崩盘，成本低约25-40%','Crash-only protection, ~25-40% cheaper')}</span>
    </button>
    <button class="lang-btn" id="lang-btn" onclick="toggleLang()" style="margin-left:auto;padding:8px 16px;border:2px solid #1a237e;border-radius:8px;background:white;cursor:pointer;font-size:13px;font-weight:700;color:#1a237e">EN</button>
  </div>
</div>

<div class="page">

<h1>{t('葡萄牙基金对冲方案','Portuguese Fund Hedge Analysis')}</h1>
<p class="meta">{t('Optimize Portugal Golden Opportunities Fund (PTOPZWHM0007)','Optimize Portugal Golden Opportunities Fund (PTOPZWHM0007)')}</p>

<div class="data-bar">
  <div class="src">
    {t('数据来源：基金NAV','Data Source: Fund NAV')} &larr; <b>{live['fund_src']}</b> &middot; {t('指数','Indices')} &larr; <b>Yahoo Finance</b><br>
    <span style="font-size:11px;color:#aaa">
      {t('基金NAV','Fund NAV')}: {live['fund_date']}{f' <span style="color:#e65100;font-weight:700">⚠ {t("NAV滞后" + str(live.get("fund_lag", 0)) + "天，基金公司延迟发布", "NAV delayed " + str(live.get("fund_lag", 0)) + " days")}</span>' if live.get('fund_lag', 0) >= 2 else ''} &middot;
      PSI20: {live['psi_date']} &middot;
      IBEX35: {live['ibex_date']} &middot;
      ESTOXX50: {live['estx_date']}
    </span><br>
    <span style="font-size:11px;color:#aaa">{t('指数来源：Yahoo Finance，基金NAV来源：','Index source: Yahoo Finance, Fund NAV source: ')}{live['fund_src']}</span>
  </div>
  <div style="text-align:right;min-width:200px;display:flex;flex-direction:column;align-items:flex-end;gap:4px">
    <div style="font-size:11px;color:#888">{t('报告生成于','Report generated at')}</div>
    <div style="font-size:15px;font-weight:800;color:#1a237e">{gen_time}</div>
    <div class="timer" id="timer"></div>
    <button onclick="fetchAll()" id="refresh-btn" style="margin-top:4px;padding:6px 16px;border:2px solid #1a237e;border-radius:8px;background:#eef2ff;cursor:pointer;font-weight:700;font-size:12px;color:#1a237e">{t('刷新数据','Refresh Data')}</button>
    <span id="refresh-status" style="font-size:11px"></span>
  </div>
</div>
<input type="hidden" id="ibex-input" value="{ibex_now:.0f}">

<div class="section">
<h2>{t('一、持仓概况','Section 1: Holdings Overview')}</h2>
<div class="cards">
  <div class="card green"><div class="lbl">{t('买入成本','Entry Cost')}</div><div class="val">&euro;{INITIAL_INV:,}</div><div class="sub">2024.07 NAV &euro;{FUND_ENTRY_PRICE:.2f}</div></div>
  <div class="card green"><div class="lbl">{t('当前市值','Current Value')}</div><div class="val">&euro;{fv:,}</div><div class="sub">NAV &euro;{fund_nav:.2f}</div></div>
  <div class="card purple"><div class="lbl">{t('浮盈','Unrealized Gain')}</div><div class="val">{"+" if fv>=INITIAL_INV else ""}&euro;{fv-INITIAL_INV:,}</div><div class="sub">{fg:+.1f}%</div></div>
  <div class="card orange"><div class="lbl">PSI20</div><div class="val" data-live="psi">{psi_now:,.0f}</div><div class="sub">{t('买入时','At entry')}{PSI20_ENTRY_ACT:,}</div></div>
</div>
<div style="display:flex;gap:14px;margin-bottom:12px;font-size:12px;color:#888">
  <span>IBEX35: <b style="color:#e65100" data-live="ibex">{ibex_now:,.0f}</b></span>
  <span>ESTOXX50: <b style="color:#6a1b9a" data-live="estx">{estx_now:,.0f}</b></span>
</div>
<div class="chart-box"><div id="c1" style="height:340px"></div></div>
<div class="alert a-warn">
  <b>{t('担忧：','Concern:')}</b>{t('俄乌停战、欧盟加息、美欧关税等全欧系统性事件导致大跌。计划持有5年，想买保险。','Systemic European events (Russia-Ukraine ceasefire, ECB rate hikes, US-EU tariffs) could cause a major crash. Planning to hold for 5 years and want insurance.')}
</div>
</div>

<div class="section">
<h2>{t('二、历史'+str(nt)+'次急跌：IBEX全部同步','Section 2: '+str(nt)+' Historical Crashes — IBEX Always in Sync')}</h2>
<div class="chart-box"><div id="c2" style="height:460px"></div></div>
<p class="note-sm">{t('图中绿色编号标记对应下表中的急跌事件。点击表格行查看放大详情。','Green numbered markers in the chart correspond to crash events in the table below. Click rows for zoom details.')}</p>

<table>
  <tr><th>#</th><th>{t('时期','Period')}</th><th>{t('基金1周跌','Fund 1-wk Drop')}</th><th>IBEX&plusmn;{t('2周跌','2-wk Drop')}</th><th>{t('IBEX点位变化','IBEX Level Change')}</th><th>{t('急跌比率','Crash Ratio')}<br><span style="font-weight:400;font-size:9px">{t('基金跌/IBEX跌','Fund drop/IBEX drop')}</span></th></tr>
  {hist_rows}
</table>

<div class="alert a-good">
  <b>{t(str(nt)+'次急跌（1周跌&gt;3%），IBEX在&plusmn;2周内全部同步下跌，0次脱钩。','All '+str(nt)+' crashes (1-wk drop &gt;3%) saw IBEX decline in sync within &plusmn;2 weeks. Zero decoupling.')}</b><br>
  {t(f'IBEX平均跌幅{np.mean([e["ibex_chg"] for e in events]):.1f}%，比基金平均跌幅{np.mean([e["fund_chg"] for e in events]):.1f}%更大。',f'IBEX avg drop {np.mean([e["ibex_chg"] for e in events]):.1f}%, larger than fund avg drop {np.mean([e["fund_chg"] for e in events]):.1f}%.')}<br>
  <span style="font-size:13px">{t(f'急跌比率（基金跌幅/IBEX跌幅）平均<b>{avg_crash_ratio:.3f}</b>，范围{min(crash_ratios):.3f}~{max(crash_ratios):.3f}。高于全样本Beta={BETA_FUND_IBEX}，说明<b>急跌时基金对IBEX的敏感度比平时更高</b>（条件Beta效应）。',f'Crash ratio (fund drop / IBEX drop) averages <b>{avg_crash_ratio:.3f}</b>, range {min(crash_ratios):.3f}~{max(crash_ratios):.3f}. Higher than full-sample Beta={BETA_FUND_IBEX}, indicating <b>higher fund sensitivity to IBEX during crashes</b> (conditional Beta effect).')}</span>
</div>

<p style="margin-bottom:8px"><b>{t('点击查看每次事件的放大走势：','Click to view zoomed chart for each event:')}</b></p>
<div class="tab-bar">{tab_btns}</div>
{tab_panels}
</div>

<div class="section">
<h2>{t('三、对冲链路与Beta说明','Section 3: Hedge Chain &amp; Beta Explanation')}</h2>
<div class="chain">
  <div class="chain-node"><div style="font-size:12px;color:#888">{t('你持有的','You hold')}</div><div style="font-size:17px;font-weight:800;color:#1565c0">{t('葡萄牙基金','PT Fund')}</div></div>
  <div class="chain-arrow">&rarr;</div>
  <div class="chain-node"><div style="font-size:12px;color:#888">{t('高度跟踪','Closely tracks')}</div><div style="font-size:17px;font-weight:800;color:#ff7f0e">PSI20</div><div style="font-size:11px;color:#2e7d32">R&sup2;=79%</div></div>
  <div class="chain-arrow">&rarr;</div>
  <div class="chain-node" style="border:2px solid #c62828"><div style="font-size:12px;color:#c62828">{t('PSI20无可用期权','No listed options on PSI20')}</div></div>
  <div class="chain-arrow">&rarr;</div>
  <div class="chain-node" style="border:2px solid #2e7d32"><div style="font-size:12px;color:#2e7d32">{t('替代','Proxy')}</div><div style="font-size:17px;font-weight:800;color:#e65100">IBEX35 Put</div><div style="font-size:11px;color:#888">MEFF &middot; {t('IBKR可交易','Tradable on IBKR')}</div></div>
</div>
<div class="alert a-info">
  {t('合约数量按Beta(基金/IBEX)='+str(BETA_FUND_IBEX)+'计算：基金对IBEX的敏感度为42%，','Contract count based on Beta(Fund/IBEX)='+str(BETA_FUND_IBEX)+': fund sensitivity to IBEX is 42%,')}
  {t('需要对冲的名义敞口','notional exposure to hedge')}=&euro;{fv:,}&times;{BETA_FUND_IBEX}=&euro;{round(fv*BETA_FUND_IBEX):,}{t('，除以IBEX点位','/ IBEX level')}=<b>{N_CONTRACTS} {t('张Mini合约','Mini contracts')}</b>{t('（乘数&euro;1/点）。','(multiplier &euro;1/pt).')}
</div>
<div class="alert a-warn">
  <b>{t('Beta说明','Beta Explanation')}</b>{t('：以下三个Beta是分别独立回归的结果，',': The following three Betas are independently regressed results,')}<b>{t('不是','NOT')}</b>{t('链式推导关系：',' chain-derived:')}<br>
  &middot; Beta({t('基金','Fund')}/PSI20) = {BETA_FUND_PSI}{t('（R&sup2;=79%，基金跟踪PSI20较好）',' (R&sup2;=79%, fund tracks PSI20 well)')}<br>
  &middot; Beta({t('基金','Fund')}/IBEX) = {BETA_FUND_IBEX}{t('（R&sup2;=42%，IBEX只能解释基金42%的波动）',' (R&sup2;=42%, IBEX explains only 42% of fund variance)')}<br>
  &middot; Beta(IBEX/PSI20) = {BETA_IBEX_PSI}{t('（用于PSI20场景→IBEX点位换算）',' (used for PSI20 scenario → IBEX level conversion)')}<br>
  <span style="font-size:12px;color:#888">{t('注：R&sup2;=42%意味着基金有58%的波动无法被IBEX解释。Put只对冲IBEX相关的那42%风险。','Note: R&sup2;=42% means 58% of fund variance cannot be explained by IBEX. Puts only hedge the IBEX-correlated 42%.')}</span>
</div>
<div class="alert a-info">
  <b>{t('条件Beta（急跌时）','Conditional Beta (During Crashes)')}</b>{t('：上面的Beta='+str(BETA_FUND_IBEX)+'是全样本（含平时+急跌）的平均值。',': The above Beta='+str(BETA_FUND_IBEX)+' is the full-sample (calm + crash) average.')}
  {t('但实证显示，在'+str(ncr)+'次急跌事件中，基金跌幅/IBEX跌幅的平均比率为','Empirical evidence shows that in '+str(ncr)+' crash events, the avg fund drop / IBEX drop ratio is')}<b>{avg_crash_ratio}</b>{t('，比全样本Beta高',', exceeding full-sample Beta by')}<b>{cr_pct_above}%</b>。<br>
  {t('这意味着：急跌时基金对IBEX的真实敏感度更高。按Beta='+str(BETA_FUND_IBEX)+'计算的16张合约，在急跌时实际只能覆盖约','This means: true fund sensitivity to IBEX is higher during crashes. The 16 contracts based on Beta='+str(BETA_FUND_IBEX)+' can only cover about')}<b>{cov_pct_actual}%</b>{t('的IBEX相关损失，而非理论上的100%。',' of IBEX-related losses during crashes, not the theoretical 100%.')}<br>
  <span style="font-size:12px;color:#888">
  {t('学术背景：危机中跨市场相关性上升是公认现象（Longin &amp; Solnik 2001）。条件Beta &gt; 无条件Beta是正常的。','Academic background: Rising cross-market correlation during crises is well-documented (Longin &amp; Solnik 2001). Conditional Beta &gt; unconditional Beta is expected.')}
  {t('但'+str(ncr)+'次样本量偏小（比率范围'+cr_min+'~'+cr_max+'），不宜过度精确化。','However, '+str(ncr)+' samples is small (ratio range '+cr_min+'~'+cr_max+'), so over-precision is unwarranted.')}
  {t('本报告仍以全样本Beta定合约数，但提醒用户实际覆盖率可能低于理论值。','This report still uses full-sample Beta for contract sizing, but reminds users actual coverage may be lower than theoretical.')}</span>
</div>
</div>

<div class="section">
<h2>{t('四、合约配置：混合行权价策略','Section 4: Contract Configuration — Mixed Strike Strategy')}</h2>

<div style="margin-bottom:14px">
<div style="display:flex;align-items:center;gap:10px;margin-bottom:8px">
  <span style="font-size:14px;font-weight:700;color:#1a237e">{t('IBEX35 Put 期权定价参考','IBEX35 Put Pricing Reference')}</span>
  <span style="font-size:11px;color:{'#2e7d32' if live.get('meff_source')=='MEFF' else '#aaa'}">
    {'MEFF ' + t('实盘IV插值','market IV interpolated') + ' (' + live.get('meff_expiry','?') + ', ' + str(len(live.get('meff_points',[]))) + t('个数据点',' data points') + ')' if live.get('meff_source')=='MEFF' else 'BS ' + t('理论价','theoretical') + ' | IV=' + str(IBEX_IMPLIED_VOL*100) + '%'}
    | r={ECB_RATE*100:.1f}% | T={T_put:.2f}yr | {t('到期','Expiry')} {live.get('meff_expiry','2027-03')}
  </span>
</div>
<table id="option-chain-table" style="font-size:13px">
  <tr>
    <th>{t('行权价','Strike')}</th>
    <th>{t('距现价','vs Spot')}</th>
    <th>IV</th>
    <th>{t('价格/张','Price/Ct')}</th>
    <th>{t('方案A','Plan A')}</th>
    <th>{t('方案B','Plan B')}</th>
  </tr>
  <tbody id="chain-body"></tbody>
</table>
<div style="font-size:11px;color:#aaa;margin-top:4px">
  {t('价格基于'+('MEFF实盘IV插值（已包含skew）' if live.get('meff_source')=='MEFF' else 'BS理论价（固定IV='+str(IBEX_IMPLIED_VOL*100)+'%，实际OTM因skew可能高30-50%）')+'。',
     'Prices based on '+('MEFF market IV interpolation (skew included)' if live.get('meff_source')=='MEFF' else 'BS theoretical (flat IV='+str(IBEX_IMPLIED_VOL*100)+'%, actual OTM may be 30-50% higher due to skew)')+'.')}
  {t('查看实盘报价：','View live quotes: ')}
  <a href="https://www.meff.es/docs/Ficheros/boletin/ing/boletiipthu.htm" target="_blank" style="color:#1a237e;font-weight:600">{t('MEFF每日行情公报（含完整期权链）','MEFF Daily Bulletin (full option chain)')}</a>
  &middot; <a href="https://www.meff.es/aspx/calculadoras/calculadoraOp.aspx?id=ing" target="_blank" style="color:#1a237e;font-weight:600">{t('MEFF期权计算器','MEFF Option Simulator')}</a>
  &middot; <a href="https://www.meff.es/ing/Financial-Derivatives/Market-Prices" target="_blank" style="color:#1a237e;font-weight:600">{t('MEFF实时行情','MEFF Market Prices')}</a>
  &middot; <a href="javascript:void(0)" onclick="toggleFullChain()" id="chain-toggle" style="color:#1a237e;font-weight:600">{t('展开完整链','Show full chain')}</a>
</div>
<div id="full-chain-box" style="display:none;margin-top:8px">
<table style="font-size:12px">
  <tr><th>{t('行权价','Strike')}</th><th>{t('距现价','vs Spot')}</th><th>{t('BS价格','BS Price')}</th></tr>
  <tbody id="full-chain-body"></tbody>
</table>
</div>
</div>

<div class="alert a-info">
  <b>{t('核心思路','Core Idea')}</b>{t('：不必全买贵的ATM Put。用一部分预算买便宜的虚值OTM Put，张数更多，大跌时赔付反而更高——同样的保费，换来更强的崩盘保护。',': No need to buy all expensive ATM Puts. Use part of the budget for cheaper OTM Puts — more contracts, higher payout in large crashes. Same premium, stronger crash protection.')}
</div>
<table>
  <tr><th style="text-align:left">{t('配置','Config')}</th><th>{t('年保费','Annual Premium')}</th><th>{t('5年总保费','5-Year Total')}</th><th>{t('IBEX跌5%','IBEX -5%')}</th><th>{t('IBEX跌10%','IBEX -10%')}</th><th>{t('IBEX跌20%','IBEX -20%')}</th><th>{t('IBEX跌30%','IBEX -30%')}</th></tr>
  <tbody id="config-tbody">{opt_rows}</tbody>
</table>
<div class="alert a-warn" style="font-size:13px">
  <b>{t('怎么读这张表','How to Read This Table')}</b>{t('：','：')}<br>
  &middot; <b>{t('纯ATM ×16','Pure ATM ×16')}</b>{t('：所有跌幅都有赔付，但大跌时赔付最少（因为只有16张）',': Payout at all drop levels, but lowest payout in large crashes (only 16 contracts)')}<br>
  &middot; <b>ATM ×8 + 90%OTM ×20 [{t('推荐','Recommended')}]</b>{t('：小跌仍有保护（8张ATM兜底），同预算但合约更多，大跌时赔付更高',': Still protected in small drops (8 ATM as floor), same budget but more contracts, higher payout in large crashes')}<br>
  &middot; <b>{t('纯90%OTM ×24','Pure 90%OTM ×24')}</b>{t('：最省钱，但10%以内的跌幅完全不赔',': Cheapest, but zero payout for drops under 10%')}<br>
  {t('OTM行权价','OTM Strike')}={K_90:,}{t('点（IBEX当前90%），ATM行权价','pts (90% of current IBEX), ATM Strike')}={K:,}{t('点','pts')}
</div>
<div class="alert a-bad">
  <b>{t('所有配置共同的局限','Common Limitation of All Configs')}</b>{t('：Put赔付基于IBEX维度。基金实际损失取决于PSI20和基金自身因素（R&sup2;=42%），Put无法覆盖IBEX以外58%的风险。在"先涨后跌"行情下，即使是ATM Put也可能因行权价被甩开而失效（Event #10、#11教训）。',': Put payout is based on IBEX dimension. Actual fund losses depend on PSI20 and fund-specific factors (R&sup2;=42%). Puts cannot cover the 58% of risk unrelated to IBEX. In "rally-then-crash" scenarios, even ATM Puts may become ineffective as the strike gets left behind (lessons from Events #10, #11).')}
</div>
</div>

<div class="section">
<h2>{t('五、滚仓频率对比','Section 5: Rolling Frequency Comparison')}</h2>
<table>
  <tr><th style="text-align:left">{t('策略','Strategy')}</th><th>{t('操作频率','Frequency')}</th><th>{t('年化成本','Annualized Cost')}</th><th>{t('5年总成本','5-Year Total Cost')}</th></tr>
  {cost_rows}
</table>
<div class="alert a-info">
  {t('三种频率保护效果接近，但成本差距大。','All three frequencies offer similar protection, but costs differ significantly.')}<b>{t(f'选最长可用到期月（当前{put_expiry_label}，约{round(T_put*12)}个月），花最少的钱，操作最简单。',f'Choose the longest available expiry (currently {put_expiry_label}, ~{round(T_put*12)} months) for lowest cost and simplest operation.')}</b>
</div>
</div>

<!-- ===== Plan A: 六 ===== -->
<div class="plan-panel-a show">
<div class="section">
<h2>{t('六、推荐方案 A：混合行权价','Section 6: Recommended Plan A — Mixed Strike')}</h2>
<div class="rec">
  <h3>ATM Put &times;8 + 90%OTM Put &times;20{t(f'，{round(T_put*12)}个月滚仓 + 动态滚仓', f', {round(T_put*12)}-Month Roll + Dynamic Rolling')}</h3>
  <div class="rec-grid">
    <div class="rec-item"><div class="rl">{t('ATM行权价','ATM Strike')}</div><div class="rv"><span id="dyn-pa-atm-k">{rec['K']:,}</span>{t('点','pts')}</div><div style="font-size:10px;color:#888">&times;8{t('张',' contracts')}</div></div>
    <div class="rec-item"><div class="rl">{t('OTM行权价','OTM Strike')}</div><div class="rv"><span id="dyn-pa-otm-k">{K_90:,}</span>{t('点','pts')}</div><div style="font-size:10px;color:#888">&times;20{t('张（90% OTM）',' contracts (90% OTM)')}</div></div>
    <div class="rec-item"><div class="rl">{t('每年保费','Annual Premium')}</div><div class="rv"><span id="dyn-pa-prem">&euro;{rec_prem:,}</span></div><div style="font-size:10px;color:#888">{t('BS理论值','BS theoretical')}, <span id="dyn-pa-prem-pct">{rec_prem/fv*100:.2f}%</span></div>{f'<div style="font-size:11px;color:#c62828;margin-top:4px">MEFF Ask: ATM €{_real_atm_ask:,.0f}×8 + OTM €{_real_otm_ask:,.0f}×20 = <b>€{round(_real_atm_ask*8+_real_otm_ask*20):,}</b></div>' if _real_atm_ask and _real_otm_ask else ''}</div>
    <div class="rec-item"><div class="rl">{t('5年总保费','5-Year Total')}</div><div class="rv"><span id="dyn-pa-5yr">&euro;{rec_prem*5:,}</span></div>{f'<div style="font-size:10px;color:#c62828">MEFF: €{round((_real_atm_ask*8+_real_otm_ask*20)*5):,}</div>' if _real_atm_ask and _real_otm_ask else ''}</div>
    <div class="rec-item"><div class="rl">vs {t('纯ATM×16','Pure ATM×16')}</div><div class="rv">{t('同预算，合约更多','Same budget, more contracts')}</div></div>
  </div>
</div>
<div class="alert a-info" style="font-size:13px">
  <b>{t('为什么混合配置','Why Mixed Config')}</b>{t('：同样~&euro;'+f'{rec_prem:,}'+'/年预算，8张ATM保住小跌时的基本保护，20张90%OTM在大跌时提供额外赔付（OTM单价仅ATM的40%，同预算可买更多张数）。对比见上表第四列"IBEX跌30%"，混合配置赔付',': With the same ~&euro;'+f'{rec_prem:,}'+'/yr budget, 8 ATM Puts retain basic protection for small drops, while 20 OTM Puts provide extra payout in large crashes (OTM is only 40% of ATM price, so same budget buys more contracts). See "IBEX -30%" column: mixed config pays')}&euro;{options[1]['scenarios'][30]['payoff']:,} vs {t('纯ATM','Pure ATM')} &euro;{options[0]['scenarios'][30]['payoff']:,}。
</div>

<p style="margin:16px 0 8px;font-weight:700">{t('如果PSI20跌到…你的基金会怎样？','What happens to your fund if PSI20 drops to…?')}</p>
<table>
  <tr><th>{t('PSI20跌到','PSI20 Drops To')}</th><th>{t('基金预估市值','Est. Fund Value')}</th><th>{t('预估亏损','Est. Loss')}</th><th>{t('Put赔付','Put Payout')}</th><th>{t('对冲后市值','Hedged Value')}</th><th>{t('覆盖率','Coverage')}</th></tr>
  <tbody id="psi-tbody-a">{psi_rows}</tbody>
</table>
<div class="alert a-warn" style="font-size:13px">
  <b>{t('线性模型局限','Linear Model Limitation')}</b>{t('：上表覆盖率随跌幅递增——小跌时8张ATM独扛（覆盖率低），大跌时20张OTM逐步启动、赔付加速上升（覆盖率可达50%以上）。这正是混合行权价策略的优势。',': Coverage in the table above increases with drop size — in small drops only 8 ATM contracts carry the load (low coverage), while in large drops the 20 OTM contracts progressively activate with accelerating payout (coverage can exceed 50%). This is the advantage of a mixed-strike strategy.')}<br>
  {t('但模型使用恒定Beta，极端行情下Beta会漂移、尾部相关性变化，实际覆盖率可能偏离。实证研究表明危机中跨市场相关性趋于上升（Longin &amp; Solnik 2001），大跌时覆盖率可能','The model uses constant Beta, but in extreme conditions Beta drifts and tail correlations change, so actual coverage may deviate. Empirical research shows cross-market correlation rises during crises (Longin &amp; Solnik 2001), so coverage in large drops may be')}<b>{t('高于','higher than')}</b>{t('此估计。',' this estimate.')}<br>
  <b>{t('更重要的是','More importantly')}</b>{t('：此表假设Put行权价在事件发生时仍为ATM/OTM。如果IBEX在持有期内先涨后跌（如Event #10），行权价被甩开变成深度虚值，实际覆盖率可能远低于表中数值。见下方"局限性"详细分析。',': This table assumes Put strikes remain ATM/OTM when the event occurs. If IBEX rallies then crashes during the holding period (e.g., Event #10), strikes get left behind as deep OTM, and actual coverage could be far lower than shown. See "Limitations" section below.')}
</div>
<div class="chart-box"><div id="c5" style="height:360px"></div></div>

<div class="alert a-good" style="font-size:14px;line-height:1.9;border:2px solid #388e3c">
  <b style="font-size:16px">{t('如果PSI20跌回买入时的'+f'{es["psi_target"]:,}'+'点？（方案A）','What if PSI20 drops back to entry level '+f'{es["psi_target"]:,}'+' pts? (Plan A)')}</b><br>
  <span style="font-size:13px;color:#888">{t('PSI20从当前跌'+str(es['psi_drop_pct'])+'%，IBEX估计跌到','PSI20 drops '+str(es['psi_drop_pct'])+'% from current, IBEX est. drops to ')}<span id="dyn-es-ibex">{es["ibex_est"]:,}</span>{t('点（Beta换算，非等比回落）',' pts (Beta conversion, not proportional)')}</span><br><br>
  <b>{t('不对冲：','Unhedged:')}</b>&euro;{fv:,} &rarr; <b><span id="dyn-es-fund">&euro;{es['fund_est']:,}</span></b>{t('，亏损',', loss')}<span id="dyn-es-loss">&euro;{es['fund_loss']:,}</span>{t('，浮盈保住',', unrealized gain retained')}<b><span id="dyn-es-nohedge">{es['no_hedge_kept']}%</span></b><br><br>
  <b style="color:#2e7d32">{t('方案A对冲后：','Plan A Hedged:')}</b><br>
  <span style="display:inline-block;width:16px"></span>{t('基金市值','Fund value')} <b>&euro;{es['fund_est']:,}</b><br>
  <span style="display:inline-block;width:16px"></span>+ {t('Put赔付','Put Payout')} <b style="color:#2e7d32"><span id="dyn-es-pa-pay">+&euro;{es['pa_pay']:,}</span></b><br>
  <span style="display:inline-block;width:16px"></span>&minus; {t('年保费','Annual Premium')} <b style="color:#c62828"><span id="dyn-es-pa-prem">&minus;&euro;{rec_prem:,}</span></b><br>
  <span style="display:inline-block;width:16px"></span>= {t('组合净值','Portfolio Net Value')} <b style="color:#2e7d32;font-size:17px"><span id="dyn-es-pa-net">&euro;{es['pa_net']:,}</span></b>
  <span style="font-size:13px;color:#888">(vs {t('买入成本','entry cost')}&euro;{INITIAL_INV:,})</span><br>
  <span style="display:inline-block;width:16px"></span><b style="color:#2e7d32">{t('浮盈保住','Unrealized gain retained')}<span id="dyn-es-pa-kept">{es['pa_kept']}%</span></b>
  <span style="font-size:13px;color:#888">({t('当前浮盈','current gain')}&euro;{es['gain_now']:,}{t('，对冲后仍盈利',', still profitable after hedge')}<span id="dyn-es-pa-gain">&euro;{es['pa_net']-INITIAL_INV:,}</span>)</span>
</div>
</div>
</div>

<!-- ===== Plan B: 六 ===== -->
<div class="plan-panel-b">
<div class="section">
<h2>{t('六、推荐方案 B：纯OTM省钱版','Section 6: Recommended Plan B — Pure OTM Budget')}</h2>
<div class="rec" style="border-color:#e65100;background:#fff3e0">
  <h3 style="color:#e65100">{t('纯90%OTM Put &times;24，12个月年滚 + 动态滚仓','Pure 90%OTM Put &times;24, 12-Month Annual Roll + Dynamic Rolling')}</h3>
  <div class="rec-grid">
    <div class="rec-item"><div class="rl">{t('OTM行权价','OTM Strike')}</div><div class="rv" style="color:#e65100"><span id="dyn-pb-otm-k">{K_90:,}</span>{t('点','pts')}</div><div style="font-size:10px;color:#888">&times;24{t('张（90% OTM）',' contracts (90% OTM)')}</div></div>
    <div class="rec-item"><div class="rl">{t('每年保费','Annual Premium')}</div><div class="rv" style="color:#e65100"><span id="dyn-pb-prem">&euro;{planb_prem:,}</span></div><div style="font-size:10px;color:#888">{t('BS理论值','BS theoretical')}, <span id="dyn-pb-prem-pct">{planb_prem/fv*100:.2f}%</span></div>{f'<div style="font-size:11px;color:#c62828;margin-top:4px">MEFF Ask: €{_real_otm_ask:,.0f}×24 = <b>€{round(_real_otm_ask*24):,}</b></div>' if _real_otm_ask else ''}</div>
    <div class="rec-item"><div class="rl">{t('5年总保费','5-Year Total')}</div><div class="rv" style="color:#e65100"><span id="dyn-pb-5yr">&euro;{planb_prem*5:,}</span></div>{f'<div style="font-size:10px;color:#c62828">MEFF: €{round(_real_otm_ask*24*5):,}</div>' if _real_otm_ask else ''}</div>
    <div class="rec-item"><div class="rl">vs {t('方案A','Plan A')}</div><div class="rv" style="color:#e65100"><span id="dyn-pb-save">{t('省'+str(round((1-planb_prem/rec_prem)*100))+'%保费','Save '+str(round((1-planb_prem/rec_prem)*100))+'% premium')}</span></div></div>
    <div class="rec-item"><div class="rl">{t('代价','Trade-off')}</div><div class="rv" style="color:#c62828;font-size:16px">{t('10%以内跌幅不赔','No payout for drops under 10%')}</div></div>
  </div>
</div>
<div class="alert a-warn" style="font-size:13px">
  <b>{t('为什么是24张？','Why 24 contracts?')}</b>{t('条件Beta（实证平均'+str(avg_crash_ratio)+'）计算出基准约'+str(round(fv*avg_crash_ratio/ibex_now))+'张，但急跌比率波动范围大（'+f'{min(crash_ratios):.2f}'+'~'+f'{max(crash_ratios):.2f}'+'），最差一次达'+f'{max(crash_ratios):.3f}'+'。取24张是在基准'+str(round(fv*avg_crash_ratio/ibex_now))+'张之上加约'+str(round((24/round(fv*avg_crash_ratio/ibex_now)-1)*100))+'%的安全边际，以应对急跌比率高于平均值的情况。','Conditional Beta (empirical avg '+str(avg_crash_ratio)+') gives a baseline of ~'+str(round(fv*avg_crash_ratio/ibex_now))+' contracts, but crash ratio varies widely ('+f'{min(crash_ratios):.2f}'+'~'+f'{max(crash_ratios):.2f}'+'), worst case '+f'{max(crash_ratios):.3f}'+'. 24 contracts adds ~'+str(round((24/round(fv*avg_crash_ratio/ibex_now)-1)*100))+'% safety margin above the '+str(round(fv*avg_crash_ratio/ibex_now))+' baseline to handle above-average crash ratios.')}<br><br>
  <b>{t('方案B的逻辑','Plan B Logic')}</b>{t('：如果你认为小幅回调（5-10%）可以承受，只想防范崩盘式暴跌（>10%），那么全部买便宜的虚值Put，省下来的保费本身就是一种保护（少花钱=少损失确定成本）。',': If you can tolerate small pullbacks (5-10%) and only want to protect against crash-level drops (>10%), then buying all cheap OTM Puts makes sense — the saved premium is itself a form of protection (less spending = less certain loss).')}<br>
  <b>{t('适合','Best for')}</b>{t('：风险承受力较高、不想每年花太多保费的投资者。',': Higher risk tolerance investors who want to minimize annual premium spending.')}<br>
  <b>{t('不适合','Not for')}</b>{t('：希望任何级别下跌都有赔付的投资者（请选方案A）。',': Investors who want payout at any drop level (choose Plan A instead).')}
</div>

<p style="margin:16px 0 8px;font-weight:700">{t('如果PSI20跌到…你的基金会怎样？（方案B）','What happens to your fund if PSI20 drops to…? (Plan B)')}</p>
<table>
  <tr><th>{t('PSI20跌到','PSI20 Drops To')}</th><th>{t('基金预估市值','Est. Fund Value')}</th><th>{t('预估亏损','Est. Loss')}</th><th>{t('Put赔付','Put Payout')}</th><th>{t('对冲后市值','Hedged Value')}</th><th>{t('覆盖率','Coverage')}</th></tr>
  <tbody id="psi-tbody-b">{psi_rows_b}</tbody>
</table>
<div class="alert a-info" style="font-size:13px">
  {t('注意：当PSI20只跌5-8%时，IBEX估计跌幅不足10%，OTM Put尚未进入实值区，赔付为零。只有PSI20跌到较低水平（约7500以下），Put才开始大额赔付。','Note: When PSI20 only drops 5-8%, IBEX estimated drop is under 10%, OTM Puts remain out of the money with zero payout. Only when PSI20 drops to lower levels (below ~7500) do the Puts begin significant payouts.')}
</div>
<div class="chart-box"><div id="c5b" style="height:360px"></div></div>

<div class="alert a-good" style="font-size:14px;line-height:1.9;border:2px solid #e65100;background:#fff3e0;border-left:5px solid #e65100">
  <b style="font-size:16px;color:#e65100">{t('如果PSI20跌回买入时的'+f'{es["psi_target"]:,}'+'点？（方案B）','What if PSI20 drops back to entry level '+f'{es["psi_target"]:,}'+' pts? (Plan B)')}</b><br>
  <span style="font-size:13px;color:#888">{t('PSI20从当前跌'+str(es['psi_drop_pct'])+'%，IBEX估计跌到'+f'{es["ibex_est"]:,}'+'点（Beta换算，非等比回落）','PSI20 drops '+str(es['psi_drop_pct'])+'% from current, IBEX est. drops to '+f'{es["ibex_est"]:,}'+' pts (Beta conversion, not proportional)')}</span><br><br>
  <b>{t('不对冲：','Unhedged:')}</b>&euro;{fv:,} &rarr; <b>&euro;{es['fund_est']:,}</b>{t('，亏损',', loss')}&euro;{es['fund_loss']:,}{t('，浮盈保住',', unrealized gain retained')}<b>{es['no_hedge_kept']}%</b><br><br>
  <b style="color:#e65100">{t('方案B对冲后：','Plan B Hedged:')}</b><br>
  <span style="display:inline-block;width:16px"></span>{t('基金市值','Fund value')} <b>&euro;{es['fund_est']:,}</b><br>
  <span style="display:inline-block;width:16px"></span>+ {t('Put赔付','Put Payout')} <b style="color:#2e7d32"><span id="dyn-es-pb-pay">+&euro;{es['pb_pay']:,}</span></b><br>
  <span style="display:inline-block;width:16px"></span>&minus; {t('年保费','Annual Premium')} <b style="color:#c62828"><span id="dyn-es-pb-prem">&minus;&euro;{planb_prem:,}</span></b><br>
  <span style="display:inline-block;width:16px"></span>= {t('组合净值','Portfolio Net Value')} <b style="color:#e65100;font-size:17px"><span id="dyn-es-pb-net">&euro;{es['pb_net']:,}</span></b>
  <span style="font-size:13px;color:#888">(vs {t('买入成本','entry cost')}&euro;{INITIAL_INV:,})</span><br>
  <span style="display:inline-block;width:16px"></span><b style="color:#e65100">{t('浮盈保住','Unrealized gain retained')}<span id="dyn-es-pb-kept">{es['pb_kept']}%</span></b>
  <span style="font-size:13px;color:#888">({t('当前浮盈','current gain')}&euro;{es['gain_now']:,}{t('，对冲后仍盈利',', still profitable after hedge')}<span id="dyn-es-pb-gain">&euro;{es['pb_net']-INITIAL_INV:,}</span>)</span><br><br>
  <span style="font-size:13px;color:#888">{t('对比方案A：保住浮盈','Compared to Plan A: retains ')}<span id="dyn-es-pa-kept2">{es['pa_kept']}</span>{t('%，多保','% of gains, ')}<span id="dyn-es-pa-diff">{es['pa_kept']-es['pb_kept']}</span>{t('个百分点，但每年多花',' percentage points more, but costs ')}<span id="dyn-es-prem-diff">&euro;{rec_prem-planb_prem:,}</span>{t('保费。',' more per year.')}</span>
</div>
</div>
</div>

{_build_live_chain_html(live, ibex_now, K, K_90, live.get('best_put_expiry_code'))}
<!-- ===== Plan A: 七 ===== -->
<div class="plan-panel-a show">
<div class="section">
<h2>{t('七、操作步骤（方案A）','Section 7: Operating Steps (Plan A)')}</h2>
<div class="steps"><ol>
  <li><b>{t('IBKR账户','IBKR Account')}</b>{t('：开通欧洲期权交易权限，交易所选MEFF。',': Enable European options trading permission, select MEFF exchange.')}</li>
  <li><b>{t('买第一腿——ATM Put &times;8','Buy Leg 1 — ATM Put &times;8')}</b>{t('：搜索Mini IBEX期权，到期月',': Search Mini IBEX options, expiry month')}<b>{put_expiry_label}</b>{t('，类型Put，行权价',', type Put, strike')}<b><span id="dyn-s7-atm-k">{rec['K']:,}</span></b>{t('（ATM，50点间距）。参考价约',' (ATM, 50-pt intervals). Ref. price ~')}<span id="dyn-s7-atm-price">&euro;{rec['price']:,.0f}</span>/{t('张','contract')}{t('，8张合计约',', 8 contracts total ~')}<span id="dyn-s7-atm-total">&euro;{round(rec['price']*8):,}</span>。</li>
  <li><b>{t('买第二腿——90%OTM Put &times;20','Buy Leg 2 — 90%OTM Put &times;20')}</b>{t('：同到期月，行权价',': Same expiry month, strike')}<b><span id="dyn-s7-otm-k">{K_90:,}</span></b> (90% OTM){t('。参考价约','. Ref. price ~')}<span id="dyn-s7-otm-price">&euro;{round(bs_put(ibex_now,K_90,T_put)):,}</span>/{t('张','contract')}{t('，20张合计约',', 20 contracts total ~')}<span id="dyn-s7-otm-total">&euro;{round(bs_put(ibex_now,K_90,T_put)*20):,}</span>。</li>
  <li><b>{t('总保费','Total Premium')}</b>{t('约',' ~')}<span id="dyn-s7-prem-a">&euro;{rec_prem:,}</span> (Black-Scholes, {'MEFF IV ' + t('插值','interpolated') if live.get('meff_source')=='MEFF' else 'IV=' + str(IBEX_IMPLIED_VOL*100) + '%'}, r={ECB_RATE*100:.1f}%){t('。','.')}
  <b>{t('实际市价预计上浮10-30%','Actual market price expected 10-30% higher')}</b>{t('，尤其OTM Put因波动率偏斜（skew）真实IV约22-25%，比报告使用的平值IV='+str(IBEX_IMPLIED_VOL*100)+'%更高，OTM部分实际价格可能高于BS理论值30-50%。下单前务必以IBKR/MEFF实际报价为准。',', especially OTM Puts due to volatility skew (actual IV ~22-25%, higher than the ATM IV='+str(IBEX_IMPLIED_VOL*100)+'% used in this report). OTM actual prices may be 30-50% above BS theoretical values. Always use IBKR/MEFF live quotes before placing orders.')}</li>
  <li><b>{t('分腿动态滚仓','Split-Leg Dynamic Rolling')}</b>{t('（关键！）：ATM腿和OTM腿职责不同，滚仓策略也不同：',' (Critical!): ATM and OTM legs have different roles, so different rolling strategies:')}<br>
  <b style="color:#1565c0">ATM×8{t('（跟踪腿）',' (Tracking Leg)')}</b>{t('：IBEX涨超10%（>','：When IBEX rises >10% (>')}<span id="dyn-s7-trigger-a">{round(ibex_now*1.10):,}</span>{t('）时',')')}<b>{t('立即滚仓',' roll immediately')}</b>{t('——卖掉旧ATM Put，买入新ATM Put重设行权价。ATM必须紧贴当前市场，否则小跌时赔不了（Event #10教训）。建议每月检查。',' — sell old ATM Put, buy new ATM Put to reset strike. ATM must stay close to current market; otherwise it won&rsquo;t pay in small drops (Event #10 lesson). Check monthly.')}<br>
  <b style="color:#e65100">OTM×20{t('（兜底腿）',' (Floor Leg)')}</b>{t('：','：')}<b>{t('不参与动态滚仓','Does NOT participate in dynamic rolling')}</b>{t('，只做年度正常到期滚仓。OTM买来就是防崩盘（-20%~-30%），IBEX涨10%后它从虚值10%变成虚值20%，但真来大崩盘时仍会深度实值，赔付差额有限。',', only annual expiry rolling. OTM is bought purely for crash protection (-20%~-30%). After IBEX rises 10%, it goes from 10% OTM to 20% OTM, but in a real crash it will still be deep ITM with limited payout difference.')}</li>
  <li><b>{t('到期前1个月滚仓','Roll 1 Month Before Expiry')}</b>{t('：ATM和OTM两条腿都正常到期滚仓，卖旧买新，周而复始。',': Both ATM and OTM legs undergo normal expiry rolling — sell old, buy new, repeat.')}</li>
</ol></div>
<div class="alert a-info" style="font-size:13px">
  <b>{t('分腿滚仓的成本优势','Cost Advantage of Split-Leg Rolling')}</b>{t('：每次动态触发只滚8张ATM（而非全部28张），大幅降低滚仓损耗。',': Each dynamic trigger only rolls 8 ATM contracts (not all 28), significantly reducing rolling costs.')}<br>
  &middot; <b>{t('全部滚仓','Roll All')}</b>{t('（旧方案）：28张全滚，单次损耗',' (old plan): roll all 28, single roll cost')} &asymp; 8&times;&euro;{rec['price']:,.0f}&times;40% + 20&times;&euro;{round(bs_put(ibex_now,K_90,T_put)):,}&times;40% = <b><span id="dyn-s7-roll-full">&euro;{round(rec['price']*8*0.4 + bs_put(ibex_now,K_90,T_put)*20*0.4):,}</span></b>{t('，每年2次',', 2x/yr')} = <span id="dyn-s7-roll-full-yr">&euro;{round((rec['price']*8*0.4 + bs_put(ibex_now,K_90,T_put)*20*0.4)*2):,}</span>/{t('年','yr')}<br>
  &middot; <b>{t('分腿滚仓','Split-Leg Rolling')}</b>{t('（推荐）：只滚8张ATM，单次损耗',' (recommended): only roll 8 ATM, single roll cost')} &asymp; 8&times;&euro;{rec['price']:,.0f}&times;40% = <b><span id="dyn-s7-roll-split">&euro;{round(rec['price']*8*0.4):,}</span></b>{t('，每年2次',', 2x/yr')} = <span id="dyn-s7-roll-split-yr">&euro;{round(rec['price']*8*0.4*2):,}</span>/{t('年','yr')}<br>
  &middot; <b>{t('每年节省约','Annual savings ~')}<span id="dyn-s7-roll-save">&euro;{round(bs_put(ibex_now,K_90,T_put)*20*0.4*2):,}</span></b>{t('，5年节省约',', 5-year savings ~')}<span id="dyn-s7-roll-save-5yr">&euro;{round(bs_put(ibex_now,K_90,T_put)*20*0.4*2*5):,}</span><br><br>
  <b>{t('代价','Trade-off')}</b>{t('：在"连续大涨后崩盘"的极端场景中，未滚仓的OTM行权价较低，赔付会少约',': In the extreme "sustained rally then crash" scenario, unrolled OTM strikes are lower, payout will be ~')}&euro;{round((ibex_now*0.1)*20):,}{t(' 少。但这笔差额与5年节省的滚仓成本大致打平。在更常见的"涨后中小跌"场景中，OTM本来就不赔，分不分开滚没有区别。',' less. But this difference roughly offsets the 5-year rolling cost savings. In the more common "rally then small drop" scenario, OTM wouldn&rsquo;t pay out anyway, so split vs. unified rolling makes no difference.')}
</div>
</div>
</div>

<!-- ===== Plan B: 七 ===== -->
<div class="plan-panel-b">
<div class="section">
<h2>{t('七、操作步骤（方案B）','Section 7: Operating Steps (Plan B)')}</h2>
<div class="steps"><ol>
  <li><b>{t('IBKR账户','IBKR Account')}</b>{t('：开通欧洲期权交易权限，交易所选MEFF。',': Enable European options trading permission, select MEFF exchange.')}</li>
  <li><b>{t('买90%OTM Put &times;24','Buy 90%OTM Put &times;24')}</b>{t('：搜索Mini IBEX期权，到期月',': Search Mini IBEX options, expiry month')}<b>{put_expiry_label}</b>{t('，类型Put，行权价',', type Put, strike')}<b><span id="dyn-s7-otm-k-b">{K_90:,}</span></b>{t('（90% OTM，50点间距）。参考价约',' (90% OTM, 50-pt intervals). Ref. price ~')}<span id="dyn-s7-otm-price-b">&euro;{round(bs_put(ibex_now,K_90,T_put)):,}</span>/{t('张','contract')}{t('，24张合计约',', 24 contracts total ~')}<span id="dyn-s7-otm-total-b">&euro;{planb_prem:,}</span>。</li>
  <li><b>{t('总保费','Total Premium')}</b>{t('约',' ~')}<span id="dyn-s7-prem-b">&euro;{planb_prem:,}</span> (Black-Scholes, {'MEFF IV ' + t('插值','interpolated') if live.get('meff_source')=='MEFF' else 'IV=' + str(IBEX_IMPLIED_VOL*100) + '%'}, r={ECB_RATE*100:.1f}%){t('。','.')}
  <b>{t('实际市价预计上浮30-50%','Actual market price expected 30-50% higher')}</b>{t('——OTM Put因波动率偏斜（skew）真实IV约22-25%，远高于平值IV='+str(IBEX_IMPLIED_VOL*100)+'%。实际年成本可能达',' — OTM Puts have higher actual IV (~22-25%) due to volatility skew, well above ATM IV='+str(IBEX_IMPLIED_VOL*100)+'%. Actual annual cost may reach')}&euro;{round(planb_prem*1.4):,}~{round(planb_prem*1.5):,}{t('。下单前务必以IBKR/MEFF实际报价为准。','. Always use IBKR/MEFF live quotes before placing orders.')}</li>
  <li><b>{t('动态滚仓','Dynamic Rolling')}</b>{t('：方案B全部是OTM Put，与方案A的分腿滚仓逻辑类似——OTM的职责是防崩盘，IBEX涨10%后从虚值10%变成虚值20%，但大崩盘时仍会深度实值。因此方案B',': Plan B is all OTM Puts, similar logic to Plan A&rsquo;s split-leg rolling — OTM&rsquo;s role is crash protection. After IBEX rises 10%, OTM goes from 10% to 20% OTM, but in a real crash it&rsquo;ll still be deep ITM. So Plan B')}<b>{t('以年度正常滚仓为主','primarily uses annual expiry rolling')}</b>{t('，不需要频繁动态触发。',', no frequent dynamic triggers needed.')}<br>
  {t('但如果IBEX','But if IBEX')}<b>{t('累计涨超20%','rises more than 20% cumulatively')}</b> (&gt;<span id="dyn-s7-trigger-b">{round(ibex_now*1.20):,}</span>){t('，OTM行权价已严重脱离市场，此时应滚仓重设行权价。建议每月检查。',', OTM strikes are too far from market — roll to reset strikes. Check monthly.')}</li>
  <li><b>{t('到期前1个月滚仓','Roll 1 Month Before Expiry')}</b>{t('：正常到期前卖旧买新，周而复始。',': Sell old, buy new before expiry, repeat.')}</li>
</ol></div>
<div class="alert a-warn" style="font-size:13px">
  <b>{t('方案B注意','Plan B Note')}</b>{t('：因为全部是OTM Put，小幅回调时Put不会赔付。这是刻意的选择——用更低成本换取"只防大灾"的保护。如果你发现自己担心5-10%的回调没有保护，应该切换到方案A。',': Since all contracts are OTM Puts, there is no payout during small pullbacks. This is intentional — lower cost in exchange for "crash-only" protection. If you find yourself worried about 5-10% pullbacks having no coverage, switch to Plan A.')}<br><br>
  <b>{t('方案B的滚仓成本优势','Plan B Rolling Cost Advantage')}</b>{t('：因为以年度滚仓为主（动态触发阈值为20%，远高于方案A的10%），预计每年额外动态滚仓0-1次，滚仓损耗远低于方案A。',': Since annual rolling is primary (dynamic trigger threshold is 20%, much higher than Plan A&rsquo;s 10%), expect 0-1 extra dynamic rolls per year, far less rolling cost than Plan A.')}
  {t('即使触发1次：','Even if triggered once: ')}24&times;&euro;{round(bs_put(ibex_now,K_90,T_put)):,}&times;40% = <span id="dyn-s7-roll-b">&euro;{round(bs_put(ibex_now,K_90,T_put)*24*0.4):,}</span>。
  {t('方案B真实年化总成本','Plan B true annualized total cost')} &asymp; &euro;{planb_prem:,} + &euro;{round(bs_put(ibex_now,K_90,T_put)*24*0.4*0.5):,} ({t('平均0.5次/年','avg 0.5x/yr')}) = <b><span id="dyn-s7-roll-b-total">&euro;{planb_prem + round(bs_put(ibex_now,K_90,T_put)*24*0.4*0.5):,}</span></b> ({t('未含skew上浮','excl. skew markup')})。
</div>
</div>
</div>

<div class="section">
<h2>{t('八、局限性（必读）','Section 8: Limitations (Must Read)')}</h2>
<div class="alert a-bad">
  <b>{t('Event #10 教训：年度滚仓的致命缺陷','Event #10 Lesson: Fatal Flaw of Annual Rolling')}</b><br>
  {t('2025年3-4月，基金跌7.4%（约&euro;49,000），这是持仓期最大一次回撤。但12月ATM Put的行权价设在2024年7月买入时的IBEX水平（约11,090点，注意：这是历史回测值，当前推荐行权价为'+f'{rec["K"]:,}'+'点），到事件发生时IBEX已涨到13,484点，即使跌到11,786点仍在行权价','In Mar-Apr 2025, the fund dropped 7.4% (~&euro;49,000), the largest drawdown during the holding period. But the 12M ATM Put strike was set at the July 2024 entry IBEX level (~11,090 pts, note: this is historical backtest value, current recommended strike is '+f'{rec["K"]:,}'+' pts). By the event, IBEX had rallied to 13,484 — even after dropping to 11,786 it was still')}<b>{t('之上','above the strike')}</b>{t('——Put几乎是废纸，覆盖仅约5%。',' — the Put was nearly worthless, covering only ~5%.')}<br><br>
  <b>{t('结论','Conclusion')}</b>{t('：固定年度滚仓在"先涨后跌"行情下保护形同虚设。这就是为什么操作步骤中加入了',': Fixed annual rolling provides virtually no protection in "rally-then-crash" scenarios. This is why the operating steps include')}<b>{t('动态滚仓触发','dynamic rolling triggers')}</b>{t('（IBEX涨超10%即提前滚仓重设行权价），并建议每月检查。',' (roll early when IBEX rises >10% to reset strikes), and monthly checks are recommended.')}
</div>
<div class="alert a-warn">
  <ol style="margin:0 0 0 18px">
    <li><b>{t('这是减震垫，不是全额保险','This is a shock absorber, not full insurance')}</b>{t('：方案A的覆盖率随跌幅递增（小跌~24%，大跌可达~57%），方案B仅在IBEX跌超10%后才启动赔付。在先涨后跌场景下可能远低于此。即使加入动态滚仓，也无法保证覆盖率。',': Plan A coverage increases with drop size (small drops ~24%, large drops up to ~57%). Plan B only activates after IBEX drops >10%. In rally-then-crash scenarios, coverage may be far lower. Even with dynamic rolling, coverage cannot be guaranteed.')}</li>
    <li><b>{t('保费是确定支出','Premium is a certain cost')}</b>{t('：每年&euro;'+f'{rec_prem:,}'+'（方案A），5年不出事白花&euro;'+f'{rec_prem*5:,}'+'（组合的'+f'{rec_prem*5/fv*100:.1f}'+'%）。',': &euro;'+f'{rec_prem:,}'+'/yr (Plan A), if nothing happens in 5 years you spend &euro;'+f'{rec_prem*5:,}'+' ('+f'{rec_prem*5/fv*100:.1f}'+'% of portfolio) for nothing.')}</li>
    <li><b>{t('R&sup2;=42%的根本限制','Fundamental R&sup2;=42% limitation')}</b>{t('：IBEX只能解释基金42%的波动。基金可能因为葡萄牙本地原因（个股暴雷、流动性危机）大跌而IBEX无动于衷，此时Put完全无效。',': IBEX explains only 42% of fund variance. The fund could crash due to Portugal-specific factors (stock blowups, liquidity crisis) while IBEX is unaffected — Puts would be completely useless.')}</li>
    <li><b>{t('条件Beta高于无条件Beta','Conditional Beta exceeds unconditional Beta')}</b>{t('：全样本Beta='+str(BETA_FUND_IBEX)+'用于计算合约数，但'+str(ncr)+'次历史急跌中实际比率平均'+str(avg_crash_ratio)+'（高'+str(cr_pct_above)+'%）。这意味着16张合约在急跌时只能覆盖约'+str(cov_pct_actual)+'%的IBEX相关损失，实际保护效果比场景表显示的更弱。样本量仅'+str(ncr)+'次，比率波动大（'+f'{min(crash_ratios):.2f}'+'~'+f'{max(crash_ratios):.2f}'+'），结论存在不确定性。',': Full-sample Beta='+str(BETA_FUND_IBEX)+' is used for contract sizing, but in '+str(ncr)+' historical crashes the actual ratio averaged '+str(avg_crash_ratio)+' ('+str(cr_pct_above)+'% higher). This means 16 contracts can only cover ~'+str(cov_pct_actual)+'% of IBEX-related losses during crashes — weaker than scenario tables suggest. Sample size is only '+str(ncr)+', with high ratio volatility ('+f'{min(crash_ratios):.2f}'+'~'+f'{max(crash_ratios):.2f}'+'), so conclusions carry uncertainty.')}</li>
    <li><b>{t('线性模型在极端行情下失真','Linear model breaks down in extreme conditions')}</b>{t('：场景表使用恒定Beta，但极端尾部事件中Beta会漂移，覆盖率可能偏离预期。',': Scenario tables use constant Beta, but in extreme tail events Beta drifts and coverage may deviate from expectations.')}</li>
    <li><b>{t('IV影响成本','IV affects cost')}</b>{t('：恐慌期Put更贵，尽量在平静期滚仓。',': Puts are more expensive during panic periods. Try to roll during calm markets.')}</li>
  </ol>
</div>
</div>

<div class="section">
<h2>{t('九、总结','Section 9: Summary')}</h2>
<div class="plan-panel-a show">
<div class="alert a-note" style="font-size:14px;line-height:1.9">
  <b style="font-size:17px;color:#4a148c">{t(f'方案A：ATM &times;8 + 90%OTM &times;20，{round(T_put*12)}个月滚仓 + 动态滚仓',f'Plan A: ATM &times;8 + 90%OTM &times;20, {round(T_put*12)}-Month Roll + Dynamic Rolling')}</b><br>
  <b>{t('能做到的','What it can do')}</b>{t('：历史'+str(nt)+'次急跌IBEX 100%同步下跌。混合配置用同样预算换更多合约，大跌时赔付高于纯ATM×16，同时8张ATM保留小跌保护。',': In all '+str(nt)+' historical crashes, IBEX dropped 100% in sync. Mixed config uses the same budget for more contracts, higher payout in large drops vs pure ATM×16, while 8 ATM contracts retain small-drop protection.')}<br>
  <b>{t('做不到的','What it cannot do')}</b>{t('：无法覆盖IBEX以外58%的风险（R&sup2;=42%）。在"先涨后跌"行情下，如果没有及时动态滚仓，Put可能接近废纸。',': Cannot cover the 58% of risk unrelated to IBEX (R&sup2;=42%). In "rally-then-crash" scenarios, Puts may become nearly worthless without timely dynamic rolling.')}<br>
  <b>{t('成本','Cost')}</b>{t('：年化'+f'{rec_prem/fv*100:.2f}'+'%（&euro;'+f'{rec_prem:,}'+'/年），5年约&euro;'+f'{rec_prem*5:,}'+'，是确定的支出。',': Annualized '+f'{rec_prem/fv*100:.2f}'+'% (&euro;'+f'{rec_prem:,}'+'/yr), ~&euro;'+f'{rec_prem*5:,}'+' over 5 years — a certain cost.')}<br>
  <b>{t('本质','In essence')}</b>{t('：这是一个减震垫，不是全额保险。它降低了系统性暴跌中的最大亏损幅度，但不能保证你不亏钱。',': This is a shock absorber, not full insurance. It reduces maximum loss magnitude during systemic crashes, but cannot guarantee you won&rsquo;t lose money.')}
</div>
</div>
<div class="plan-panel-b">
<div class="alert a-note" style="font-size:14px;line-height:1.9;border-color:#e65100;background:#fff3e0">
  <b style="font-size:17px;color:#e65100">{t(f'方案B：纯90%OTM &times;24，{round(T_put*12)}个月滚仓 + 动态滚仓',f'Plan B: Pure 90%OTM &times;24, {round(T_put*12)}-Month Roll + Dynamic Rolling')}</b><br>
  <b>{t('能做到的','What it can do')}</b>{t('：在崩盘式暴跌（>10%）时提供赔付，成本比方案A低'+str(round((1-planb_prem/rec_prem)*100))+'%。',': Provides payout during crash-level drops (>10%), '+str(round((1-planb_prem/rec_prem)*100))+'% cheaper than Plan A.')}<br>
  <b>{t('做不到的','What it cannot do')}</b>{t('：5-10%的中等回调完全没有保护。同样无法覆盖IBEX以外58%的风险。',': Zero protection for 5-10% medium pullbacks. Also cannot cover the 58% of risk unrelated to IBEX.')}<br>
  <b>{t('成本','Cost')}</b>{t('：年化'+f'{planb_prem/fv*100:.2f}'+'%（&euro;'+f'{planb_prem:,}'+'/年），5年约&euro;'+f'{planb_prem*5:,}'+'。',': Annualized '+f'{planb_prem/fv*100:.2f}'+'% (&euro;'+f'{planb_prem:,}'+'/yr), ~&euro;'+f'{planb_prem*5:,}'+' over 5 years.')}<br>
  <b>{t('适合','Best for')}</b>{t('：能承受中等回调、只想防黑天鹅的投资者。省下的保费本身也是一种保护。',': Investors who can tolerate moderate pullbacks and only want black swan protection. The saved premium itself is a form of protection.')}
</div>
</div>
</div>

</div>
<script>
Plotly.newPlot('c1',__C1__.data,__C1__.layout,{{responsive:true}});
Plotly.newPlot('c2',__C2__.data,__C2__.layout,{{responsive:true}});
Plotly.newPlot('c5',__C5__.data,__C5__.layout,{{responsive:true}});
var c5bData=__C5B__;
function switchPlan(p){{
  var a=document.querySelectorAll('.plan-panel-a'),b=document.querySelectorAll('.plan-panel-b');
  var btnA=document.getElementById('btn-a'),btnB=document.getElementById('btn-b');
  if(p==='a'){{
    a.forEach(function(el){{el.classList.add('show')}});b.forEach(function(el){{el.classList.remove('show')}});
    btnA.className='plan-btn active-a';btnB.className='plan-btn';
  }}else{{
    b.forEach(function(el){{el.classList.add('show')}});a.forEach(function(el){{el.classList.remove('show')}});
    btnB.className='plan-btn active-b';btnA.className='plan-btn';
    var e5b=document.getElementById('c5b');
    if(e5b&&!e5b.dataset.r){{Plotly.newPlot('c5b',c5bData.data,c5bData.layout,{{responsive:true}});e5b.dataset.r='1'}}
  }}
  window.scrollTo({{top:document.querySelector('.plan-panel-'+(p==='a'?'a':'b')+'.show').offsetTop-80,behavior:'smooth'}});
}}
var Z=__ZOOM__;
function showTab(i){{
  document.querySelectorAll('.tab-btn').forEach((b,j)=>b.classList.toggle('active',j===i));
  document.querySelectorAll('.tab-panel').forEach((p,j)=>{{
    p.classList.toggle('active',j===i);
    if(j===i&&Z[i]){{var e=document.getElementById('zoom_'+i);if(e&&!e.dataset.r){{Plotly.newPlot('zoom_'+i,Z[i].data,Z[i].layout,{{responsive:true}});e.dataset.r='1'}}}}
  }});
  // Highlight row
  document.querySelectorAll('table tr[onclick]').forEach((r,j)=>r.style.outline=j===i?'2px solid #1a237e':'none');
}}
if(Z[0]){{Plotly.newPlot('zoom_0',Z[0].data,Z[0].layout,{{responsive:true}});document.getElementById('zoom_0').dataset.r='1'}}
// Countdown since generation
(function(){{
  var gen=new Date('{gen_time.replace(" ","T")}:00');
  var el=document.getElementById('timer');
  function tick(){{
    var now=new Date();var d=Math.floor((now-gen)/1000);
    if(d<0)d=0;
    var dd=Math.floor(d/86400),hh=Math.floor(d%86400/3600),mm=Math.floor(d%3600/60);
    var parts=[];
    var isEn=document.body.classList.contains('lang-en');
    if(dd>0)parts.push(dd+(isEn?' days':' 天'));
    parts.push(hh+(isEn?'h':'时')+mm+(isEn?'m':'分'));
    el.textContent=(isEn?'Elapsed: ':'已过 ')+parts.join('')+(isEn?', recommend refreshing weekly':'，建议每周刷新一次');
    if(dd>=7)el.style.color='#c62828';
  }}
  tick();setInterval(tick,60000);
}})();
function toggleLang(){{
  var b=document.body, btn=document.getElementById('lang-btn');
  if(b.classList.contains('lang-zh')){{
    b.classList.remove('lang-zh');b.classList.add('lang-en');btn.textContent='中文';
  }}else{{
    b.classList.remove('lang-en');b.classList.add('lang-zh');btn.textContent='EN';
  }}
}}

// ═══ BS Put pricing in JS ═══
function normCdf(x) {{
  var a1=0.254829592, a2=-0.284496736, a3=1.421413741, a4=-1.453152027, a5=1.061405429, p=0.3275911;
  var sign = x < 0 ? -1 : 1;
  x = Math.abs(x) / Math.sqrt(2);
  var t = 1.0 / (1.0 + p * x);
  var y = 1.0 - (((((a5*t + a4)*t) + a3)*t + a2)*t + a1)*t * Math.exp(-x*x);
  return 0.5 * (1.0 + sign * y);
}}
function bsPut(S, K, T, r, sigma) {{
  if (T <= 0) return Math.max(K - S, 0);
  var d1 = (Math.log(S/K) + (r + 0.5*sigma*sigma)*T) / (sigma*Math.sqrt(T));
  var d2 = d1 - sigma*Math.sqrt(T);
  return K*Math.exp(-r*T)*normCdf(-d2) - S*normCdf(-d1);
}}
function bsPutAuto(S, K, T, r) {{
  return bsPut(S, K, T, r, interpIv(K));
}}

// ═══ Parameters as JS globals ═══
var P = {{
  fv: {fv}, ibex0: {ibex_now}, psi: {psi_now},
  betaFI: {BETA_FUND_IBEX}, betaFP: {BETA_FUND_PSI}, betaIP: {BETA_IBEX_PSI},
  iv: {IBEX_IMPLIED_VOL}, r: {ECB_RATE}, initInv: {INITIAL_INV}, psiEntry: {PSI20_ENTRY_ACT},
  avgCR: {avg_crash_ratio}, fundNav: {fund_nav}, fundUnits: {FUND_UNITS},
  meffPts: {str([dict(strike=p['strike'], iv=p['iv']) for p in _meff_iv_points]) if _meff_iv_points else '[]'}
}};
function interpIv(K) {{
  var pts = P.meffPts;
  if (!pts || pts.length < 2) return P.iv;
  pts.sort(function(a,b){{return a.strike-b.strike}});
  if (K <= pts[0].strike) {{
    var slope = (pts[1].iv - pts[0].iv) / (pts[1].strike - pts[0].strike);
    return Math.max(0.10, Math.min(0.50, pts[0].iv + slope * (K - pts[0].strike)));
  }}
  if (K >= pts[pts.length-1].strike) {{
    var n = pts.length;
    var slope = (pts[n-1].iv - pts[n-2].iv) / (pts[n-1].strike - pts[n-2].strike);
    return Math.max(0.08, Math.min(0.50, pts[n-1].iv + slope * (K - pts[n-1].strike)));
  }}
  for (var i = 0; i < pts.length - 1; i++) {{
    if (K >= pts[i].strike && K <= pts[i+1].strike) {{
      var t = (K - pts[i].strike) / (pts[i+1].strike - pts[i].strike);
      return pts[i].iv * (1 - t) + pts[i+1].iv * t;
    }}
  }}
  return P.iv;
}}

// ═══ Fetch all live prices ═══
function fetchAll() {{
  var st = document.getElementById('refresh-status');
  var btn = document.getElementById('refresh-btn');
  var isEn = document.body.classList.contains('lang-en');
  st.textContent = isEn ? 'Fetching...' : '获取中...';
  st.style.color = '#888';
  btn.disabled = true;
  // 1) 尝试同域 prices.json（GitHub Pages每日更新，无CORS问题）
  fetch('prices.json?t=' + Date.now())
    .then(function(r) {{ if (!r.ok) throw new Error(r.status); return r.json(); }})
    .then(function(d) {{ applyPrices(d, st, btn, isEn); }})
    .catch(function() {{
      // 2) 回退：尝试 Yahoo Finance via CORS proxy
      fetch('https://corsproxy.io/?url=' + encodeURIComponent('https://query1.finance.yahoo.com/v8/finance/chart/%5EIBEX?range=1d&interval=1d'))
        .then(function(r) {{ return r.json(); }})
        .then(function(d) {{
          var price = d.chart.result[0].meta.regularMarketPrice;
          document.getElementById('ibex-input').value = Math.round(price);
          var el = document.querySelector('[data-live="ibex"]');
          if (el) el.textContent = Math.round(price).toLocaleString();
          st.textContent = 'IBEX=' + Math.round(price) + ' \u2713';
          st.style.color = '#2e7d32';
          btn.disabled = false;
          recalcAll();
        }})
        .catch(function() {{
          st.innerHTML = isEn
            ? 'Auto-refresh unavailable. Run <code>python3 hedge_final.py</code> locally.'
            : '自动刷新不可用，请本地运行 <code>python3 hedge_final.py</code>';
          st.style.color = '#c62828';
          btn.disabled = false;
        }});
    }});
}}
function applyPrices(d, st, btn, isEn) {{
  if (d.ibex) {{
    document.getElementById('ibex-input').value = Math.round(d.ibex);
    var el = document.querySelector('[data-live="ibex"]');
    if (el) el.textContent = Math.round(d.ibex).toLocaleString();
  }}
  if (d.psi) {{
    var el = document.querySelector('[data-live="psi"]');
    if (el) el.textContent = Math.round(d.psi).toLocaleString();
  }}
  if (d.estx) {{
    var el = document.querySelector('[data-live="estx"]');
    if (el) el.textContent = Math.round(d.estx).toLocaleString();
  }}
  var parts = [];
  if (d.ibex) parts.push('IBEX=' + Math.round(d.ibex));
  if (d.psi) parts.push('PSI=' + Math.round(d.psi));
  if (d.estx) parts.push('ESTX=' + Math.round(d.estx));
  if (d.ts) parts.push(d.ts);
  st.textContent = parts.join(' | ') + ' \u2713';
  st.style.color = '#2e7d32';
  btn.disabled = false;
  recalcAll();
}}

// ═══ Helper functions ═══
function setText(id, val) {{
  var el = document.getElementById(id);
  if (el) el.textContent = val;
}}
function setHtml(id, val) {{
  var el = document.getElementById(id);
  if (el) el.innerHTML = val;
}}

// ═══ Build compact option chain (key strikes only) ═══
function buildChain(ibex, K, K90) {{
  var body = document.getElementById('chain-body');
  if (!body) return;
  // Key strikes: 85%, 88%, 90%, 93%, 95%, ATM, 103% + ensure K and K90 are included
  var keyPcts = [0.85, 0.88, 0.90, 0.93, 0.95, 1.00, 1.03];
  var strikes = {{}};
  keyPcts.forEach(function(p) {{ var s = Math.round(ibex * p / 50) * 50; strikes[s] = true; }});
  strikes[K] = true;
  strikes[K90] = true;
  var sorted = Object.keys(strikes).map(Number).sort(function(a,b){{return a-b}});
  var rows = '';
  sorted.forEach(function(s) {{
    var pct = ((s / ibex - 1) * 100).toFixed(1);
    var price = bsPutAuto(ibex, s, 1.0, P.r);
    var planA = '', planB = '';
    if (s === K) planA = '\u00d78';
    if (s === K90) {{ planA += (planA ? ' + ' : '') + '\u00d720'; planB = '\u00d724'; }}
    var iv = interpIv(s);
    var hl = (s === K || s === K90) ? ' style="background:#f0fff0;font-weight:700"' : '';
    rows += '<tr' + hl + '><td>' + s.toLocaleString() + '</td><td>' + pct + '%</td>';
    rows += '<td style="font-size:11px;color:#888">' + (iv*100).toFixed(1) + '%</td>';
    rows += '<td>\u20AC' + Math.round(price).toLocaleString() + '</td>';
    rows += '<td style="color:#2e7d32;font-weight:700">' + planA + '</td>';
    rows += '<td style="color:#e65100;font-weight:700">' + planB + '</td></tr>';
  }});
  body.innerHTML = rows;
  // Also rebuild full chain if visible
  if (document.getElementById('full-chain-box').style.display !== 'none') buildFullChain(ibex, K, K90);
}}
function toggleFullChain() {{
  var box = document.getElementById('full-chain-box');
  var link = document.getElementById('chain-toggle');
  var isEn = document.body.classList.contains('lang-en');
  if (box.style.display === 'none') {{
    box.style.display = 'block';
    buildFullChain(parseFloat(document.getElementById('ibex-input').value),
      Math.round(parseFloat(document.getElementById('ibex-input').value)/50)*50,
      Math.round(parseFloat(document.getElementById('ibex-input').value)*0.9/50)*50);
    link.textContent = isEn ? 'Collapse' : '收起';
  }} else {{
    box.style.display = 'none';
    link.textContent = isEn ? 'Show full chain' : '展开完整链';
  }}
}}
function buildFullChain(ibex, K, K90) {{
  var body = document.getElementById('full-chain-body');
  if (!body) return;
  var rows = '';
  var lo = Math.round(ibex * 0.80 / 50) * 50;
  var hi = Math.round(ibex * 1.05 / 50) * 50;
  for (var s = lo; s <= hi; s += 50) {{
    var pct = ((s / ibex - 1) * 100).toFixed(1);
    var price = bsPutAuto(ibex, s, 1.0, P.r);
    var hl = (s === K || s === K90) ? ' style="background:#f0fff0;font-weight:600"' : '';
    rows += '<tr' + hl + '><td>' + s.toLocaleString() + '</td><td>' + pct + '%</td><td>\u20AC' + Math.round(price).toLocaleString() + '</td></tr>';
  }}
  body.innerHTML = rows;
}}

// ═══ Update config comparison table ═══
function updateConfigTable(ibex, K, K90, pAtm, pOtm) {{
  var body = document.getElementById('config-tbody');
  if (!body) return;
  var isEn = document.body.classList.contains('lang-en');
  var configs = [
    {{label: isEn ? 'Pure ATM x16' : 'ATM x16', desc: isEn ? 'Current: all ATM' : 'All ATM', legs: [{{n:16,k:K,p:pAtm}}], rec: false}},
    {{label: 'ATM x8 + 90%OTM x20', desc: isEn ? 'Mixed: floor + crash protection' : 'Mixed', legs: [{{n:8,k:K,p:pAtm}},{{n:20,k:K90,p:pOtm}}], rec: true}},
    {{label: 'ATM x4 + 90%OTM x30', desc: isEn ? 'Aggressive: heavy crash protection' : 'Aggressive', legs: [{{n:4,k:K,p:pAtm}},{{n:30,k:K90,p:pOtm}}], rec: false}},
    {{label: isEn ? 'Pure 90%OTM x24' : '90%OTM x24', desc: isEn ? 'Budget: crash-only' : 'Budget', legs: [{{n:24,k:K90,p:pOtm}}], rec: false}}
  ];
  var rows = '';
  configs.forEach(function(c) {{
    var prem = 0;
    c.legs.forEach(function(l) {{ prem += l.p * l.n; }});
    prem = Math.round(prem);
    var st = c.rec ? ' style="background:#f0fff0;font-weight:600"' : '';
    var tag = c.rec ? ' <span style="color:#2e7d32;font-size:11px">[' + (isEn?'Rec.':'Rec.') + ']</span>' : '';
    rows += '<tr' + st + '>';
    rows += '<td style="text-align:left">' + c.label + tag + '<br><span style="font-size:10px;color:#888">' + c.desc + '</span></td>';
    rows += '<td>&euro;' + prem.toLocaleString() + '<br><span style="font-size:10px;color:#888">' + (prem/P.fv*100).toFixed(2) + '%/' + (isEn?'yr':'yr') + '</span></td>';
    rows += '<td>&euro;' + (prem*5).toLocaleString() + '<br><span style="font-size:10px;color:#888">' + (prem*5/P.fv*100).toFixed(1) + '%</span></td>';
    [5,10,20,30].forEach(function(drop) {{
      var ibexDrop = ibex * (1 - drop/100);
      var pay = 0;
      c.legs.forEach(function(l) {{ pay += Math.max(l.k - ibexDrop, 0) * l.n; }});
      pay = Math.round(pay);
      var color = pay > 0 ? '#2e7d32' : '#ccc';
      var fw = drop >= 20 ? ';font-weight:' + (drop >= 30 ? '700' : '600') : '';
      rows += '<td style="color:' + color + fw + '">&euro;' + pay.toLocaleString() + '</td>';
    }});
    rows += '</tr>';
  }});
  body.innerHTML = rows;
}}

// ═══ Update PSI scenario table ═══
function updatePsiTable(tbodyId, ibex, legs, prem) {{
  var body = document.getElementById(tbodyId);
  if (!body) return;
  var targets = [8500, 8000, 7500, 7000, 6000];
  var rows = '';
  targets.forEach(function(psiT) {{
    var psiDrop = (psiT - P.psi) / P.psi;
    var ibexEst = ibex * (1 + P.betaIP * psiDrop);
    var fundEst = Math.round(P.fv * (1 + P.betaFP * psiDrop));
    var fundLoss = P.fv - fundEst;
    var putPay = 0;
    legs.forEach(function(l) {{ putPay += Math.max(l.k - ibexEst, 0) * l.n; }});
    putPay = Math.round(putPay);
    var net = fundEst + putPay - prem;
    var cov = fundLoss > 0 ? Math.round(putPay / fundLoss * 100) : 0;
    var cc = cov >= 80 ? '#2e7d32' : (cov >= 40 ? '#e65100' : '#c62828');
    var dropPct = ((psiT - P.psi) / P.psi * 100).toFixed(0);
    rows += '<tr>';
    rows += '<td style="font-weight:700">' + psiT.toLocaleString() + ' <span style="font-size:10px;color:#888">(' + (dropPct>0?'+':'') + dropPct + '%)</span></td>';
    rows += '<td>&euro;' + fundEst.toLocaleString() + '</td>';
    rows += '<td style="color:#c62828;font-weight:600">-&euro;' + fundLoss.toLocaleString() + '</td>';
    rows += '<td style="color:#2e7d32;font-weight:600">+&euro;' + putPay.toLocaleString() + '</td>';
    rows += '<td style="font-weight:700">&euro;' + net.toLocaleString() + '</td>';
    rows += '<td style="color:' + cc + ';font-weight:700">' + cov + '%</td>';
    rows += '</tr>';
  }});
  body.innerHTML = rows;
}}

// ═══ Update entry scenario ═══
function updateEntryScenario(ibex, K, K90, pAtm, pOtm, premA, premB) {{
  var psiDrop = (P.psiEntry - P.psi) / P.psi;
  var ibexEst = Math.round(ibex * (1 + P.betaIP * psiDrop));
  var fundEst = Math.round(P.fv * (1 + P.betaFP * psiDrop));
  var fundLoss = P.fv - fundEst;
  var gainNow = P.fv - P.initInv;

  var paPay = Math.round(Math.max(K - ibexEst, 0) * 8 + Math.max(K90 - ibexEst, 0) * 20);
  var paNet = fundEst + paPay - premA;
  var paKept = gainNow > 0 ? Math.round((paNet - P.initInv) / gainNow * 100) : 0;
  var noHedgeKept = gainNow > 0 ? Math.round((fundEst - P.initInv) / gainNow * 100) : 0;

  var pbPay = Math.round(Math.max(K90 - ibexEst, 0) * 24);
  var pbNet = fundEst + pbPay - premB;
  var pbKept = gainNow > 0 ? Math.round((pbNet - P.initInv) / gainNow * 100) : 0;

  setText('dyn-es-ibex', ibexEst.toLocaleString());
  setText('dyn-es-fund', '\u20AC' + fundEst.toLocaleString());
  setText('dyn-es-loss', '\u20AC' + fundLoss.toLocaleString());
  setText('dyn-es-nohedge', noHedgeKept + '%');

  setText('dyn-es-pa-pay', '+\u20AC' + paPay.toLocaleString());
  setText('dyn-es-pa-prem', '\u2212\u20AC' + premA.toLocaleString());
  setText('dyn-es-pa-net', '\u20AC' + paNet.toLocaleString());
  setText('dyn-es-pa-gain', '\u20AC' + (paNet - P.initInv).toLocaleString());
  setText('dyn-es-pa-kept', paKept + '%');

  setText('dyn-es-pb-pay', '+\u20AC' + pbPay.toLocaleString());
  setText('dyn-es-pb-prem', '\u2212\u20AC' + premB.toLocaleString());
  setText('dyn-es-pb-net', '\u20AC' + pbNet.toLocaleString());
  setText('dyn-es-pb-gain', '\u20AC' + (pbNet - P.initInv).toLocaleString());
  setText('dyn-es-pb-kept', pbKept + '%');
  setText('dyn-es-pa-kept2', paKept);
  setText('dyn-es-pa-diff', (paKept - pbKept) + '');
  setText('dyn-es-prem-diff', '\u20AC' + (premA - premB).toLocaleString());
}}

// ═══ Update section 7 operation step prices ═══
function updateStepPrices(ibex, K, K90, pAtm, pOtm, premA, premB) {{
  setText('dyn-s7-atm-k', K.toLocaleString());
  setText('dyn-s7-otm-k', K90.toLocaleString());
  setText('dyn-s7-atm-price', '\u20AC' + Math.round(pAtm).toLocaleString());
  setText('dyn-s7-atm-total', '\u20AC' + Math.round(pAtm * 8).toLocaleString());
  setText('dyn-s7-otm-price', '\u20AC' + Math.round(pOtm).toLocaleString());
  setText('dyn-s7-otm-total', '\u20AC' + Math.round(pOtm * 20).toLocaleString());
  setText('dyn-s7-prem-a', '\u20AC' + premA.toLocaleString());
  setText('dyn-s7-prem-b', '\u20AC' + premB.toLocaleString());
  setText('dyn-s7-otm-total-b', '\u20AC' + premB.toLocaleString());
  setText('dyn-s7-otm-price-b', '\u20AC' + Math.round(pOtm).toLocaleString());
  setText('dyn-s7-otm-k-b', K90.toLocaleString());
  var rollAtm = Math.round(pAtm * 8 * 0.4);
  var rollOtm = Math.round(pOtm * 20 * 0.4);
  setText('dyn-s7-roll-full', '\u20AC' + (rollAtm + rollOtm).toLocaleString());
  setText('dyn-s7-roll-full-yr', '\u20AC' + ((rollAtm + rollOtm) * 2).toLocaleString());
  setText('dyn-s7-roll-split', '\u20AC' + rollAtm.toLocaleString());
  setText('dyn-s7-roll-split-yr', '\u20AC' + (rollAtm * 2).toLocaleString());
  setText('dyn-s7-roll-save', '\u20AC' + (rollOtm * 2).toLocaleString());
  setText('dyn-s7-roll-save-5yr', '\u20AC' + (rollOtm * 2 * 5).toLocaleString());
  setText('dyn-s7-trigger-a', Math.round(ibex * 1.10).toLocaleString());
  setText('dyn-s7-trigger-b', Math.round(ibex * 1.20).toLocaleString());
  var rollB = Math.round(pOtm * 24 * 0.4);
  setText('dyn-s7-roll-b', '\u20AC' + rollB.toLocaleString());
  setText('dyn-s7-roll-b-total', '\u20AC' + (premB + Math.round(rollB * 0.5)).toLocaleString());
}}

// ═══ Update payoff chart ═══
function updatePayoffChart(divId, ibex, K, K90, legs, prem) {{
  var el = document.getElementById(divId);
  if (!el) return;
  var xs = [], yFund = [], yHedged = [], yPut = [];
  for (var psi = 5000; psi <= 11000; psi += 100) {{
    var psiDrop = (psi - P.psi) / P.psi;
    var fundEst = P.fv * (1 + P.betaFP * psiDrop);
    var ibexEst = ibex * (1 + P.betaIP * psiDrop);
    var putPay = 0;
    legs.forEach(function(l) {{ putPay += Math.max(l.k - ibexEst, 0) * l.n; }});
    xs.push(psi);
    yFund.push(Math.round(fundEst));
    yHedged.push(Math.round(fundEst + putPay - prem));
    yPut.push(Math.round(putPay));
  }}
  var isEn = document.body.classList.contains('lang-en');
  Plotly.react(divId, [
    {{x: xs, y: yFund, name: isEn ? 'Fund (no hedge)' : 'Fund (no hedge)', line: {{color:'#c62828',width:2,dash:'dash'}}}},
    {{x: xs, y: yHedged, name: isEn ? 'Fund + Put hedge' : 'Fund+Put', line: {{color:'#2e7d32',width:3}}}},
    {{x: xs, y: yPut, name: isEn ? 'Put payout' : 'Put payout', line: {{color:'#1565c0',width:2,dash:'dot'}}, yaxis: 'y2'}}
  ], {{
    template: 'plotly_white', height: 360,
    xaxis: {{title: isEn ? 'PSI20 Level' : 'PSI20'}},
    yaxis: {{title: isEn ? 'Fund Value (EUR)' : 'Fund Value (EUR)', tickformat: ','}},
    yaxis2: {{title: isEn ? 'Put Payout (EUR)' : 'Put Payout (EUR)', overlaying: 'y', side: 'right', tickformat: ',', showgrid: false}},
    legend: {{x:0.01,y:0.01,bgcolor:'rgba(255,255,255,0.9)'}},
    margin: {{t:10,b:50,l:80,r:80}}, hovermode: 'x unified'
  }}, {{responsive: true}});
}}

// ═══ Master recalculation ═══
function recalcAll() {{
  var ibex = parseFloat(document.getElementById('ibex-input').value);
  if (isNaN(ibex) || ibex < 5000 || ibex > 30000) return;

  var K = Math.round(ibex / 50) * 50;
  var K90 = Math.round(ibex * 0.9 / 50) * 50;
  var pAtm = bsPutAuto(ibex, K, 1.0, P.r);
  var pOtm = bsPutAuto(ibex, K90, 1.0, P.r);

  var premA = Math.round(pAtm * 8 + pOtm * 20);
  var premB = Math.round(pOtm * 24);

  // Update option chain
  buildChain(ibex, K, K90);

  // Update Plan A rec box
  setText('dyn-pa-atm-k', K.toLocaleString());
  setText('dyn-pa-otm-k', K90.toLocaleString());
  setText('dyn-pa-prem', '\u20AC' + premA.toLocaleString());
  setText('dyn-pa-prem-pct', (premA/P.fv*100).toFixed(2) + '%');
  setText('dyn-pa-5yr', '\u20AC' + (premA*5).toLocaleString());

  // Update Plan B rec box
  setText('dyn-pb-otm-k', K90.toLocaleString());
  setText('dyn-pb-prem', '\u20AC' + premB.toLocaleString());
  setText('dyn-pb-prem-pct', (premB/P.fv*100).toFixed(2) + '%');
  setText('dyn-pb-5yr', '\u20AC' + (premB*5).toLocaleString());
  setText('dyn-pb-save', Math.round((1-premB/premA)*100) + '%');

  // Update config comparison table
  updateConfigTable(ibex, K, K90, pAtm, pOtm);

  // Update PSI scenario tables
  updatePsiTable('psi-tbody-a', ibex, [{{n:8,k:K}}, {{n:20,k:K90}}], premA);
  updatePsiTable('psi-tbody-b', ibex, [{{n:24,k:K90}}], premB);

  // Update entry scenario
  updateEntryScenario(ibex, K, K90, pAtm, pOtm, premA, premB);

  // Update section 7 prices
  updateStepPrices(ibex, K, K90, pAtm, pOtm, premA, premB);

  // Update payoff charts
  updatePayoffChart('c5', ibex, K, K90, [{{n:8,k:K}},{{n:20,k:K90}}], premA);
  updatePayoffChart('c5b', ibex, K, K90, [{{n:24,k:K90}}], premB);
}}

// Build initial option chain on page load
buildChain({ibex_now}, {rec['K']}, {K_90});
</script></body></html>"""

    html = html.replace('__C5B__',planb_chart).replace('__C1__',c1).replace('__C2__',c2).replace('__C5__',c5)
    html = html.replace('__ZOOM__', '['+','.join(c if c else 'null' for c in zooms)+']')
    return html


def main():
    from datetime import datetime
    print('获取实时价格...')
    prices = fetch_live_prices()
    fund_nav = prices.get('fund', 17.15)  # fallback
    _fund_is_fallback = 'fund' not in prices
    psi_now = prices.get('psi', 8862)
    ibex_now = prices.get('ibex', 17062)
    estx_now = prices.get('estx', 6138)
    fund_value = round(fund_nav * FUND_UNITS)
    gen_time = datetime.now().strftime('%Y-%m-%d %H:%M')
    fund_src = prices.get('fund_src', 'Yahoo Finance')
    # 计算NAV滞后天数（工作日口径）
    fund_lag = 0
    fund_date_str = prices.get('fund_date', '?')
    if fund_date_str != '?':
        fund_dt = datetime.strptime(fund_date_str, '%Y-%m-%d').date()
        today = datetime.now().date()
        fund_lag = (today - fund_dt).days
    live = dict(fund_nav=fund_nav, fund_value=fund_value, psi=psi_now,
                ibex=ibex_now, estx=estx_now, gen_time=gen_time, fund_src=fund_src,
                fund_date=fund_date_str, fund_lag=fund_lag,
                _fund_is_fallback=_fund_is_fallback,
                psi_date=prices.get('psi_date','?'),
                ibex_date=prices.get('ibex_date','?'),
                estx_date=prices.get('estx_date','?'))
    print(f'  基金NAV=€{fund_nav:.2f}({fund_src}, {prices.get("fund_date","?")}) 市值=€{fund_value:,}')
    print(f'  PSI20={psi_now:,.0f} IBEX={ibex_now:,.0f} ESTX={estx_now:,.0f}')

    print('获取MEFF期权IV...')
    global _meff_iv_points, _meff_expiry, _meff_source
    meff_data = fetch_meff_iv()
    if meff_data:
        _meff_expiry, _meff_iv_points = pick_expiry(meff_data)
        if _meff_iv_points and len(_meff_iv_points) >= 2:
            _meff_source = 'MEFF'
            pts_str = ', '.join(f'K={p["strike"]:,.0f}→IV={p["iv"]*100:.1f}%' for p in _meff_iv_points)
            print(f'  MEFF {_meff_expiry}: {len(_meff_iv_points)}个数据点 [{pts_str}]')
        else:
            print(f'  MEFF数据不足，使用BS固定IV={IBEX_IMPLIED_VOL*100}%')
    else:
        print(f'  MEFF抓取失败，使用BS固定IV={IBEX_IMPLIED_VOL*100}%')
    live['meff_source'] = _meff_source
    live['meff_expiry'] = _meff_expiry or 'N/A'
    live['meff_points'] = _meff_iv_points

    print('获取MEFF实时期权链...')
    global _meff_live_chain
    chain_data = fetch_meff_live_chain(ibex_now)
    if chain_data and chain_data.get('chains'):
        _meff_live_chain = chain_data
        live['meff_chain'] = chain_data
        total = sum(len(v) for v in chain_data['chains'].values())
        put_exps = [k for k in chain_data['chains'] if k.startswith('OPE')]
        print(f'  {len(put_exps)}个到期月, 共{total}个报价')
        # 找最长的Put到期月，计算真实T
        best_code, best_date, best_days = None, None, 0
        for code in put_exps:
            try:
                exp = datetime.strptime(code[3:], '%Y%m%d')
                d = (exp - datetime.now()).days
                if d > best_days:
                    best_code, best_date, best_days = code, exp, d
            except ValueError:
                continue
        if best_code:
            live['best_put_expiry_code'] = best_code
            live['best_put_expiry_date'] = best_date.strftime('%Y-%m-%d')
            live['best_put_expiry_label'] = chain_data.get('labels', {}).get(best_code, best_date.strftime('%d/%m/%Y'))
            live['best_put_T'] = round(best_days / 365, 3)
            print(f'  最长Put到期: {live["best_put_expiry_label"]} (T={live["best_put_T"]:.2f}年, {best_days}天)')
    else:
        print('  实时期权链抓取失败')
        live['meff_chain'] = {}

    # 将live NAV写入CSV，避免数据缺口
    update_csv_with_live(live)

    print('加载历史数据...')
    fund_df, ibex_df, psi_df = load_data(live)
    print(f'  基金{len(fund_df)}条 IBEX{len(ibex_df)}条 PSI{len(psi_df)}条')

    print('分析...')
    res = analyze(fund_df, ibex_df, psi_df, live)
    ev = res['events']
    print(f'  {len(ev)}次急跌({sum(1 for e in ev if e["sync"])}次同步) 持仓期{sum(1 for e in ev if e["in_hold"])}次')

    print('生成报告...')
    html = generate_html(fund_df, psi_df, res, live)
    out = os.path.join(DATA_DIR, 'hedge_final.html')
    with open(out, 'w', encoding='utf-8') as f: f.write(html)
    print(f'→ {out}')
    # 输出 prices.json 供页面刷新按钮使用（同域，无CORS问题）
    import json as _json
    prices_json = _json.dumps(dict(
        ibex=ibex_now, psi=psi_now, estx=estx_now,
        fund_nav=fund_nav, fund_value=fund_value,
        ts=gen_time, fund_src=fund_src,
        fund_date=prices.get('fund_date','?'),
        psi_date=prices.get('psi_date','?'),
        ibex_date=prices.get('ibex_date','?'),
        estx_date=prices.get('estx_date','?'),
    ))
    docs_dir = os.path.join(DATA_DIR, 'docs')
    if os.path.isdir(docs_dir):
        pj = os.path.join(docs_dir, 'prices.json')
        with open(pj, 'w') as f: f.write(prices_json)
        print(f'→ {pj}')
    import subprocess, platform
    if platform.system() == 'Darwin': subprocess.run(['open', out])

if __name__ == '__main__': main()
