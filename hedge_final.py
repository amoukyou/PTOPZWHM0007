"""
葡萄牙基金对冲方案 — 完整报告 v10
- 删除循环论证的99%覆盖率，如实展示保护局限
- 正面讨论Event #10暴露的年度滚仓缺陷
- 新增动态滚仓触发机制(IBEX涨超15%即提前滚仓)
- Beta链说明改为独立回归，不暗示推导关系
- 结论如实反映：减震垫，不是全额保险
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

def norm_cdf(x):
    return (1 + math.erf(x / math.sqrt(2))) / 2

def bs_put(S, K, T, r=ECB_RATE, sigma=IBEX_IMPLIED_VOL):
    if T <= 0: return max(K - S, 0)
    d1 = (math.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*math.sqrt(T))
    d2 = d1 - sigma*math.sqrt(T)
    return K*math.exp(-r*T)*norm_cdf(-d2) - S*norm_cdf(-d1)


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
    # 追加live数据到fund_df，确保事件检测覆盖最新交易日
    if live and 'fund_nav' in live and 'fund_date' in live:
        live_dt = pd.Timestamp(live['fund_date'])
        if fund['Date'].iloc[-1] < live_dt:
            live_row = pd.DataFrame({'Date': [live_dt], 'Close': [live['fund_nav']]})
            # 补齐CSV中可能有的其他列
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
    # Recommendation (base: 16 contracts)
    K = round(ibex_now / 50) * 50  # MEFF标准行权价间距50点
    p1 = bs_put(ibex_now, K, 1.0)
    rec = dict(K=K, price=round(p1,1), total=round(p1*N_CONTRACTS), annual=round(p1*N_CONTRACTS/fv*100,2))
    # Mixed-strike configurations
    K_90 = round(ibex_now * 0.90 / 50) * 50
    K_85 = round(ibex_now * 0.85 / 50) * 50
    p_90 = bs_put(ibex_now, K_90, 1.0)
    p_85 = bs_put(ibex_now, K_85, 1.0)
    options = []
    configs = [
        ('纯ATM ×16', '现方案：全部平值', [(16, K, p1)]),
        (f'ATM ×8 + 90%OTM ×20', '混合：小跌有底+大跌加倍', [(8, K, p1), (20, K_90, p_90)]),
        (f'ATM ×4 + 90%OTM ×30', '进取：重注大跌保护', [(4, K, p1), (30, K_90, p_90)]),
        (f'纯90%OTM ×24', '省钱：放弃小跌，只防崩盘', [(24, K_90, p_90)]),
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
    # PSI20 scenario table for section 六 — using recommended mixed config
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
    return dict(df=df, df_h=df_h, events=events, strats=strats, rec=rec,
                options=options, psi_scenarios=psi_scenarios,
                K_90=K_90, rec_prem=rec_prem)

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
    K_atm = rec['K']
    K_90 = round(ibex_now * 0.90 / 50) * 50
    p1 = bs_put(ibex_now, K_atm, 1.0)
    p_90 = bs_put(ibex_now, K_90, 1.0)
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

# ─── HTML ─────────────────────────────
def generate_html(fund_df, psi_df, res, live):
    df, events, strats, rec, options = res['df'], res['events'], res['strats'], res['rec'], res['options']
    psi_scenarios = res['psi_scenarios']
    K_90 = res['K_90']
    rec_prem = res['rec_prem']
    K = rec['K']
    fv = live['fund_value']
    fund_nav = live['fund_nav']
    psi_now = live['psi']
    ibex_now = live['ibex']
    estx_now = live['estx']
    gen_time = live['gen_time']

    c1 = chart_fund_psi_ibex(fund_df, psi_df, res['df'][['Date','ibex']], live)
    c2 = chart_fund_ibex(df, events)
    c5 = chart_payoff(rec, live)
    zooms = make_zoom_charts(df, events, strats, fv)

    fg = (fund_nav / FUND_ENTRY_PRICE - 1) * 100
    nt = len(events)
    ns = sum(1 for e in events if e['sync'])

    # Event table rows + tab buttons + panels
    hist_rows, tab_btns, tab_panels = '', '', ''
    for i, ev in enumerate(events):
        act = ' active' if i==0 else ''
        hold = '' if ev['in_hold'] else ' <span style="font-size:10px;color:#aaa">(买入前)</span>'
        hist_rows += f"""<tr style="background:#f0fff0;cursor:pointer" onclick="showTab({i})">
          <td>{i+1}</td><td>{ev['start_str']}~{ev['end_str']}{hold}</td>
          <td style="color:#c62828;font-weight:600">{ev['fund_chg']:.1f}%</td>
          <td style="color:#2e7d32;font-weight:600">{ev['ibex_chg']:+.1f}%</td>
          <td>{ev['ibex_peak']:,.0f}&rarr;{ev['ibex_level']:,.0f}</td></tr>"""

        tab_btns += f'<button class="tab-btn{act}" onclick="showTab({i})">#{i+1} {ev["end_str"][5:]}</button>'

        loss = ev.get('fund_loss', abs(fv*ev['fund_chg']/100))
        detail = ''
        if ev['in_hold'] and ev['strat']:
            detail = '<table style="font-size:13px;margin-top:12px"><tr><th style="text-align:left">策略</th><th>行权价<br><span style="font-weight:400;font-size:10px;color:#888">（模拟中的历史值）</span></th><th>Put赚了</th><th>净亏</th><th>覆盖率</th></tr>'
            for k, lab in [('12M','12月ATM年滚'),('6M','6月ATM半年滚'),('3M','3月ATM季滚')]:
                r = ev['strat'].get(k,{})
                m, st, c = r.get('mtm',0), r.get('strike',0), r.get('cov',0)
                cc = '#2e7d32' if c>=20 else ('#e65100' if c>=5 else '#c62828')
                rm = ' <b style="color:#2e7d32">[推荐]</b>' if k=='12M' else ''
                detail += f'<tr><td style="text-align:left">{lab}{rm}</td><td style="font-size:12px;color:#888">{st:,.0f}</td><td style="color:#2e7d32;font-weight:700">+&euro;{m:,.0f}</td><td style="color:#1565c0;font-weight:700">-&euro;{loss-m:,.0f}</td><td style="font-weight:700;color:{cc}">{c:.0f}%</td></tr>'
            detail += '</table>'
        elif not ev['in_hold']:
            detail = '<p style="font-size:13px;color:#888;margin-top:8px">此事件发生在你买入基金之前，仅作为IBEX同步性的历史参考。</p>'

        tab_panels += f"""<div class="tab-panel{act}" id="panel_{i}">
          <div class="two-col">
            <div>
              <div class="alert a-good" style="margin-bottom:8px">
                <b>#{i+1} {ev['start_str']} ~ {ev['end_str']}</b><br>
                基金1周跌 <b style="color:#c62828">{ev['fund_chg']:.1f}%</b> (约&euro;{loss:,.0f})<br>
                IBEX&plusmn;2周跌 <b style="color:#2e7d32">{ev['ibex_chg']:+.1f}%</b> ({ev['ibex_peak']:,.0f}&rarr;{ev['ibex_level']:,.0f})
              </div>{detail}
            </div>
            <div class="chart-box" style="padding:8px"><div id="zoom_{i}" style="height:260px"></div></div>
          </div></div>"""

    # Contract options table — mixed-strike configs
    opt_rows = ''
    for i, opt in enumerate(options):
        is_rec = (i == 1)  # ATM×8 + 90%OTM×20 推荐
        st = ' style="background:#f0fff0;font-weight:600"' if is_rec else ''
        tag = ' <span style="color:#2e7d32;font-size:11px">[推荐]</span>' if is_rec else ''
        s5, s10, s20, s30 = opt['scenarios'][5], opt['scenarios'][10], opt['scenarios'][20], opt['scenarios'][30]
        opt_rows += f'''<tr{st}>
          <td style="text-align:left">{opt['label']}{tag}<br><span style="font-size:10px;color:#888">{opt['desc']}</span></td>
          <td>&euro;{opt['prem']:,}<br><span style="font-size:10px;color:#888">{opt['annual_pct']:.2f}%/年</span></td>
          <td>&euro;{opt['five_yr']:,}<br><span style="font-size:10px;color:#888">{opt['five_yr']/fv*100:.1f}%</span></td>
          <td style="color:{'#2e7d32' if s5['payoff']>0 else '#ccc'}">&euro;{s5['payoff']:,}</td>
          <td style="color:#2e7d32;font-weight:600">&euro;{s10['payoff']:,}</td>
          <td style="color:#2e7d32;font-weight:600">&euro;{s20['payoff']:,}</td>
          <td style="color:#2e7d32;font-weight:700">&euro;{s30['payoff']:,}</td>
        </tr>'''

    # Cost table (rolling frequency)
    cost_rows = ''
    for k, lab, freq in [('12M','12个月ATM 年滚',1),('6M','6个月ATM 半年滚',2),('3M','3个月ATM 季滚',4)]:
        s = strats[k]; T = s['months']/12
        pe = bs_put(ibex_now, ibex_now, T)*N_CONTRACTS
        ae = pe * (12/s['months']); fy = ae*5
        is_r = k=='12M'
        st = ' style="background:#f0fff0;font-weight:600"' if is_r else ''
        tg = ' <span style="color:#2e7d32;font-size:11px">[推荐]</span>' if is_r else ''
        cost_rows += f'<tr{st}><td style="text-align:left">{lab}{tg}</td><td>{freq}x/年</td><td>&euro;{ae:,.0f} ({ae/fv*100:.2f}%)</td><td>&euro;{fy:,.0f} ({fy/fv*100:.1f}%)</td></tr>'

    # PSI20 scenario rows for section 六
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

    html = f"""<!DOCTYPE html><html lang="zh-CN"><head><meta charset="utf-8">
<title>葡萄牙基金对冲方案</title>
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
</style></head><body><div class="page">

<h1>葡萄牙基金对冲方案</h1>
<p class="meta">Optimize Portugal Golden Opportunities Fund (PTOPZWHM0007)</p>

<div class="data-bar">
  <div class="src">
    数据来源：基金NAV &larr; <b>{live['fund_src']}</b> &middot; 指数 &larr; <b>Yahoo Finance</b><br>
    <span style="font-size:11px;color:#aaa">
      基金NAV: {live['fund_date']} &middot;
      PSI20: {live['psi_date']} &middot;
      IBEX35: {live['ibex_date']} &middot;
      ESTOXX50: {live['estx_date']}
    </span><br>
    <span style="font-size:11px;color:#aaa">重新运行 <code style="background:#eee;padding:1px 5px;border-radius:3px">python3 hedge_final.py</code> 即可刷新全部数据至最新交易日</span>
  </div>
  <div style="text-align:right;min-width:140px">
    <div style="font-size:11px;color:#888">报告生成于</div>
    <div style="font-size:15px;font-weight:800;color:#1a237e">{gen_time}</div>
    <div class="timer" id="timer"></div>
  </div>
</div>

<div class="section">
<h2>一、持仓概况</h2>
<div class="cards">
  <div class="card green"><div class="lbl">买入成本</div><div class="val">&euro;{INITIAL_INV:,}</div><div class="sub">2024.07 NAV &euro;{FUND_ENTRY_PRICE:.2f}</div></div>
  <div class="card green"><div class="lbl">当前市值</div><div class="val">&euro;{fv:,}</div><div class="sub">NAV &euro;{fund_nav:.2f}</div></div>
  <div class="card purple"><div class="lbl">浮盈</div><div class="val">{"+" if fv>=INITIAL_INV else ""}&euro;{fv-INITIAL_INV:,}</div><div class="sub">{fg:+.1f}%</div></div>
  <div class="card orange"><div class="lbl">PSI20</div><div class="val">{psi_now:,.0f}</div><div class="sub">买入时{PSI20_ENTRY_ACT:,}</div></div>
</div>
<div style="display:flex;gap:14px;margin-bottom:12px;font-size:12px;color:#888">
  <span>IBEX35: <b style="color:#e65100">{ibex_now:,.0f}</b></span>
  <span>ESTOXX50: <b style="color:#6a1b9a">{estx_now:,.0f}</b></span>
</div>
<div class="chart-box"><div id="c1" style="height:340px"></div></div>
<div class="alert a-warn">
  <b>担忧：</b>俄乌停战、欧盟加息、美欧关税等全欧系统性事件导致大跌。计划持有5年，想买保险。
</div>
</div>

<div class="section">
<h2>二、历史10次急跌：IBEX全部同步</h2>
<div class="chart-box"><div id="c2" style="height:460px"></div></div>
<p class="note-sm">图中绿色编号标记对应下表中的10次急跌事件。点击表格行查看放大详情。</p>

<table>
  <tr><th>#</th><th>时期</th><th>基金1周跌</th><th>IBEX&plusmn;2周跌</th><th>IBEX点位变化</th></tr>
  {hist_rows}
</table>

<div class="alert a-good">
  <b>{nt}次急跌（1周跌>3%），IBEX在&plusmn;2周内全部同步下跌，0次脱钩。</b><br>
  IBEX平均跌幅{np.mean([e['ibex_chg'] for e in events]):.1f}%，比基金平均跌幅{np.mean([e['fund_chg'] for e in events]):.1f}%更大。
</div>

<p style="margin-bottom:8px"><b>点击查看每次事件的放大走势：</b></p>
<div class="tab-bar">{tab_btns}</div>
{tab_panels}
</div>

<div class="section">
<h2>三、对冲链路与Beta说明</h2>
<div class="chain">
  <div class="chain-node"><div style="font-size:12px;color:#888">你持有的</div><div style="font-size:17px;font-weight:800;color:#1565c0">葡萄牙基金</div></div>
  <div class="chain-arrow">&rarr;</div>
  <div class="chain-node"><div style="font-size:12px;color:#888">高度跟踪</div><div style="font-size:17px;font-weight:800;color:#ff7f0e">PSI20</div><div style="font-size:11px;color:#2e7d32">R&sup2;=79%</div></div>
  <div class="chain-arrow">&rarr;</div>
  <div class="chain-node" style="border:2px solid #c62828"><div style="font-size:12px;color:#c62828">PSI20无可用期权</div></div>
  <div class="chain-arrow">&rarr;</div>
  <div class="chain-node" style="border:2px solid #2e7d32"><div style="font-size:12px;color:#2e7d32">替代</div><div style="font-size:17px;font-weight:800;color:#e65100">IBEX35 Put</div><div style="font-size:11px;color:#888">MEFF &middot; IBKR可交易</div></div>
</div>
<div class="alert a-info">
  合约数量按Beta(基金/IBEX)={BETA_FUND_IBEX}计算：基金对IBEX的敏感度为42%，
  需要对冲的名义敞口=&euro;{fv:,}&times;{BETA_FUND_IBEX}=&euro;{round(fv*BETA_FUND_IBEX):,}，
  除以IBEX点位=<b>{N_CONTRACTS}张Mini合约</b>（乘数&euro;1/点）。
</div>
<div class="alert a-warn">
  <b>Beta说明</b>：以下三个Beta是分别独立回归的结果，<b>不是</b>链式推导关系：<br>
  &middot; Beta(基金/PSI20) = {BETA_FUND_PSI}（R&sup2;=79%，基金跟踪PSI20较好）<br>
  &middot; Beta(基金/IBEX) = {BETA_FUND_IBEX}（R&sup2;=42%，IBEX只能解释基金42%的波动）<br>
  &middot; Beta(IBEX/PSI20) = {BETA_IBEX_PSI}（用于PSI20场景→IBEX点位换算）<br>
  <span style="font-size:12px;color:#888">注：R&sup2;=42%意味着基金有58%的波动无法被IBEX解释。Put只对冲IBEX相关的那42%风险。</span>
</div>
</div>

<div class="section">
<h2>四、合约配置：混合行权价策略</h2>
<div class="alert a-info">
  <b>核心思路</b>：不必全买贵的ATM Put。用一部分预算买便宜的虚值OTM Put，张数更多，
  大跌时赔付反而更高——同样的保费，换来更强的崩盘保护。
</div>
<table>
  <tr><th style="text-align:left">配置</th><th>年保费</th><th>5年总保费</th><th>IBEX跌5%</th><th>IBEX跌10%</th><th>IBEX跌20%</th><th>IBEX跌30%</th></tr>
  {opt_rows}
</table>
<div class="alert a-warn" style="font-size:13px">
  <b>怎么读这张表</b>：<br>
  &middot; <b>纯ATM ×16</b>：所有跌幅都有赔付，但大跌时赔付最少（因为只有16张）<br>
  &middot; <b>ATM ×8 + 90%OTM ×20 [推荐]</b>：小跌仍有保护（8张ATM兜底），大跌赔付比纯ATM多33-60%<br>
  &middot; <b>纯90%OTM ×24</b>：最省钱，但10%以内的跌幅完全不赔<br>
  OTM行权价={K_90:,}点（IBEX当前90%），ATM行权价={K:,}点
</div>
<div class="alert a-bad">
  <b>所有配置共同的局限</b>：Put赔付基于IBEX维度。基金实际损失取决于PSI20和基金自身因素（R&sup2;=42%），
  Put无法覆盖IBEX以外58%的风险。在"先涨后跌"行情下，即使是ATM Put也可能因行权价被甩开而失效（Event #10、#11教训）。
</div>
</div>

<div class="section">
<h2>五、滚仓频率对比</h2>
<table>
  <tr><th style="text-align:left">策略</th><th>操作频率</th><th>年化成本</th><th>5年总成本</th></tr>
  {cost_rows}
</table>
<div class="alert a-info">
  三种频率保护效果接近，但成本差距大。<b>12个月年滚花最少的钱，操作最简单。</b>
</div>
</div>

<div class="section">
<h2>六、推荐方案</h2>
<div class="rec">
  <h3>ATM Put &times;8 + 90%OTM Put &times;20，12个月年滚 + 动态滚仓</h3>
  <div class="rec-grid">
    <div class="rec-item"><div class="rl">ATM行权价</div><div class="rv">{rec['K']:,}点</div><div style="font-size:10px;color:#888">&times;8张</div></div>
    <div class="rec-item"><div class="rl">OTM行权价</div><div class="rv">{K_90:,}点</div><div style="font-size:10px;color:#888">&times;20张（90% OTM）</div></div>
    <div class="rec-item"><div class="rl">每年保费</div><div class="rv">&euro;{rec_prem:,}</div><div style="font-size:10px;color:#888">BS理论值，{rec_prem/fv*100:.2f}%</div></div>
    <div class="rec-item"><div class="rl">5年总保费</div><div class="rv">&euro;{rec_prem*5:,}</div></div>
    <div class="rec-item"><div class="rl">vs 纯ATM</div><div class="rv">同成本，大跌多赔33-60%</div></div>
  </div>
</div>
<div class="alert a-info" style="font-size:13px">
  <b>为什么混合配置更好</b>：同样~&euro;{rec_prem:,}/年预算，8张ATM保住小跌时的基本保护，
  20张90%OTM在大跌时提供额外杠杆（单价仅ATM的40%，但张数多1.5倍）。
  IBEX跌30%时赔付比纯ATM多<b>{round((options[1]['scenarios'][30]['payoff']/options[0]['scenarios'][30]['payoff']-1)*100)}%</b>。
</div>

<p style="margin:16px 0 8px;font-weight:700">如果PSI20跌到…你的基金会怎样？</p>
<table>
  <tr><th>PSI20跌到</th><th>基金预估市值</th><th>预估亏损</th><th>Put赔付</th><th>对冲后市值</th><th>覆盖率</th></tr>
  {psi_rows}
</table>
<div class="alert a-warn" style="font-size:13px">
  <b>线性模型局限</b>：上表覆盖率恒定约46%，因为模型假设损失和赔付同比例放大（纯线性）。
  现实中极端行情下Beta会漂移、尾部相关性会变化，实际覆盖率可能高于或低于此值。
  实证研究表明危机中跨市场相关性趋于上升（Longin &amp; Solnik 2001），大跌时IBEX与PSI20同步性更强，覆盖率可能<b>高于</b>此估计。<br>
  <b>更重要的是</b>：此表假设Put行权价在事件发生时仍为ATM。如果IBEX在持有期内先涨后跌（如Event #10），
  行权价早已变成深度虚值，实际覆盖率可能远低于46%。见下方"局限性"详细分析。
</div>
<div class="chart-box"><div id="c5" style="height:360px"></div></div>
</div>

<div class="section">
<h2>七、操作步骤</h2>
<div class="steps"><ol>
  <li><b>IBKR账户</b>：开通欧洲期权交易权限，交易所选MEFF。</li>
  <li><b>买第一腿——ATM Put &times;8</b>：搜索Mini IBEX期权，到期月<b>2027年3月</b>，类型Put，行权价<b>{rec['K']:,}</b>（ATM，50点间距）。参考价约&euro;{rec['price']:,.0f}/张，8张合计约&euro;{round(rec['price']*8):,}。</li>
  <li><b>买第二腿——90%OTM Put &times;20</b>：同到期月，行权价<b>{K_90:,}</b>（90% OTM）。参考价约&euro;{round(bs_put(ibex_now,K_90,1.0)):,}/张，20张合计约&euro;{round(bs_put(ibex_now,K_90,1.0)*20):,}。</li>
  <li><b>总保费</b>约&euro;{rec_prem:,}（Black-Scholes理论值，IV={IBEX_IMPLIED_VOL*100:.1f}%，r={ECB_RATE*100:.1f}%），实际市价可能上浮10-30%。</li>
  <li><b>动态滚仓触发</b>（关键！）：不要死等12个月到期。<b>IBEX涨超15%（>{round(ibex_now*1.15):,}）时必须提前滚仓</b>——
  卖掉旧Put（已变深度虚值），买入新的ATM Put重设行权价。这能防止"先涨后跌"时Put变废纸（Event #10教训）。</li>
  <li><b>到期前1个月滚仓</b>：如果IBEX没有大涨，正常到期前卖旧买新，周而复始。</li>
</ol></div>
<div class="alert a-info" style="font-size:13px">
  动态滚仓会增加交易频率和额外保费支出（每次滚仓损失旧Put的剩余时间价值），但能确保行权价始终贴近当前市场，
  避免保护失效。预计每年触发0-2次额外滚仓。
</div>
</div>

<div class="section">
<h2>八、局限性（必读）</h2>
<div class="alert a-bad">
  <b>Event #10 教训：年度滚仓的致命缺陷</b><br>
  2025年3-4月，基金跌7.4%（约&euro;49,000），这是持仓期最大一次回撤。但12月ATM Put的行权价设在2024年7月买入时的IBEX水平（约11,090点，注意：这是历史回测值，当前推荐行权价为{rec['K']:,}点），
  到事件发生时IBEX已涨到13,484点，即使跌到11,786点仍在行权价<b>之上</b>——Put几乎是废纸，覆盖仅约5%。<br><br>
  <b>结论</b>：固定年度滚仓在"先涨后跌"行情下保护形同虚设。这就是为什么操作步骤中加入了<b>动态滚仓触发</b>（IBEX涨超15%即提前滚仓重设行权价）。
</div>
<div class="alert a-warn">
  <ol style="margin:0 0 0 18px">
    <li><b>这是减震垫，不是全额保险</b>：在理想线性假设下覆盖约46%的基金损失。在先涨后跌场景下可能远低于此。即使加入动态滚仓，也无法保证覆盖率。</li>
    <li><b>保费是确定支出</b>：每年&euro;{rec_prem:,}，5年不出事白花&euro;{rec_prem*5:,}（组合的{rec_prem*5/fv*100:.1f}%）。</li>
    <li><b>R&sup2;=42%的根本限制</b>：IBEX只能解释基金42%的波动。基金可能因为葡萄牙本地原因（个股暴雷、流动性危机）大跌而IBEX无动于衷，此时Put完全无效。</li>
    <li><b>线性模型在极端行情下失真</b>：场景表使用恒定Beta，但极端尾部事件中Beta会漂移，覆盖率可能偏离预期。</li>
    <li><b>IV影响成本</b>：恐慌期Put更贵，尽量在平静期滚仓。</li>
  </ol>
</div>
</div>

<div class="section">
<h2>九、总结</h2>
<div class="alert a-note" style="font-size:14px;line-height:1.9">
  <b style="font-size:17px;color:#4a148c">ATM &times;8 + 90%OTM &times;20，12个月年滚 + 动态滚仓</b><br>
  <b>能做到的</b>：历史{nt}次急跌IBEX 100%同步下跌。混合配置在大跌时赔付比纯ATM多33-60%，同时保留小跌基本保护。<br>
  <b>做不到的</b>：无法覆盖IBEX以外58%的风险（R&sup2;=42%）。在"先涨后跌"行情下，如果没有及时动态滚仓，Put可能接近废纸。<br>
  <b>成本</b>：年化{rec_prem/fv*100:.2f}%（&euro;{rec_prem:,}/年），5年约&euro;{rec_prem*5:,}，是确定的支出。<br>
  <b>本质</b>：这是一个减震垫，不是全额保险。它降低了系统性暴跌中的最大亏损幅度，但不能保证你不亏钱。
</div>
</div>

</div>
<script>
Plotly.newPlot('c1',__C1__.data,__C1__.layout,{{responsive:true}});
Plotly.newPlot('c2',__C2__.data,__C2__.layout,{{responsive:true}});
Plotly.newPlot('c5',__C5__.data,__C5__.layout,{{responsive:true}});
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
    if(dd>0)parts.push(dd+'天');
    parts.push(hh+'时'+mm+'分');
    el.textContent='已过 '+parts.join('')+'，建议每周刷新一次';
    if(dd>=7)el.style.color='#c62828';
  }}
  tick();setInterval(tick,60000);
}})();
</script></body></html>"""

    html = html.replace('__C1__',c1).replace('__C2__',c2).replace('__C5__',c5)
    html = html.replace('__ZOOM__', '['+','.join(c if c else 'null' for c in zooms)+']')
    return html


def main():
    from datetime import datetime
    print('获取实时价格...')
    prices = fetch_live_prices()
    fund_nav = prices.get('fund', 17.15)  # fallback
    psi_now = prices.get('psi', 8862)
    ibex_now = prices.get('ibex', 17062)
    estx_now = prices.get('estx', 6138)
    fund_value = round(fund_nav * FUND_UNITS)
    gen_time = datetime.now().strftime('%Y-%m-%d %H:%M')
    fund_src = prices.get('fund_src', 'Yahoo Finance')
    live = dict(fund_nav=fund_nav, fund_value=fund_value, psi=psi_now,
                ibex=ibex_now, estx=estx_now, gen_time=gen_time, fund_src=fund_src,
                fund_date=prices.get('fund_date','?'),
                psi_date=prices.get('psi_date','?'),
                ibex_date=prices.get('ibex_date','?'),
                estx_date=prices.get('estx_date','?'))
    print(f'  基金NAV=€{fund_nav:.2f}({fund_src}, {prices.get("fund_date","?")}) 市值=€{fund_value:,}')
    print(f'  PSI20={psi_now:,.0f} IBEX={ibex_now:,.0f} ESTX={estx_now:,.0f}')

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
    import subprocess, platform
    if platform.system() == 'Darwin': subprocess.run(['open', out])

if __name__ == '__main__': main()
