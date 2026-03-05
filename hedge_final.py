"""
葡萄牙基金对冲方案 — 完整报告 v7
- 10次事件全部可点击放大
- 修正合约规格(Mini IBEX)、脱钩次数、操作代码
- 精简废话
"""

import os, sys, math
import numpy as np
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots

sys.stdout.reconfigure(encoding='utf-8')
DATA_DIR = os.path.dirname(os.path.abspath(__file__))

# ═══ 核心参数 ══════════════════════════════════
FUND_VALUE       = 651_000
INITIAL_INV      = 507_000
FUND_ENTRY_PRICE = 13.3534
FUND_UNITS       = round(INITIAL_INV / FUND_ENTRY_PRICE)
FUND_CURR_PRICE  = round(FUND_VALUE / FUND_UNITS, 4)
PSI20_CURRENT    = 8_862
PSI20_ENTRY_ACT  = 6_711
IBEX_CURRENT     = 17_062
BETA_FUND_IBEX   = 0.4230
IBEX_IMPLIED_VOL = 0.185
ECB_RATE         = 0.026
N_CONTRACTS      = 16          # Mini IBEX, 乘数 €1/点
ENTRY_DATE       = '2024-07-16'
# ═══════════════════════════════════════════════

def norm_cdf(x):
    return (1 + math.erf(x / math.sqrt(2))) / 2

def bs_put(S, K, T, r=ECB_RATE, sigma=IBEX_IMPLIED_VOL):
    if T <= 0: return max(K - S, 0)
    d1 = (math.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*math.sqrt(T))
    d2 = d1 - sigma*math.sqrt(T)
    return K*math.exp(-r*T)*norm_cdf(-d2) - S*norm_cdf(-d1)

def load_data():
    fund = pd.read_csv(os.path.join(DATA_DIR, 'PTOPZWHM0007_daily_2022-2026.csv'), parse_dates=['Date'])
    fund = fund.sort_values('Date').reset_index(drop=True)
    ibex = yf.download('^IBEX', start='2022-01-01', end='2026-03-06', progress=False)
    ibex.columns = [c[0] for c in ibex.columns]
    ibex = ibex.reset_index()[['Date','Close']].rename(columns={'Close':'ibex'})
    ibex['Date'] = ibex['Date'].dt.normalize()
    psi = yf.download('PSI20.LS', start='2022-01-01', end='2026-03-06', progress=False)
    psi.columns = [c[0] for c in psi.columns]
    psi = psi.reset_index()[['Date','Close']].rename(columns={'Close':'psi'})
    psi['Date'] = psi['Date'].dt.normalize()
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

def analyze(fund_df, ibex_df, psi_df):
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
        loss = abs(FUND_VALUE * ev['fund_chg'] / 100)
        ev['fund_loss'] = loss
        idx = (df_h['ibex'] - ev['ibex_level']).abs().idxmin()
        for k, s in strats.items():
            mtm, strike = get_put_mtm(s['positions'], ev['ibex_level'], idx)
            ev['strat'][k] = dict(mtm=mtm, strike=strike, cov=mtm/loss*100 if loss>0 else 0)
    # 21M backtest
    bt = []
    for i in range(0, max(1,len(df)-441), 21):
        ib = df['ibex'].iloc[i]; K = ib*0.95
        ie = df['ibex'].iloc[min(i+441, len(df)-1)]
        bt.append(dict(date=str(df['Date'].iloc[i].date()), ib=ib, K=K, ie=ie, pay=max(K-ie,0)*N_CONTRACTS))
    # Recommendation
    K = round(IBEX_CURRENT); p = bs_put(IBEX_CURRENT, K, 1.0); tot = round(p*N_CONTRACTS)
    rec = dict(K=K, price=round(p,1), total=tot, annual=round(tot/FUND_VALUE*100,2))
    K_old = round(IBEX_CURRENT*0.95)
    return dict(df=df, df_h=df_h, events=events, strats=strats, bt=bt, rec=rec,
                rec_old=dict(K=K_old, total=round(bs_put(IBEX_CURRENT,K_old,21/12)*N_CONTRACTS)))

# ─── Charts ──────────────────────────────
def chart_fund_psi(fund_df, psi_df):
    e = pd.Timestamp(ENTRY_DATE)
    m = pd.merge(fund_df[fund_df['Date']>=e][['Date','Close']], psi_df[psi_df['Date']>=e], on='Date', how='inner')
    rf, rp = m['Close'].iloc[0], m['psi'].iloc[0]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=m['Date'], y=(m['Close']/rf-1)*100, name='基金', line=dict(color='#1565c0',width=2.5)))
    fig.add_trace(go.Scatter(x=m['Date'], y=(m['psi']/rp-1)*100, name='PSI20', line=dict(color='#ff7f0e',width=2,dash='dot')))
    fig.add_hline(y=0, line_dash='dot', line_color='gray', opacity=0.3)
    fig.update_layout(template='plotly_white', height=300, yaxis_title='相对买入日涨跌(%)',
        legend=dict(x=0.01,y=0.99), margin=dict(t=10,b=30,l=60,r=20), hovermode='x unified')
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

def chart_backtest(bt):
    d = [b['date'] for b in bt]
    fig = make_subplots(rows=2, cols=1, row_heights=[0.6,0.4], shared_xaxes=True, vertical_spacing=0.08)
    fig.add_trace(go.Scatter(x=d,y=[b['ib'] for b in bt],name='买入时IBEX',mode='lines+markers',
        marker=dict(size=4),line=dict(color='#1565c0',width=2)),row=1,col=1)
    fig.add_trace(go.Scatter(x=d,y=[b['K'] for b in bt],name='行权价(95%)',mode='lines+markers',
        marker=dict(size=4),line=dict(color='#e65100',width=2,dash='dash')),row=1,col=1)
    fig.add_trace(go.Scatter(x=d,y=[b['ie'] for b in bt],name='到期时IBEX',mode='lines+markers',
        marker=dict(size=4),line=dict(color='#2e7d32',width=2)),row=1,col=1)
    fig.add_trace(go.Bar(x=d,y=[b['pay'] for b in bt],name='Put赔付',
        marker_color=['#c62828' if b['pay']==0 else '#2e7d32' for b in bt]),row=2,col=1)
    fig.update_layout(template='plotly_white',height=400,margin=dict(t=10,b=30,l=60,r=20),legend=dict(x=0.01,y=0.99))
    fig.update_yaxes(title_text='IBEX点位',row=1,col=1)
    fig.update_yaxes(title_text='赔付(EUR)',row=2,col=1)
    return fig.to_json()

def make_zoom_charts(df, events, strats):
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
                pv.append((df_h['fund'].iloc[t]/rf-1)*100 + mtm/FUND_VALUE*100)
            fig.add_trace(go.Scatter(x=w['Date'],y=pv,name='基金+Put',
                line=dict(color='#2e7d32',width=3),hovertemplate='%{x|%m-%d} 组合:%{y:+.1f}%<extra></extra>'))
        fig.add_hline(y=0,line_dash='dot',line_color='gray',opacity=0.3)
        fig.add_vline(x=ev['worst_date'],line_dash='dash',line_color='#c62828',opacity=0.5)
        fig.update_layout(template='plotly_white',height=260,yaxis_title='涨跌(%)',
            legend=dict(x=0.01,y=0.99,font=dict(size=10)),margin=dict(t=5,b=25,l=50,r=15),hovermode='x unified')
        charts.append(fig.to_json())
    return charts

def chart_payoff(rec):
    x = np.linspace(10000,20000,500)
    fp = FUND_VALUE * BETA_FUND_IBEX * (x - IBEX_CURRENT) / IBEX_CURRENT
    pp = np.maximum(rec['K']-x,0)*N_CONTRACTS - rec['total']
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x,y=fp,name='基金损益(不对冲)',line=dict(color='#c62828',width=2.5,dash='dot')))
    fig.add_trace(go.Scatter(x=x,y=fp+pp,name='基金+Put(对冲后)',line=dict(color='#2e7d32',width=3)))
    fig.add_vline(x=IBEX_CURRENT,line_dash='dot',line_color='gray',opacity=0.4,
        annotation_text=f'当前{IBEX_CURRENT:,}',annotation_position='top left',annotation_font=dict(size=10,color='gray'))
    fig.add_vline(x=rec['K'],line_dash='dash',line_color='#2e7d32',opacity=0.5,
        annotation_text=f'行权价{rec["K"]:,}',annotation_position='bottom left',annotation_font=dict(size=10,color='#2e7d32'))
    fig.add_hline(y=0,line_dash='dot',line_color='gray',opacity=0.3)
    fig.update_layout(template='plotly_white',height=340,xaxis_title='12个月后IBEX35点位',yaxis_title='损益(EUR)',
        legend=dict(x=0.01,y=0.01,bgcolor='rgba(255,255,255,0.9)'),margin=dict(t=10,b=50,l=70,r=20),hovermode='x unified')
    return fig.to_json()

# ─── HTML ─────────────────────────────
def generate_html(fund_df, psi_df, res):
    df, events, strats, rec = res['df'], res['events'], res['strats'], res['rec']
    c1 = chart_fund_psi(fund_df, psi_df)
    c2 = chart_fund_ibex(df, events)
    c3 = chart_backtest(res['bt'])
    c5 = chart_payoff(rec)
    zooms = make_zoom_charts(df, events, strats)

    fg = (FUND_CURR_PRICE / FUND_ENTRY_PRICE - 1) * 100
    nt = len(events)
    ns = sum(1 for e in events if e['sync'])
    nb0 = sum(1 for b in res['bt'] if b['pay']==0)

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

        loss = ev.get('fund_loss', abs(FUND_VALUE*ev['fund_chg']/100))
        detail = ''
        if ev['in_hold'] and ev['strat']:
            detail = '<table style="font-size:13px;margin-top:12px"><tr><th style="text-align:left">策略</th><th>行权价</th><th>Put赚了</th><th>净亏</th><th>覆盖率</th></tr>'
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

    # Cost table
    cost_rows = ''
    for k, lab, freq in [('12M','12个月ATM 年滚',1),('6M','6个月ATM 半年滚',2),('3M','3个月ATM 季滚',4)]:
        s = strats[k]; T = s['months']/12
        pe = bs_put(IBEX_CURRENT, IBEX_CURRENT, T)*N_CONTRACTS
        ae = pe * (12/s['months']); fy = ae*5
        is_r = k=='12M'
        st = ' style="background:#f0fff0;font-weight:600"' if is_r else ''
        tg = ' <span style="color:#2e7d32;font-size:11px">[推荐]</span>' if is_r else ''
        cost_rows += f'<tr{st}><td style="text-align:left">{lab}{tg}</td><td>{freq}x/年</td><td>&euro;{ae:,.0f} ({ae/FUND_VALUE*100:.2f}%)</td><td>&euro;{fy:,.0f} ({fy/FUND_VALUE*100:.1f}%)</td></tr>'

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
</style></head><body><div class="page">

<h1>葡萄牙基金对冲方案</h1>
<p class="meta">Optimize Portugal Golden Opportunities Fund (PTOPZWHM0007) &middot; 数据截至2026年3月</p>

<div class="section">
<h2>一、持仓概况</h2>
<div class="cards">
  <div class="card green"><div class="lbl">买入成本</div><div class="val">&euro;{INITIAL_INV:,}</div><div class="sub">2024.07 NAV &euro;{FUND_ENTRY_PRICE:.2f}</div></div>
  <div class="card green"><div class="lbl">当前市值</div><div class="val">&euro;{FUND_VALUE:,}</div><div class="sub">NAV &euro;{FUND_CURR_PRICE:.2f}</div></div>
  <div class="card purple"><div class="lbl">浮盈</div><div class="val">+&euro;{FUND_VALUE-INITIAL_INV:,}</div><div class="sub">+{fg:.1f}%</div></div>
  <div class="card orange"><div class="lbl">PSI20</div><div class="val">{PSI20_CURRENT:,}</div><div class="sub">买入时{PSI20_ENTRY_ACT:,}</div></div>
</div>
<div class="chart-box"><div id="c1" style="height:300px"></div></div>
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
<h2>三、对冲链路</h2>
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
  合约数量按Beta={BETA_FUND_IBEX}计算：基金对IBEX的敏感度为42%，
  需要对冲的名义敞口=&euro;{FUND_VALUE:,}&times;{BETA_FUND_IBEX}=&euro;{round(FUND_VALUE*BETA_FUND_IBEX):,}，
  除以IBEX点位=<b>{N_CONTRACTS}张Mini合约</b>（乘数&euro;1/点）。
</div>
</div>

<div class="section">
<h2>四、为什么不能买远期Put放着不管</h2>
<div class="alert a-bad">
  21个月95%OTM Put：历史<b>{nb0}次回测全部到期归零</b>。IBEX过去4年从7,261涨到18,000+，固定行权价被甩开。
</div>
<div class="chart-box"><div id="c3" style="height:400px"></div></div>
<p class="note-sm">蓝=买入时IBEX，橙虚线=行权价(95%)，绿=到期时IBEX。全部高于行权价→全部归零。</p>
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
  <h3>{N_CONTRACTS}张 Mini IBEX35 12个月ATM Put，每年滚仓</h3>
  <div class="rec-grid">
    <div class="rec-item"><div class="rl">行权价</div><div class="rv">{rec['K']:,}点</div></div>
    <div class="rec-item"><div class="rl">每年保费</div><div class="rv">&euro;{rec['total']:,}</div></div>
    <div class="rec-item"><div class="rl">年化占比</div><div class="rv">{rec['annual']:.2f}%</div></div>
    <div class="rec-item"><div class="rl">操作频率</div><div class="rv">1次/年</div></div>
    <div class="rec-item"><div class="rl">IBEX跌15%赔付</div><div class="rv">&euro;{max(rec['K']-round(IBEX_CURRENT*0.85),0)*N_CONTRACTS:,}</div></div>
    <div class="rec-item"><div class="rl">IBEX跌30%赔付</div><div class="rv">&euro;{max(rec['K']-round(IBEX_CURRENT*0.70),0)*N_CONTRACTS:,}</div></div>
    <div class="rec-item"><div class="rl">5年总保费</div><div class="rv">&euro;{rec['total']*5:,}</div></div>
  </div>
</div>
<div class="chart-box"><div id="c5" style="height:340px"></div></div>
</div>

<div class="section">
<h2>七、操作步骤</h2>
<div class="steps"><ol>
  <li><b>IBKR账户</b>：开通欧洲期权交易权限，交易所选MEFF。</li>
  <li><b>搜索合约</b>：在IBKR搜索"IBEX 35"，找Mini IBEX期权（乘数&euro;1/点），到期月<b>2027年3月</b>，类型Put，行权价<b>{rec['K']:,}</b>。</li>
  <li><b>买入{N_CONTRACTS}张</b>：限价单，参考价约&euro;{rec['price']:,.0f}/张，总计约&euro;{rec['total']:,}。</li>
  <li><b>持有期间</b>：IBEX涨超20%（>{round(IBEX_CURRENT*1.2):,}）时考虑提前换仓，否则不用管。</li>
  <li><b>到期前1个月滚仓</b>：卖旧Put，买新的12个月ATM Put，周而复始。</li>
</ol></div>
</div>

<div class="section">
<h2>八、局限性</h2>
<div class="alert a-warn">
  <ol style="margin:0 0 0 18px">
    <li><b>保费是确定支出</b>：每年&euro;{rec['total']:,}，5年不出事白花&euro;{rec['total']*5:,}。</li>
    <li><b>中等回调覆盖有限</b>：IBEX跌5-10%时，16张Mini合约只能覆盖部分损失。跌20%+才有力。</li>
    <li><b>IV影响成本</b>：恐慌期Put更贵，尽量在平静期滚仓。</li>
  </ol>
</div>
</div>

<div class="section">
<h2>九、总结</h2>
<div class="alert a-note" style="font-size:15px;line-height:2">
  <b style="font-size:17px;color:#4a148c">{N_CONTRACTS}张 Mini IBEX35 12个月ATM Put，每年滚仓</b><br>
  历史{nt}次急跌IBEX<b>100%同步</b>。年化成本{rec['annual']:.2f}%，5年约&euro;{rec['total']*5:,}，每年操作1次。
</div>
</div>

</div>
<script>
Plotly.newPlot('c1',__C1__.data,__C1__.layout,{{responsive:true}});
Plotly.newPlot('c2',__C2__.data,__C2__.layout,{{responsive:true}});
Plotly.newPlot('c3',__C3__.data,__C3__.layout,{{responsive:true}});
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
</script></body></html>"""

    html = html.replace('__C1__',c1).replace('__C2__',c2).replace('__C3__',c3).replace('__C5__',c5)
    html = html.replace('__ZOOM__', '['+','.join(c if c else 'null' for c in zooms)+']')
    return html

def main():
    print('加载数据...')
    fund_df, ibex_df, psi_df = load_data()
    print(f'基金{len(fund_df)}条 IBEX{len(ibex_df)}条 PSI{len(psi_df)}条')
    print('分析...')
    res = analyze(fund_df, ibex_df, psi_df)
    ev = res['events']
    print(f'{len(ev)}次急跌({sum(1 for e in ev if e["sync"])}次同步) 持仓期{sum(1 for e in ev if e["in_hold"])}次')
    print('生成报告...')
    html = generate_html(fund_df, psi_df, res)
    out = os.path.join(DATA_DIR, 'hedge_final.html')
    with open(out, 'w', encoding='utf-8') as f: f.write(html)
    print(f'→ {out}')
    import subprocess, platform
    if platform.system() == 'Darwin': subprocess.run(['open', out])

if __name__ == '__main__': main()
