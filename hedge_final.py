"""
葡萄牙基金对冲方案 — 完整报告 v6
改动：
  - 事件检测改为"1周内跌超3%"(更全面，10次事件)
  - 组合净值改为Tab切换，每个事件单独放大展示
  - 去掉21M已否决方案的对比列
"""

import os, sys, math, json
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

BETA_FUND_PSI20  = 0.6271
BETA_FUND_IBEX   = 0.4230

IBEX_IMPLIED_VOL = 0.185
ECB_RATE         = 0.026
N_CONTRACTS      = 16
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

    ibex_full = yf.download('^IBEX', start='2022-01-01', end='2026-03-06', progress=False)
    ibex_full.columns = [c[0] for c in ibex_full.columns]
    ibex_full = ibex_full.reset_index()[['Date','Close']].rename(columns={'Close':'ibex'})
    ibex_full['Date'] = ibex_full['Date'].dt.normalize()

    psi = yf.download('PSI20.LS', start='2022-01-01', end='2026-03-06', progress=False)
    psi.columns = [c[0] for c in psi.columns]
    psi = psi.reset_index()[['Date','Close']].rename(columns={'Close':'psi'})
    psi['Date'] = psi['Date'].dt.normalize()

    return fund, ibex_full, psi


def find_weekly_drops(df, threshold=-3.0):
    """找出基金1周(5个交易日)内跌幅超过threshold%的所有事件"""
    df = df.copy()
    df['fund_5d'] = df['fund'].pct_change(5) * 100
    df['ibex_5d'] = df['ibex'].pct_change(5) * 100

    drops = df[df['fund_5d'] < threshold].copy()
    events = []
    prev_end = None

    for _, row in drops.iterrows():
        d = row['Date']
        if prev_end is not None and (d - prev_end).days < 10:
            if row['fund_5d'] < events[-1]['fund_chg']:
                events[-1]['fund_chg'] = row['fund_5d']
                events[-1]['ibex_chg'] = row['ibex_5d']
                events[-1]['worst_date'] = d
                events[-1]['ibex_level'] = row['ibex']
                events[-1]['fund_level'] = row['fund']
            events[-1]['end'] = d
        else:
            # find the start of the 5-day window
            idx = df.index[df['Date'] == d][0]
            start_idx = max(0, idx - 5)
            events.append(dict(
                start=df['Date'].iloc[start_idx],
                end=d,
                worst_date=d,
                fund_chg=row['fund_5d'],
                ibex_chg=row['ibex_5d'],
                ibex_level=row['ibex'],
                fund_level=row['fund'],
                ibex_start=df['ibex'].iloc[start_idx],
                fund_start=df['fund'].iloc[start_idx],
                sync=row['ibex_5d'] < -1.5,
            ))
        prev_end = d

    # update sync flag for merged events
    for ev in events:
        ev['sync'] = ev['ibex_chg'] < -1.5
        ev['start_str'] = ev['start'].strftime('%Y-%m-%d')
        ev['end_str'] = ev['worst_date'].strftime('%Y-%m-%d')
        ev['in_hold'] = ev['worst_date'] >= pd.Timestamp(ENTRY_DATE)

    return events


def simulate_strategy(df_holding, months, strike_pct):
    roll_days = months * 21
    T_years = months / 12
    positions = []
    total_premium = 0

    i = 0
    while i < len(df_holding):
        ibex_now = df_holding['ibex'].iloc[i]
        K = ibex_now * strike_pct
        prem = bs_put(ibex_now, K, T_years) * N_CONTRACTS
        total_premium += prem
        exp_idx = min(i + roll_days, len(df_holding) - 1)
        positions.append(dict(buy_idx=i, strike=K, expiry_idx=exp_idx, ibex_at_buy=ibex_now, premium=prem))
        i = exp_idx + 1

    return dict(
        positions=positions,
        total_premium=total_premium,
        n_rolls=len(positions),
        months=months,
        annual_cost=total_premium / max(len(df_holding)/252, 0.5),
        annual_pct=total_premium / max(len(df_holding)/252, 0.5) / FUND_VALUE * 100,
    )


def get_put_mtm(positions, ibex_val, day_idx):
    """Get put MTM value at a given day index."""
    for pos in positions:
        if pos['buy_idx'] <= day_idx <= pos['expiry_idx']:
            remaining_T = max((pos['expiry_idx'] - day_idx) / 252, 0.001)
            mtm = bs_put(ibex_val, pos['strike'], remaining_T) * N_CONTRACTS
            return mtm, pos['strike']
    return 0, 0


def analyze(fund_df, ibex_df, psi_df):
    df = pd.merge(
        fund_df[['Date','Close']].rename(columns={'Close':'fund'}),
        ibex_df, on='Date', how='inner'
    ).sort_values('Date').reset_index(drop=True)

    res = {'df': df}

    # Weekly drop events (full history)
    events = find_weekly_drops(df)
    res['events'] = events

    # Holding period
    df_hold = df[df['Date'] >= ENTRY_DATE].reset_index(drop=True)
    res['df_hold'] = df_hold

    # Strategies
    strats = {}
    for key, label, months, freq in [
        ('3M_ATM',  '3个月ATM 季滚',   3, 4),
        ('6M_ATM',  '6个月ATM 半年滚',  6, 2),
        ('12M_ATM', '12个月ATM 年滚',  12, 1),
    ]:
        s = simulate_strategy(df_hold, months, 1.00)
        s['label'] = label
        s['freq'] = freq
        strats[key] = s
    res['strategies'] = strats

    # For each event in holding period, calc put performance per strategy
    for ev in events:
        ev['strat_results'] = {}
        if not ev['in_hold']:
            continue
        # find this event's worst_date in df_hold
        mask = df_hold['Date'] == ev['worst_date']
        if mask.sum() == 0:
            continue
        day_idx = df_hold.index[mask][0]
        fund_loss = abs(FUND_VALUE * ev['fund_chg'] / 100)
        ev['fund_loss'] = fund_loss

        for skey, s in strats.items():
            mtm, strike = get_put_mtm(s['positions'], ev['ibex_level'], day_idx)
            coverage = mtm / fund_loss * 100 if fund_loss > 0 else 0
            ev['strat_results'][skey] = dict(mtm=mtm, strike=strike, coverage=coverage)

    # Backtest: 21-month 95% OTM
    backtest = []
    horizon = 441
    for i in range(0, max(1, len(df) - horizon), 21):
        ib = df['ibex'].iloc[i]
        K = ib * 0.95
        ie = df['ibex'].iloc[min(i + horizon, len(df) - 1)]
        backtest.append(dict(buy_date=str(df['Date'].iloc[i].date()),
                             ibex_buy=ib, strike=K, ibex_exp=ie,
                             payoff=max(K - ie, 0) * N_CONTRACTS))
    res['backtest'] = backtest

    # Recommendation
    K_new = round(IBEX_CURRENT)
    price_new = bs_put(IBEX_CURRENT, K_new, 1.0)
    total_new = round(price_new * N_CONTRACTS)
    res['rec'] = dict(K=K_new, T=1.0, price=round(price_new, 1), total=total_new,
                      annual=round(total_new / FUND_VALUE * 100, 2))
    K_old = round(IBEX_CURRENT * 0.95)
    res['rec_old'] = dict(K=K_old, total=round(bs_put(IBEX_CURRENT, K_old, 21/12) * N_CONTRACTS))

    return res


# ─── Charts ──────────────────────────────────────

def chart_fund_psi(fund_df, psi_df):
    entry = pd.Timestamp(ENTRY_DATE)
    m = pd.merge(fund_df[fund_df['Date'] >= entry][['Date','Close']],
                 psi_df[psi_df['Date'] >= entry], on='Date', how='inner')
    rf, rp = m['Close'].iloc[0], m['psi'].iloc[0]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=m['Date'], y=(m['Close']/rf-1)*100, name='基金',
        mode='lines', line=dict(color='#1565c0', width=2.5)))
    fig.add_trace(go.Scatter(x=m['Date'], y=(m['psi']/rp-1)*100, name='PSI20',
        mode='lines', line=dict(color='#ff7f0e', width=2, dash='dot')))
    fig.add_hline(y=0, line_dash='dot', line_color='gray', opacity=0.3)
    fig.update_layout(template='plotly_white', height=320,
        yaxis_title='相对买入日涨跌 (%)', legend=dict(x=0.01, y=0.99),
        margin=dict(t=10, b=30, l=60, r=20), hovermode='x unified')
    return fig.to_json()


def chart_fund_ibex(df, events):
    rf, ri = df['fund'].iloc[0], df['ibex'].iloc[0]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['Date'], y=(df['fund']/rf-1)*100, name='基金',
        mode='lines', line=dict(color='#1565c0', width=2.5)))
    fig.add_trace(go.Scatter(x=df['Date'], y=(df['ibex']/ri-1)*100, name='IBEX35',
        mode='lines', line=dict(color='#e65100', width=2.5)))
    fig.add_hline(y=0, line_dash='dot', line_color='gray', opacity=0.3)

    for ev in events:
        color = '#c8e6c9' if ev['sync'] else '#fff9c4'
        fig.add_vrect(x0=ev['start'], x1=ev['end'], fillcolor=color, opacity=0.2)

    fig.add_vline(x=ENTRY_DATE, line_dash='dot', line_color='#c62828', opacity=0.5)
    fig.add_annotation(x=ENTRY_DATE, y=0.02, yref='paper', text='买入日',
        showarrow=False, font=dict(size=10, color='#c62828'))
    fig.update_layout(template='plotly_white', height=460,
        yaxis_title='相对2022年初涨跌 (%)', legend=dict(x=0.01, y=0.99),
        margin=dict(t=10, b=30, l=60, r=20), hovermode='x unified')
    return fig.to_json()


def chart_backtest(backtest):
    dates = [b['buy_date'] for b in backtest]
    fig = make_subplots(rows=2, cols=1, row_heights=[0.6, 0.4], shared_xaxes=True, vertical_spacing=0.08)
    fig.add_trace(go.Scatter(x=dates, y=[b['ibex_buy'] for b in backtest], name='买入时IBEX',
        mode='lines+markers', marker=dict(size=4), line=dict(color='#1565c0', width=2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=dates, y=[b['strike'] for b in backtest], name='行权价(95%)',
        mode='lines+markers', marker=dict(size=4), line=dict(color='#e65100', width=2, dash='dash')), row=1, col=1)
    fig.add_trace(go.Scatter(x=dates, y=[b['ibex_exp'] for b in backtest], name='到期时IBEX',
        mode='lines+markers', marker=dict(size=4), line=dict(color='#2e7d32', width=2)), row=1, col=1)
    fig.add_trace(go.Bar(x=dates, y=[b['payoff'] for b in backtest], name='Put赔付',
        marker_color=['#c62828' if b['payoff']==0 else '#2e7d32' for b in backtest]), row=2, col=1)
    fig.update_layout(template='plotly_white', height=420, margin=dict(t=10, b=30, l=60, r=20),
        legend=dict(x=0.01, y=0.99))
    fig.update_yaxes(title_text='IBEX点位', row=1, col=1)
    fig.update_yaxes(title_text='赔付(EUR)', row=2, col=1)
    return fig.to_json()


def make_event_zoom_charts(df, events, strategies):
    """For each holding-period event, create a zoomed ±15 trading days chart
    showing fund, IBEX, and portfolio NAV (12M ATM)."""
    charts = []
    df_hold = df[df['Date'] >= ENTRY_DATE].reset_index(drop=True)

    for ev in events:
        if not ev['in_hold']:
            charts.append(None)
            continue

        # Find window: ±15 trading days around worst_date
        center_mask = df_hold['Date'] == ev['worst_date']
        if center_mask.sum() == 0:
            charts.append(None)
            continue
        center_idx = df_hold.index[center_mask][0]
        lo = max(0, center_idx - 15)
        hi = min(len(df_hold) - 1, center_idx + 15)
        window = df_hold.iloc[lo:hi+1]

        ref_f = window['fund'].iloc[0]
        ref_i = window['ibex'].iloc[0]

        fig = go.Figure()
        # Fund
        fig.add_trace(go.Scatter(x=window['Date'], y=(window['fund']/ref_f-1)*100,
            name='基金', mode='lines', line=dict(color='#1565c0', width=2.5),
            hovertemplate='%{x|%m-%d}<br>基金: %{y:+.1f}%<extra></extra>'))
        # IBEX
        fig.add_trace(go.Scatter(x=window['Date'], y=(window['ibex']/ref_i-1)*100,
            name='IBEX35', mode='lines', line=dict(color='#e65100', width=2.5),
            hovertemplate='%{x|%m-%d}<br>IBEX: %{y:+.1f}%<extra></extra>'))

        # 12M ATM portfolio
        s12 = strategies['12M_ATM']
        port_vals = []
        cum_prem = sum(p['premium'] for p in s12['positions'] if p['buy_idx'] <= lo)
        for t_idx in range(lo, hi+1):
            mtm, strike = get_put_mtm(s12['positions'], df_hold['ibex'].iloc[t_idx], t_idx)
            fund_val = df_hold['fund'].iloc[t_idx] / ref_f
            # portfolio relative change = fund change + (put mtm change) / fund_value_at_ref
            port_vals.append((df_hold['fund'].iloc[t_idx]/ref_f - 1)*100 + (mtm / FUND_VALUE)*100)

        fig.add_trace(go.Scatter(x=window['Date'], y=port_vals,
            name='基金+12M Put', mode='lines', line=dict(color='#2e7d32', width=3),
            hovertemplate='%{x|%m-%d}<br>组合: %{y:+.1f}%<extra></extra>'))

        fig.add_hline(y=0, line_dash='dot', line_color='gray', opacity=0.3)
        fig.add_vline(x=ev['worst_date'], line_dash='dash', line_color='#c62828', opacity=0.5)

        fig.update_layout(template='plotly_white', height=280,
            yaxis_title='涨跌 (%)', legend=dict(x=0.01, y=0.99, font=dict(size=10)),
            margin=dict(t=5, b=25, l=50, r=15), hovermode='x unified')

        charts.append(fig.to_json())

    return charts


def chart_payoff(rec):
    ibex_range = np.linspace(10000, 20000, 500)
    fund_pnl = FUND_VALUE * BETA_FUND_IBEX * (ibex_range - IBEX_CURRENT) / IBEX_CURRENT
    put_pnl = np.maximum(rec['K'] - ibex_range, 0) * N_CONTRACTS - rec['total']
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ibex_range, y=fund_pnl, name='基金损益(不对冲)',
        mode='lines', line=dict(color='#c62828', width=2.5, dash='dot')))
    fig.add_trace(go.Scatter(x=ibex_range, y=fund_pnl + put_pnl, name='基金+Put(对冲后)',
        mode='lines', line=dict(color='#2e7d32', width=3)))
    fig.add_vline(x=IBEX_CURRENT, line_dash='dot', line_color='gray', opacity=0.4,
        annotation_text=f'当前 {IBEX_CURRENT:,}', annotation_position='top left',
        annotation_font=dict(size=10, color='gray'))
    fig.add_vline(x=rec['K'], line_dash='dash', line_color='#2e7d32', opacity=0.5,
        annotation_text=f'行权价 {rec["K"]:,}', annotation_position='bottom left',
        annotation_font=dict(size=10, color='#2e7d32'))
    fig.add_hline(y=0, line_dash='dot', line_color='gray', opacity=0.3)
    fig.update_layout(template='plotly_white', height=360,
        xaxis_title='12个月后IBEX35点位', yaxis_title='损益 (EUR)',
        legend=dict(x=0.01, y=0.01, bgcolor='rgba(255,255,255,0.9)'),
        margin=dict(t=10, b=50, l=70, r=20), hovermode='x unified')
    return fig.to_json()


# ─── HTML ─────────────────────────────────────

def generate_html(fund_df, psi_df, res):
    df = res['df']
    events = res['events']
    strats = res['strategies']
    rec = res['rec']
    rec_old = res['rec_old']
    backtest = res['backtest']

    c1 = chart_fund_psi(fund_df, psi_df)
    c2 = chart_fund_ibex(df, events)
    c3 = chart_backtest(backtest)
    c5 = chart_payoff(rec)
    zoom_charts = make_event_zoom_charts(df, events, strats)

    fund_gain = (FUND_CURR_PRICE / FUND_ENTRY_PRICE - 1) * 100
    n_total = len(events)
    n_sync = sum(1 for e in events if e['sync'])
    n_bt_zero = sum(1 for b in backtest if b['payoff'] == 0)

    # ── Section 2: weekly drop event table ──
    hist_rows = ''
    for i, ev in enumerate(events):
        sync_tag = '<span style="color:#2e7d32;font-weight:700">同步</span>' if ev['sync'] else '<span style="color:#e65100">脱钩</span>'
        hold_mark = '' if ev['in_hold'] else ' <span style="font-size:10px;color:#aaa">(买入前)</span>'
        hist_rows += f"""<tr{' style="background:#f0fff0"' if ev['sync'] else ''}>
          <td>{i+1}</td>
          <td>{ev['start_str']} ~ {ev['end_str']}{hold_mark}</td>
          <td style="color:#c62828;font-weight:600">{ev['fund_chg']:.1f}%</td>
          <td style="color:{'#2e7d32' if ev['ibex_chg']<-1.5 else '#c62828'};font-weight:600">{ev['ibex_chg']:+.1f}%</td>
          <td>{ev['ibex_level']:,.0f}</td>
          <td>{sync_tag}</td>
        </tr>"""

    # ── Section 5: cost table ──
    cost_rows = ''
    for skey in ['12M_ATM', '6M_ATM', '3M_ATM']:
        s = strats[skey]
        is_rec = (skey == '12M_ATM')
        style = ' style="background:#f0fff0;font-weight:600"' if is_rec else ''
        tag = ' <span style="color:#2e7d32;font-size:11px">[推荐]</span>' if is_rec else ''
        T_est = s['months'] / 12
        prem_est = bs_put(IBEX_CURRENT, IBEX_CURRENT, T_est) * N_CONTRACTS
        rolls_yr = 12 / s['months']
        annual_est = prem_est * rolls_yr
        five_yr = annual_est * 5
        cost_rows += f"""<tr{style}>
          <td style="text-align:left">{s['label']}{tag}</td>
          <td>{s['freq']}x/年</td>
          <td>&euro;{annual_est:,.0f} <span style="font-size:11px;color:#888">({annual_est/FUND_VALUE*100:.2f}%)</span></td>
          <td>&euro;{five_yr:,.0f} <span style="font-size:11px;color:#888">({five_yr/FUND_VALUE*100:.1f}%)</span></td>
        </tr>"""

    # ── Section 5: Tab content for each holding-period event ──
    hold_events = [e for e in events if e['in_hold']]
    tab_buttons = ''
    tab_panels = ''
    for idx, ev in enumerate(hold_events):
        active = ' active' if idx == 0 else ''
        sync_label = '同步' if ev['sync'] else '脱钩'
        tab_buttons += f'<button class="tab-btn{active}" onclick="showTab({idx})">{ev["end_str"][5:]} 基金{ev["fund_chg"]:.1f}%</button>'

        # Build panel content
        fund_loss = ev.get('fund_loss', abs(FUND_VALUE * ev['fund_chg'] / 100))
        strat_detail = ''
        for skey, slabel in [('12M_ATM','12月ATM年滚'), ('6M_ATM','6月ATM半年滚'), ('3M_ATM','3月ATM季滚')]:
            r = ev['strat_results'].get(skey, {})
            mtm = r.get('mtm', 0)
            strike = r.get('strike', 0)
            coverage = r.get('coverage', 0)
            net = fund_loss - mtm
            cov_color = '#2e7d32' if coverage >= 20 else ('#e65100' if coverage >= 5 else '#c62828')
            rec_mark = ' <b style="color:#2e7d32">[推荐]</b>' if skey == '12M_ATM' else ''
            strat_detail += f"""<tr>
              <td style="text-align:left">{slabel}{rec_mark}</td>
              <td style="font-size:12px;color:#888">{strike:,.0f}</td>
              <td style="color:#2e7d32;font-weight:700">+&euro;{mtm:,.0f}</td>
              <td style="color:#1565c0;font-weight:700">-&euro;{net:,.0f}</td>
              <td style="font-weight:700;color:{cov_color}">{coverage:.0f}%</td>
            </tr>"""

        # Find zoom chart index in full events list
        full_idx = events.index(ev)
        chart_json = zoom_charts[full_idx] if zoom_charts[full_idx] else 'null'
        chart_div_id = f'zoom_{idx}'

        tab_panels += f"""<div class="tab-panel{''+' active' if idx==0 else ''}" id="panel_{idx}">
          <div class="two-col">
            <div>
              <div class="alert {'a-good' if ev['sync'] else 'a-warn'}" style="margin-bottom:12px">
                <b>{ev['start_str']} ~ {ev['end_str']}</b><br>
                基金1周跌 <b style="color:#c62828">{ev['fund_chg']:.1f}%</b>
                (约&euro;{fund_loss:,.0f})<br>
                IBEX同期 <b style="color:{'#2e7d32' if ev['sync'] else '#e65100'}">{ev['ibex_chg']:+.1f}%</b>
                (跌到{ev['ibex_level']:,.0f}点)<br>
                状态：<b>{sync_label}</b> {'— Put有保护作用' if ev['sync'] else '— IBEX没跟，Put无效'}
              </div>
              <table style="font-size:13px">
                <tr><th style="text-align:left">策略</th><th>行权价</th><th>Put赚了</th><th>相抵后净亏</th><th>覆盖率</th></tr>
                {strat_detail}
              </table>
            </div>
            <div class="chart-box" style="padding:10px"><div id="{chart_div_id}" style="height:280px"></div></div>
          </div>
        </div>"""

    # Collect zoom chart JSONs for JS
    zoom_js_data = []
    for idx, ev in enumerate(hold_events):
        full_idx = events.index(ev)
        zoom_js_data.append(zoom_charts[full_idx])

    # ── Summary stats for section 5 ──
    hold_sync_events = [e for e in hold_events if e['sync']]
    avg_cov_12m = np.mean([e['strat_results'].get('12M_ATM',{}).get('coverage',0) for e in hold_sync_events]) if hold_sync_events else 0
    avg_cov_6m = np.mean([e['strat_results'].get('6M_ATM',{}).get('coverage',0) for e in hold_sync_events]) if hold_sync_events else 0
    avg_cov_3m = np.mean([e['strat_results'].get('3M_ATM',{}).get('coverage',0) for e in hold_sync_events]) if hold_sync_events else 0

    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>葡萄牙基金对冲方案</title>
<script src="https://cdn.plot.ly/plotly-3.4.0.min.js" crossorigin="anonymous"></script>
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;
     background:#f4f6fb;color:#1a1a2e;font-size:15px;line-height:1.65}}
.page{{max-width:1100px;margin:0 auto;padding:32px 20px 70px}}
h1{{font-size:26px;font-weight:800;color:#1a237e;margin-bottom:6px}}
.meta{{color:#777;font-size:13px;margin-bottom:36px}}
h2{{font-size:17px;font-weight:700;color:#1a237e;margin-bottom:16px;
    padding:6px 12px;border-left:4px solid #1a237e;background:#eef2ff;border-radius:0 6px 6px 0}}
.section{{margin-bottom:48px}}
.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:14px;margin-bottom:18px}}
.card{{background:white;border-radius:12px;padding:18px 16px;box-shadow:0 2px 10px rgba(0,0,0,0.07)}}
.card .lbl{{font-size:11px;color:#999;text-transform:uppercase;letter-spacing:.6px;margin-bottom:4px}}
.card .val{{font-size:24px;font-weight:800;margin-bottom:3px}}
.card .sub{{font-size:12px;color:#bbb}}
.green .val{{color:#2e7d32}}.red .val{{color:#c62828}}.blue .val{{color:#1565c0}}
.orange .val{{color:#e65100}}.purple .val{{color:#6a1b9a}}
.alert{{border-radius:10px;padding:15px 18px;margin-bottom:16px;font-size:14px;line-height:1.8}}
.a-warn{{background:#fff8e1;border-left:5px solid #f9a825}}
.a-info{{background:#e3f2fd;border-left:5px solid #1565c0}}
.a-good{{background:#e8f5e9;border-left:5px solid #388e3c}}
.a-bad{{background:#fdecea;border-left:5px solid #c62828}}
.a-note{{background:#f3e5f5;border-left:5px solid #7b1fa2}}
.chart-box{{background:white;border-radius:12px;padding:18px;
            box-shadow:0 2px 10px rgba(0,0,0,0.07);margin-bottom:18px}}
table{{width:100%;border-collapse:collapse;background:white;border-radius:12px;overflow:hidden;
       box-shadow:0 2px 10px rgba(0,0,0,0.07);font-size:13px;margin-bottom:18px}}
th{{background:#1a237e;color:white;padding:11px 8px;text-align:center;font-size:11px;font-weight:600}}
td{{padding:10px 8px;text-align:center;border-bottom:1px solid #eee}}
tr:last-child td{{border:none}} tr:hover td{{background:#f5f5ff}}
.note-sm{{font-size:12px;color:#aaa;margin-top:8px}}
.rec{{background:#e8f5e9;border:2px solid #388e3c;border-radius:12px;padding:22px 26px;margin-bottom:18px}}
.rec h3{{color:#1b5e20;font-size:17px;margin-bottom:14px}}
.rec-grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:12px}}
.rec-item{{background:white;border-radius:8px;padding:12px 14px}}
.rec-item .rl{{font-size:11px;color:#888;margin-bottom:3px}}
.rec-item .rv{{font-size:21px;font-weight:800;color:#1b5e20}}
.steps ol{{padding-left:22px}}.steps li{{margin-bottom:12px;font-size:14px;line-height:1.8}}
.steps li b{{color:#1a237e}}
.chain{{display:flex;align-items:center;gap:0;margin:18px 0;flex-wrap:wrap}}
.chain-node{{background:white;border-radius:10px;padding:14px 18px;
             box-shadow:0 2px 8px rgba(0,0,0,0.07);text-align:center;min-width:150px}}
.chain-arrow{{font-size:24px;color:#1a237e;padding:0 8px;font-weight:800}}
.two-col{{display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-bottom:18px}}
/* Tabs */
.tab-bar{{display:flex;gap:4px;margin-bottom:0;flex-wrap:wrap}}
.tab-btn{{padding:8px 14px;border:none;background:#e0e0e0;border-radius:8px 8px 0 0;
          cursor:pointer;font-size:12px;font-weight:600;color:#555;transition:all .2s}}
.tab-btn.active{{background:#1a237e;color:white}}
.tab-btn:hover{{background:#c5cae9}}
.tab-panel{{display:none;background:white;border-radius:0 12px 12px 12px;padding:20px;
            box-shadow:0 2px 10px rgba(0,0,0,0.07);margin-bottom:18px}}
.tab-panel.active{{display:block}}
@media(max-width:700px){{.cards{{grid-template-columns:1fr 1fr}}.two-col{{grid-template-columns:1fr}}
  .chain{{flex-direction:column}}.chain-arrow{{transform:rotate(90deg)}}}}
</style>
</head>
<body>
<div class="page">

<h1>葡萄牙基金对冲方案</h1>
<p class="meta">Optimize Portugal Golden Opportunities Fund (PTOPZWHM0007) &middot; 数据截至2026年3月</p>

<!-- ═══ 一、持仓 ═══ -->
<div class="section">
<h2>一、你的持仓</h2>
<div class="cards">
  <div class="card green"><div class="lbl">买入成本</div><div class="val">&euro;{INITIAL_INV:,}</div>
    <div class="sub">2024年7月 &middot; NAV &euro;{FUND_ENTRY_PRICE:.2f}</div></div>
  <div class="card green"><div class="lbl">当前市值</div><div class="val">&euro;{FUND_VALUE:,}</div>
    <div class="sub">NAV &euro;{FUND_CURR_PRICE:.2f}</div></div>
  <div class="card purple"><div class="lbl">浮盈</div><div class="val">+&euro;{FUND_VALUE-INITIAL_INV:,}</div>
    <div class="sub">+{fund_gain:.1f}%</div></div>
  <div class="card orange"><div class="lbl">PSI20当前</div><div class="val">{PSI20_CURRENT:,}</div>
    <div class="sub">买入时{PSI20_ENTRY_ACT:,} &middot; 已涨{(PSI20_CURRENT/PSI20_ENTRY_ACT-1)*100:.0f}%</div></div>
</div>
<div class="chart-box"><div id="c1" style="height:320px"></div></div>
<p class="note-sm">你的基金高度跟踪PSI20 (R&sup2;=79%)。你本质上是在做多葡萄牙股市。</p>
<div class="alert a-warn">
  <b>你的担忧：</b>俄乌停战资金撤离、欧盟加息、美欧关税等<b>全欧系统性事件</b>导致大跌。
  你想在继续持有（计划5年）的同时买一份保险。
</div>
</div>

<!-- ═══ 二、历史证据 ═══ -->
<div class="section">
<h2>二、历史上基金每次急跌，IBEX跟不跟？</h2>
<div class="chart-box"><div id="c2" style="height:460px"></div></div>
<div class="alert a-info">
  <b style="color:#2e7d32">绿色阴影</b> = 基金急跌且IBEX同步下跌。
  <b style="color:#e65100">黄色阴影</b> = 基金急跌但IBEX没跟。
</div>

<p style="margin-bottom:12px">基金成立以来，<b>1周内跌幅超过3%</b>的全部<b>{n_total}次</b>事件：</p>
<table>
  <tr><th>#</th><th>时期</th><th>基金1周跌幅</th><th>IBEX同期</th><th>IBEX到达</th><th>关系</th></tr>
  {hist_rows}
</table>

<div class="alert a-good">
  <b>结论：{n_total}次急跌中{n_sync}次IBEX同步下跌</b>，仅{n_total-n_sync}次脱钩。<br>
  规律很清晰：全欧系统性冲击 → IBEX同步甚至跌得更凶；仅有的1次脱钩是葡萄牙本地波动（2024.7）。<br>
  <b>你担心的系统性风险场景中，IBEX Put {n_sync}/{n_total} = {n_sync/n_total*100:.0f}%有效。</b>
</div>
</div>

<!-- ═══ 三、为什么IBEX35 ═══ -->
<div class="section">
<h2>三、为什么用IBEX35做对冲工具</h2>
<div class="chain">
  <div class="chain-node">
    <div style="font-size:12px;color:#888">你持有的</div>
    <div style="font-size:18px;font-weight:800;color:#1565c0">葡萄牙基金</div>
  </div>
  <div class="chain-arrow">&rarr;</div>
  <div class="chain-node">
    <div style="font-size:12px;color:#888">高度跟踪</div>
    <div style="font-size:18px;font-weight:800;color:#ff7f0e">PSI20</div>
    <div style="font-size:11px;color:#2e7d32">R&sup2;=79%</div>
  </div>
  <div class="chain-arrow">&rarr;</div>
  <div class="chain-node" style="border:2px solid #c62828">
    <div style="font-size:12px;color:#c62828">PSI20无可用期权</div>
    <div style="font-size:11px;color:#888">Euronext Lisbon 价差5-10%</div>
  </div>
  <div class="chain-arrow">&rarr;</div>
  <div class="chain-node" style="border:2px solid #2e7d32">
    <div style="font-size:12px;color:#2e7d32">最佳替代</div>
    <div style="font-size:18px;font-weight:800;color:#e65100">IBEX35 Put</div>
    <div style="font-size:11px;color:#888">MEFF交易所 &middot; IBKR可交易</div>
  </div>
</div>
</div>

<!-- ═══ 四、远期Put的缺陷 ═══ -->
<div class="section">
<h2>四、为什么不能买远期Put一劳永逸</h2>
<div class="alert a-bad">
  如果买一张21个月的远期Put（行权价{rec_old['K']:,}点），放着不管会怎样？<br>
  <b>历史回测：{n_bt_zero}次模拟全部到期归零。</b>因为IBEX过去4年从7,261涨到18,000+，
  固定行权价被涨幅远远甩开。
</div>
<div class="chart-box"><div id="c3" style="height:420px"></div></div>
<p class="note-sm">
  蓝线=买入时IBEX，橙虚线=行权价(95%)，绿线=到期时IBEX。到期IBEX始终远高于行权价 → 全部归零。
</p>
<div class="alert a-warn">
  <b>所以必须定期滚仓</b>刷新行权价，不能买了放着不管。问题是多久滚一次？
</div>
</div>

<!-- ═══ 五、策略对比 + Tab事件详情 ═══ -->
<div class="section">
<h2>五、三种滚仓频率对比</h2>

<table>
  <tr><th style="text-align:left">策略</th><th>操作频率</th><th>年化成本</th><th>5年总成本</th></tr>
  {cost_rows}
</table>

<p style="margin-bottom:12px">
  <b>你持仓期内有{len(hold_events)}次急跌</b>，点击下面的Tab查看每次的详情——
  基金亏了多少、Put那边赚回来多少、相抵后净亏多少：
</p>

<div class="tab-bar">{tab_buttons}</div>
{tab_panels}

<div class="alert a-info">
  <b>同步事件中的平均覆盖率：</b>
  12月年滚 <b>{avg_cov_12m:.0f}%</b> &asymp;
  6月半年滚 <b>{avg_cov_6m:.0f}%</b> &asymp;
  3月季滚 <b>{avg_cov_3m:.0f}%</b><br>
  <b>三种频率保护效果接近，但5年成本差距巨大。年滚最划算。</b>
</div>

<div class="alert a-warn">
  <b>为什么覆盖率都不算高？</b><br>
  因为持仓期内没有发生真正的系统性崩盘——最大的一次（2025.4关税冲击）IBEX也只跌了12%。
  如果发生2022级别的危机（IBEX跌30%+），Put会深度价内，赚回的钱足以覆盖大部分基金损失。
  <b>你买的是灾难保险，不是小波动保险。</b>
</div>
</div>

<!-- ═══ 六、推荐方案 ═══ -->
<div class="section">
<h2>六、推荐方案</h2>
<div class="rec">
  <h3>{N_CONTRACTS}张 IBEX35 12个月ATM Put，每年到期前滚仓</h3>
  <div class="rec-grid">
    <div class="rec-item"><div class="rl">当前行权价</div><div class="rv">{rec['K']:,}点</div></div>
    <div class="rec-item"><div class="rl">每年成本</div><div class="rv">&euro;{rec['total']:,}</div></div>
    <div class="rec-item"><div class="rl">年化占比</div><div class="rv">{rec['annual']:.2f}%</div></div>
    <div class="rec-item"><div class="rl">操作频率</div><div class="rv">1次/年</div></div>
    <div class="rec-item"><div class="rl">IBEX跌15%赔付</div><div class="rv">&euro;{max(rec['K']-round(IBEX_CURRENT*0.85),0)*N_CONTRACTS:,}</div></div>
    <div class="rec-item"><div class="rl">IBEX跌30%赔付</div><div class="rv">&euro;{max(rec['K']-round(IBEX_CURRENT*0.70),0)*N_CONTRACTS:,}</div></div>
    <div class="rec-item"><div class="rl">5年总保费</div><div class="rv">&euro;{rec['total']*5:,}</div></div>
    <div class="rec-item"><div class="rl">5年占持仓比</div><div class="rv">{rec['total']*5/FUND_VALUE*100:.1f}%</div></div>
  </div>
</div>
<div class="chart-box"><div id="c5" style="height:360px"></div></div>
<p class="note-sm">
  横轴=12个月后IBEX点位。红虚线=不对冲的基金损益，绿实线=持有Put后的净损益。
  IBEX跌破{rec['K']:,}后绿线开始兜底。
</p>
</div>

<!-- ═══ 七、操作步骤 ═══ -->
<div class="section">
<h2>七、操作步骤</h2>
<div class="steps">
<ol>
  <li><b>IBKR开通欧洲期权权限</b><br>搜索 IBEX 35 Index Options (代码OI)，交易所MEFF。</li>
  <li><b>找到合约</b><br>到期月<b>2027年3月</b>，类型<b>Put</b>，行权价<b>{rec['K']:,}点</b> (当前ATM)。</li>
  <li><b>限价单买入{N_CONTRACTS}张</b><br>理论价约&euro;{rec['price']:,.0f}/张，总预算约&euro;{rec['total']:,}。用Limit Order。</li>
  <li><b>年中看一个指标</b><br>IBEX涨超20% (超{round(IBEX_CURRENT*1.2):,}点) 则考虑提前换仓。否则不管。</li>
  <li><b>到期前1个月滚仓</b><br>卖出旧Put，买入新的12个月ATM Put。周而复始。</li>
</ol>
</div>
</div>

<!-- ═══ 八、局限性 ═══ -->
<div class="section">
<h2>八、局限性</h2>
<div class="alert a-warn">
  <ol style="margin:0 0 0 18px">
    <li><b>脱钩期无法保护：</b>葡萄牙本地问题导致基金跌、IBEX不跌时，Put不赔。历史上仅出现1次。</li>
    <li><b>保费是确定支出：</b>每年&euro;{rec['total']:,}，5年不出事白花&euro;{rec['total']*5:,}。</li>
    <li><b>中等回调覆盖有限：</b>IBEX只跌5-10%时，Put覆盖率10-30%。真正有力需跌20%+。</li>
    <li><b>IV波动影响成本：</b>市场恐慌时Put更贵，尽量在平静期滚仓。</li>
  </ol>
</div>
<div class="alert a-note">
  你防的是系统性暴跌。历史证明这类事件中IBEX同步率达{n_sync/n_total*100:.0f}%，
  且跌幅通常远超10%。<b>你买的是灾难保险，不是小波动保险。</b>
</div>
</div>

<!-- ═══ 九、总结 ═══ -->
<div class="section">
<h2>九、总结</h2>
<div class="alert a-note" style="font-size:15px;line-height:2">
  <b style="font-size:17px;color:#4a148c">16张 IBEX35 12个月ATM Put，每年滚仓一次</b><br><br>
  历史{n_total}次基金急跌中IBEX同步率<b>{n_sync/n_total*100:.0f}%</b>，证明IBEX Put在系统性风险中有效。<br>
  远期Put在牛市中失效（回测全归零），必须定期滚仓。<br>
  季度/半年/年度三种频率保护效果接近，<b>12个月年滚成本最低、操作最简</b>。<br><br>
  年化成本<b>{rec['annual']:.2f}%</b>，5年约<b>&euro;{rec['total']*5:,}</b>，每年操作1次。<br>
  用确定的保险费，换取对全欧系统性暴跌的保护。
</div>
</div>

</div>

<script>
// Charts
Plotly.newPlot('c1',__C1__.data,__C1__.layout,{{responsive:true}});
Plotly.newPlot('c2',__C2__.data,__C2__.layout,{{responsive:true}});
Plotly.newPlot('c3',__C3__.data,__C3__.layout,{{responsive:true}});
Plotly.newPlot('c5',__C5__.data,__C5__.layout,{{responsive:true}});

// Zoom charts data
var zoomData = __ZOOM_DATA__;

// Tab switching
function showTab(idx) {{
  document.querySelectorAll('.tab-btn').forEach((b,i) => b.classList.toggle('active', i===idx));
  document.querySelectorAll('.tab-panel').forEach((p,i) => {{
    p.classList.toggle('active', i===idx);
    if (i===idx && zoomData[idx]) {{
      var divId = 'zoom_'+idx;
      var el = document.getElementById(divId);
      if (el && !el.dataset.rendered) {{
        Plotly.newPlot(divId, zoomData[idx].data, zoomData[idx].layout, {{responsive:true}});
        el.dataset.rendered = '1';
      }}
    }}
  }});
}}
// Render first tab's chart
if (zoomData[0]) {{
  Plotly.newPlot('zoom_0', zoomData[0].data, zoomData[0].layout, {{responsive:true}});
  document.getElementById('zoom_0').dataset.rendered = '1';
}}
</script>
</body></html>"""

    html = html.replace('__C1__', c1).replace('__C2__', c2).replace('__C3__', c3).replace('__C5__', c5)
    # Zoom data: array of chart JSON objects
    zoom_json_str = '[' + ','.join(c if c else 'null' for c in zoom_js_data) + ']'
    html = html.replace('__ZOOM_DATA__', zoom_json_str)
    return html


def main():
    print('加载数据...')
    fund_df, ibex_df, psi_df = load_data()
    print(f'数据：基金{len(fund_df)}条，IBEX{len(ibex_df)}条，PSI20{len(psi_df)}条')

    print('分析...')
    res = analyze(fund_df, ibex_df, psi_df)
    n_ev = len(res['events'])
    n_sync = sum(1 for e in res['events'] if e['sync'])
    n_hold = sum(1 for e in res['events'] if e['in_hold'])
    print(f'完成：{len(res["df"])}交易日')
    print(f'1周跌>3%事件：{n_ev}次（{n_sync}次同步），其中持仓期内{n_hold}次')
    print(f'21M回测：{len(res["backtest"])}次，{sum(1 for b in res["backtest"] if b["payoff"]==0)}次零赔付')

    print('生成报告...')
    html = generate_html(fund_df, psi_df, res)
    out = os.path.join(DATA_DIR, 'hedge_final.html')
    with open(out, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f'报告已生成: {out}')

    import subprocess, platform
    if platform.system() == 'Darwin':
        subprocess.run(['open', out])


if __name__ == '__main__':
    main()
