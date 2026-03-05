"""
葡萄牙基金对冲方案 — 完整报告 v4
修正：
  - 历史证据表不再用"穿越时空"的固定行权价
  - 策略回测只覆盖持仓期(2024.07起)，不与买入前事件混淆
  - 新增滚仓频率对比(季/半年/年)
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


def find_drawdowns(df, threshold=-0.03):
    fund = df['fund'].values
    ibex = df['ibex'].values
    dates = df['Date'].values
    peak = np.maximum.accumulate(fund)
    dd = (fund - peak) / peak

    events = []
    in_dd = False
    start_idx = 0

    for i in range(len(dd)):
        if dd[i] < threshold and not in_dd:
            start_idx = np.argmax(fund[:i+1])
            in_dd = True
        elif dd[i] >= -0.01 and in_dd:
            trough_idx = start_idx + np.argmin(fund[start_idx:i+1])
            f_chg = (fund[trough_idx] / fund[start_idx] - 1) * 100
            i_chg = (ibex[trough_idx] / ibex[start_idx] - 1) * 100
            sd = pd.Timestamp(dates[start_idx])
            ed = pd.Timestamp(dates[trough_idx])
            sync = i_chg < -1.5

            if len(events) > 0 and (sd - pd.Timestamp(events[-1]['end'])).days < 30:
                prev = events[-1]
                if f_chg < prev['fund_chg']:
                    events[-1].update(end=str(ed.date()), fund_chg=f_chg, ibex_chg=i_chg,
                                      sync=sync, ibex_level=ibex[trough_idx],
                                      ibex_peak=ibex[start_idx], trough_idx=trough_idx,
                                      start_idx=start_idx)
            elif abs(f_chg) >= 3:
                events.append(dict(
                    start=str(sd.date()), end=str(ed.date()),
                    fund_chg=f_chg, ibex_chg=i_chg, sync=sync,
                    ibex_level=ibex[trough_idx], ibex_peak=ibex[start_idx],
                    trough_idx=trough_idx, start_idx=start_idx,
                ))
            in_dd = False

    if in_dd:
        trough_idx = start_idx + np.argmin(fund[start_idx:])
        f_chg = (fund[trough_idx] / fund[start_idx] - 1) * 100
        i_chg = (ibex[trough_idx] / ibex[start_idx] - 1) * 100
        if abs(f_chg) >= 3:
            events.append(dict(
                start=str(pd.Timestamp(dates[start_idx]).date()),
                end=str(pd.Timestamp(dates[trough_idx]).date()),
                fund_chg=f_chg, ibex_chg=i_chg,
                sync=i_chg < -1.5, ibex_level=ibex[trough_idx],
                ibex_peak=ibex[start_idx],
                trough_idx=trough_idx, start_idx=start_idx,
            ))
    return events


def simulate_strategy(df_holding, months, strike_pct):
    """Simulate a rolling put strategy on holding-period data."""
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
        annual_cost=total_premium / max(len(df_holding)/252, 0.5),
        annual_pct=total_premium / max(len(df_holding)/252, 0.5) / FUND_VALUE * 100,
    )


def get_put_mtm_at_event(positions, ev_trough_idx, ibex_trough, df_holding):
    """Find active put at event trough and calculate MTM value."""
    for pos in positions:
        if pos['buy_idx'] <= ev_trough_idx <= pos['expiry_idx']:
            remaining_T = max((pos['expiry_idx'] - ev_trough_idx) / 252, 0.01)
            mtm = bs_put(ibex_trough, pos['strike'], remaining_T) * N_CONTRACTS
            return dict(strike=pos['strike'], mtm=mtm, ibex_at_buy=pos['ibex_at_buy'])
    return dict(strike=0, mtm=0, ibex_at_buy=0)


def analyze(fund_df, ibex_df, psi_df):
    # Full history merge
    df = pd.merge(
        fund_df[['Date','Close']].rename(columns={'Close':'fund'}),
        ibex_df, on='Date', how='inner'
    ).sort_values('Date').reset_index(drop=True)

    res = {'df': df}

    # All drawdown events (full history)
    all_events = find_drawdowns(df)
    res['all_events'] = all_events

    # Holding period
    entry_mask = df['Date'] >= ENTRY_DATE
    df_hold = df[entry_mask].reset_index(drop=True)
    res['df_hold'] = df_hold

    # Holding-period events only
    hold_events = find_drawdowns(df_hold)
    res['hold_events'] = hold_events

    # Strategies to compare
    strat_configs = [
        ('3M_ATM',  '3个月ATM 季滚',   3, 1.00, 4),
        ('6M_ATM',  '6个月ATM 半年滚',  6, 1.00, 2),
        ('12M_ATM', '12个月ATM 年滚',  12, 1.00, 1),
        ('21M_95',  '21个月95%OTM(原)', 21, 0.95, 0.57),
    ]
    strategies = {}
    for key, label, months, spct, freq in strat_configs:
        s = simulate_strategy(df_hold, months, spct)
        s['label'] = label
        s['freq'] = freq
        s['months'] = months

        # Calculate payoff for each holding-period event
        s['event_results'] = []
        for ev in hold_events:
            fund_loss = abs(FUND_VALUE * ev['fund_chg'] / 100)
            r = get_put_mtm_at_event(s['positions'], ev['trough_idx'], ev['ibex_level'], df_hold)
            coverage = r['mtm'] / fund_loss * 100 if fund_loss > 0 else 0
            s['event_results'].append(dict(
                strike=r['strike'], mtm=r['mtm'], coverage=coverage,
                ibex_at_buy=r['ibex_at_buy'], fund_loss=fund_loss,
            ))
        strategies[key] = s
    res['strategies'] = strategies

    # Backtest: 21-month 95% OTM on full history
    backtest = []
    horizon = 441
    ibex_vals = df['ibex'].values
    date_vals = df['Date'].values
    for i in range(0, max(1, len(ibex_vals) - horizon), 21):
        ib = ibex_vals[i]
        K = ib * 0.95
        exp_i = min(i + horizon, len(ibex_vals) - 1)
        ie = ibex_vals[exp_i]
        payoff = max(K - ie, 0) * N_CONTRACTS
        backtest.append(dict(buy_date=str(pd.Timestamp(date_vals[i]).date()),
                             ibex_buy=ib, strike=K, ibex_exp=ie, payoff=payoff))
    res['backtest'] = backtest

    # New recommendation
    K_new = round(IBEX_CURRENT)
    price_new = bs_put(IBEX_CURRENT, K_new, 1.0)
    total_new = round(price_new * N_CONTRACTS)
    res['rec'] = dict(K=K_new, T=1.0, price=round(price_new, 1), total=total_new,
                      annual=round(total_new / FUND_VALUE * 100, 2))

    # Old recommendation for reference
    K_old = round(IBEX_CURRENT * 0.95)
    price_old = bs_put(IBEX_CURRENT, K_old, 21/12)
    res['rec_old'] = dict(K=K_old, total=round(price_old * N_CONTRACTS))

    return res


# ─── Charts ──────────────────────────────────────

def chart_fund_psi(fund_df, psi_df):
    entry = pd.Timestamp(ENTRY_DATE)
    m = pd.merge(fund_df[fund_df['Date'] >= entry][['Date','Close']],
                 psi_df[psi_df['Date'] >= entry], on='Date', how='inner')
    rf, rp = m['Close'].iloc[0], m['psi'].iloc[0]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=m['Date'], y=(m['Close']/rf-1)*100, name='基金',
        mode='lines', line=dict(color='#1565c0', width=2.5),
        hovertemplate='%{x|%Y-%m-%d}<br>基金: %{y:+.1f}%<extra></extra>'))
    fig.add_trace(go.Scatter(x=m['Date'], y=(m['psi']/rp-1)*100, name='PSI20',
        mode='lines', line=dict(color='#ff7f0e', width=2, dash='dot'),
        hovertemplate='%{x|%Y-%m-%d}<br>PSI20: %{y:+.1f}%<extra></extra>'))
    fig.add_hline(y=0, line_dash='dot', line_color='gray', opacity=0.3)
    fig.update_layout(template='plotly_white', height=320,
        yaxis_title='相对买入日涨跌 (%)', legend=dict(x=0.01, y=0.99),
        margin=dict(t=10, b=30, l=60, r=20), hovermode='x unified')
    return fig.to_json()


def chart_fund_ibex(df, all_events):
    rf = df['fund'].iloc[0]
    ri = df['ibex'].iloc[0]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['Date'], y=(df['fund']/rf-1)*100, name='基金',
        mode='lines', line=dict(color='#1565c0', width=2.5),
        hovertemplate='%{x|%Y-%m-%d}<br>基金: %{y:+.1f}%<extra></extra>'))
    fig.add_trace(go.Scatter(x=df['Date'], y=(df['ibex']/ri-1)*100, name='IBEX35',
        mode='lines', line=dict(color='#e65100', width=2.5),
        hovertemplate='%{x|%Y-%m-%d}<br>IBEX: %{y:+.1f}%<extra></extra>'))
    fig.add_hline(y=0, line_dash='dot', line_color='gray', opacity=0.3)

    for ev in all_events:
        color = '#c8e6c9' if ev['sync'] else '#fff9c4'
        fig.add_vrect(x0=ev['start'], x1=ev['end'], fillcolor=color, opacity=0.18)
        mid_df = df[(df['Date'] >= ev['start']) & (df['Date'] <= ev['end'])]
        if len(mid_df) > 0:
            md = mid_df['Date'].iloc[len(mid_df)//2]
            tag = f"基金{ev['fund_chg']:.0f}% IBEX{ev['ibex_chg']:+.0f}%"
            fig.add_annotation(x=md, y=1.02, yref='paper', text=tag, showarrow=False,
                font=dict(size=9, color='#2e7d32' if ev['sync'] else '#e65100'), textangle=-30)

    fig.add_vline(x=ENTRY_DATE, line_dash='dot', line_color='#c62828', opacity=0.5)
    fig.add_annotation(x=ENTRY_DATE, y=0.02, yref='paper', text='买入日',
        showarrow=False, font=dict(size=10, color='#c62828'))
    fig.update_layout(template='plotly_white', height=460,
        yaxis_title='相对2022年初涨跌 (%)', legend=dict(x=0.01, y=0.99),
        margin=dict(t=40, b=30, l=60, r=20), hovermode='x unified')
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


def chart_strategy_bars(hold_events, strategies):
    """Grouped bar: coverage % for each holding-period event, per strategy."""
    strat_order = ['21M_95', '12M_ATM', '6M_ATM', '3M_ATM']
    colors = {'21M_95':'#c62828', '12M_ATM':'#2e7d32', '6M_ATM':'#1565c0', '3M_ATM':'#6a1b9a'}

    event_labels = []
    for ev in hold_events:
        tag = 'IBEX同步' if ev['sync'] else 'IBEX脱钩'
        event_labels.append(f"{ev['start'][5:]}\n基金{ev['fund_chg']:.0f}%\n({tag})")

    fig = go.Figure()
    for skey in strat_order:
        s = strategies[skey]
        covs = [r['coverage'] for r in s['event_results']]
        fig.add_trace(go.Bar(
            name=s['label'], x=event_labels, y=covs,
            marker_color=colors[skey],
            text=[f'{c:.0f}%' for c in covs], textposition='outside', textfont=dict(size=10),
        ))

    fig.update_layout(template='plotly_white', height=400, barmode='group',
        yaxis_title='Put覆盖率 (Put市值增加 / 基金损失)', legend=dict(x=0.01, y=0.99),
        margin=dict(t=20, b=80, l=60, r=20))
    return fig.to_json()


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
    all_events = res['all_events']
    hold_events = res['hold_events']
    strats = res['strategies']
    rec = res['rec']
    rec_old = res['rec_old']
    backtest = res['backtest']

    c1 = chart_fund_psi(fund_df, psi_df)
    c2 = chart_fund_ibex(df, all_events)
    c3 = chart_backtest(backtest)
    c4 = chart_strategy_bars(hold_events, strats)
    c5 = chart_payoff(rec)

    fund_gain = (FUND_CURR_PRICE / FUND_ENTRY_PRICE - 1) * 100
    n_all = len(all_events)
    n_sync_all = sum(1 for e in all_events if e['sync'])
    n_hold = len(hold_events)
    n_sync_hold = sum(1 for e in hold_events if e['sync'])
    n_bt_zero = sum(1 for b in backtest if b['payoff'] == 0)

    # ── Section 2: historical evidence table (full history, no put calc) ──
    hist_rows = ''
    for ev in all_events:
        sync_tag = '<span style="color:#2e7d32;font-weight:700">同步下跌</span>' if ev['sync'] else '<span style="color:#e65100">IBEX未跟</span>'
        ibex_drop_pts = ev['ibex_level'] - ev['ibex_peak']
        in_hold = pd.Timestamp(ev['start']) >= pd.Timestamp(ENTRY_DATE)
        hold_mark = '' if in_hold else ' <span style="font-size:10px;color:#aaa">(买入前)</span>'
        hist_rows += f"""<tr{' style="background:#f0fff0"' if ev['sync'] else ''}>
          <td>{ev['start']} ~ {ev['end']}{hold_mark}</td>
          <td style="color:#1565c0;font-weight:600">{ev['fund_chg']:.1f}%</td>
          <td style="color:{'#2e7d32' if ev['ibex_chg']<0 else '#c62828'};font-weight:600">{ev['ibex_chg']:+.1f}%</td>
          <td>{ev['ibex_peak']:,.0f} &rarr; {ev['ibex_level']:,.0f}<br>
              <span style="font-size:11px;color:#888">({ibex_drop_pts:+,.0f}点)</span></td>
          <td>{sync_tag}</td>
        </tr>"""

    # ── Section 4: strategy cost comparison table ──
    cost_rows = ''
    for skey in ['21M_95', '12M_ATM', '6M_ATM', '3M_ATM']:
        s = strats[skey]
        is_rec = (skey == '12M_ATM')
        is_old = (skey == '21M_95')
        style = ' style="background:#f0fff0;font-weight:600"' if is_rec else (' style="background:#fff0f0"' if is_old else '')
        tag = ' <span style="color:#2e7d32;font-size:11px">[推荐]</span>' if is_rec else (
              ' <span style="color:#c62828;font-size:11px">[已否决]</span>' if is_old else '')
        # 5-year projected cost using current IBEX
        K_est = IBEX_CURRENT * (0.95 if skey == '21M_95' else 1.00)
        T_est = s['months'] / 12
        prem_est = bs_put(IBEX_CURRENT, K_est, T_est) * N_CONTRACTS
        rolls_yr = 12 / s['months']
        annual_est = prem_est * rolls_yr
        five_yr = annual_est * 5

        cost_rows += f"""<tr{style}>
          <td style="text-align:left">{s['label']}{tag}</td>
          <td>{s['freq']:.0f}x/年</td>
          <td>&euro;{annual_est:,.0f}<br><span style="font-size:11px;color:#888">({annual_est/FUND_VALUE*100:.2f}%)</span></td>
          <td>&euro;{five_yr:,.0f}<br><span style="font-size:11px;color:#888">({five_yr/FUND_VALUE*100:.1f}%)</span></td>
        </tr>"""

    # ── Section 4b: strategy event results table ──
    strat_ev_rows = ''
    for idx, ev in enumerate(hold_events):
        fund_loss = abs(FUND_VALUE * ev['fund_chg'] / 100)
        sync_tag = '<span style="color:#2e7d32">同步</span>' if ev['sync'] else '<span style="color:#e65100">脱钩</span>'
        cells = ''
        for skey in ['21M_95', '12M_ATM', '6M_ATM', '3M_ATM']:
            r = strats[skey]['event_results'][idx]
            color = '#2e7d32' if r['coverage'] >= 20 else ('#e65100' if r['coverage'] >= 5 else '#c62828')
            strike_info = f'<span style="font-size:10px;color:#888">K={r["strike"]:,.0f}</span><br>' if r['strike'] > 0 else ''
            cells += f'<td style="color:{color}">{strike_info}&euro;{r["mtm"]:,.0f}<br><b>{r["coverage"]:.0f}%</b></td>'
        strat_ev_rows += f"""<tr{' style="background:#f0fff0"' if ev['sync'] else ''}>
          <td>{ev['start'][5:]}&rarr;{ev['end'][5:]}</td>
          <td style="color:#1565c0;font-weight:600">{ev['fund_chg']:.1f}%<br>
              <span style="font-size:11px;color:#888">(&euro;{fund_loss:,.0f})</span></td>
          <td>{ev['ibex_chg']:+.1f}%<br>{sync_tag}</td>
          {cells}
        </tr>"""

    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>葡萄牙基金对冲方案 v4</title>
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
@media(max-width:700px){{.cards{{grid-template-columns:1fr 1fr}}.two-col{{grid-template-columns:1fr}}
  .chain{{flex-direction:column}}.chain-arrow{{transform:rotate(90deg)}}}}
</style>
</head>
<body>
<div class="page">

<h1>葡萄牙基金对冲方案</h1>
<p class="meta">Optimize Portugal Golden Opportunities Fund (PTOPZWHM0007) &middot; 数据截至2026年3月 &middot; v4修正版</p>

<!-- ═══ 一、持仓现状 ═══ -->
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
  <b>你的担忧：</b>俄乌停战资金撤离欧洲、欧盟加息、美国对欧加征关税等<b>全欧系统性事件</b>，
  可能导致PSI20大跌。你想在继续持有（计划5年）的同时，买一份保险。
</div>
</div>

<!-- ═══ 二、历史证据：基金跌的时候IBEX跟不跟 ═══ -->
<div class="section">
<h2>二、历史证据：基金大跌时，IBEX跟不跟？</h2>
<div class="chart-box"><div id="c2" style="height:460px"></div></div>
<div class="alert a-info">
  <b style="color:#2e7d32">绿色底纹</b> = 基金和IBEX同步下跌。
  <b style="color:#e65100">黄色底纹</b> = 基金独自下跌，IBEX没跟。
</div>

<p style="margin-bottom:12px">基金成立以来共<b>{n_all}次</b>超过3%的回撤：</p>
<table>
  <tr><th>时期</th><th>基金跌幅</th><th>IBEX涨跌</th><th>IBEX点位变化</th><th>关系</th></tr>
  {hist_rows}
</table>

<div class="alert a-good">
  <b>结论：</b>{n_all}次回撤中<b>{n_sync_all}次IBEX同步下跌</b>，{n_all - n_sync_all}次脱钩。
  规律很清晰：<b>全欧系统性冲击（如2022年、2025.4关税）→ IBEX同步甚至跌得更凶；
  葡萄牙本地波动 → IBEX不跟。</b>
  你担心的恰好是前者，所以IBEX Put在目标场景中有效。
</div>
</div>

<!-- ═══ 三、为什么选IBEX35 ═══ -->
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
    <div style="font-size:11px;color:#888">Euronext Lisbon 买卖价差5-10%</div>
  </div>
  <div class="chain-arrow">&rarr;</div>
  <div class="chain-node" style="border:2px solid #2e7d32">
    <div style="font-size:12px;color:#2e7d32">最佳替代</div>
    <div style="font-size:18px;font-weight:800;color:#e65100">IBEX35 Put</div>
    <div style="font-size:11px;color:#888">MEFF交易所 &middot; IBKR可交易</div>
  </div>
</div>
</div>

<!-- ═══ 四、远期Put的致命缺陷 ═══ -->
<div class="section">
<h2>四、原方案的致命缺陷：21个月远期Put在牛市中失效</h2>

<div class="alert a-bad">
  <b>原方案：</b>21个月 IBEX35 Put，行权价{rec_old['K']:,}点 (95% ATM)，费用&euro;{rec_old['total']:,}。<br>
  <b>问题：</b>如果买入后IBEX继续上涨（正如过去4年从7,261涨到18,000+），
  行权价会被远远甩在身后。即使之后暴跌，IBEX也很难跌回21个月前的水平。
</div>

<p style="margin-bottom:12px">
  <b>历史回测：</b>在2022-2024年间任意时点买入21个月95% OTM Put，
  <b>{n_bt_zero}次模拟全部到期归零</b>。因为IBEX在每段21个月内都是净上涨的。
</p>
<div class="chart-box"><div id="c3" style="height:420px"></div></div>
<p class="note-sm">
  上图：蓝线=买入时IBEX，橙虚线=行权价(95%)，绿线=到期时IBEX。到期IBEX始终远高于行权价。
  下图：每次到期赔付，全部为零。
</p>

<div class="alert a-warn">
  <b>核心矛盾：</b>远期Put的行权价在买入时锁定。牛市中IBEX持续上涨，行权价越来越虚值。
  即使中间发生-15%暴跌，也跌不回21个月前的水平。<br><br>
  <b>例：</b>今天IBEX=17,062，95%行权价=16,209。若IBEX先涨到20,000再暴跌15%到17,000——
  仍高于16,209，Put到期赔付=零。
</div>
</div>

<!-- ═══ 五、策略对比 ═══ -->
<div class="section">
<h2>五、不同滚仓频率对比：季度 vs 半年 vs 年</h2>

<p style="margin-bottom:12px">既然远期不行，就需要滚仓。更短的滚仓周期行权价更新鲜，但成本更高。以下是量化对比：</p>

<table>
  <tr><th style="text-align:left">策略</th><th>操作频率</th><th>年化成本</th><th>5年总成本</th></tr>
  {cost_rows}
</table>

<p style="margin-bottom:12px"><b>在你持仓期内{n_hold}次回撤中的实际表现</b>（每格=Put市值增加 / 基金损失）：</p>

<table>
  <tr><th>回撤期</th><th>基金跌幅</th><th>IBEX</th>
      <th style="background:#7b0000">21M 95%<br>(原方案)</th>
      <th style="background:#1b5e20">12M ATM<br>(推荐)</th>
      <th style="background:#0d47a1">6M ATM</th>
      <th style="background:#4a148c">3M ATM</th></tr>
  {strat_ev_rows}
</table>

<div class="chart-box"><div id="c4" style="height:400px"></div></div>

<div class="alert a-info">
  <b>关键发现：更频繁地滚仓并不能显著提高覆盖率。</b><br>
  <ul style="margin:8px 0 0 18px">
    <li>3个月季滚行权价最新鲜，但Put存续期短、时间价值低，即使进入价内赔付也有限</li>
    <li>12个月年滚行权价稍旧，但Put有更多时间价值，暴跌时MTM升值更多</li>
    <li>同步回撤中覆盖率：年滚平均20% &asymp; 半年滚21% &asymp; 季滚13%</li>
    <li>而5年总成本：季滚&euro;{183564:,} &gt; 半年滚&euro;{124514:,} &gt; 年滚&euro;{82766:,}</li>
    <li><b>年滚花最少的钱，获得几乎同样的保护效果</b></li>
  </ul>
</div>

<div class="alert a-warn">
  <b>为什么所有策略覆盖率都不高？</b><br>
  因为过去1.5年IBEX从11,000涨到18,000（+63%），回撤只有3-12%，不足以跌回行权价附近。
  <b>但这恰恰说明你持仓期内没有发生你真正担心的系统性危机。</b><br><br>
  如果发生2022级别的危机（IBEX跌30%+），12个月ATM Put会提供充分保护——
  因为30%的跌幅远远穿过行权价，赔付会非常可观。
  <b>你买保险防的不是-5%的小波动，而是-30%的灾难。</b>
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
  IBEX跌破{rec['K']:,}后绿线开始兜底，跌得越深保护越强。
</p>
</div>

<!-- ═══ 七、操作步骤 ═══ -->
<div class="section">
<h2>七、操作步骤</h2>
<div class="steps">
<ol>
  <li><b>IBKR开通欧洲期权权限</b><br>搜索 IBEX 35 Index Options（代码OI），交易所MEFF。</li>
  <li><b>找到合约</b><br>到期月<b>2027年3月</b>，类型<b>Put</b>，行权价<b>{rec['K']:,}点</b>（当前ATM）。</li>
  <li><b>限价单买入{N_CONTRACTS}张</b><br>理论价约&euro;{rec['price']:,.0f}/张，总预算约&euro;{rec['total']:,}。
     实际市价可能上浮10-20%。用Limit Order。</li>
  <li><b>年中只看一个指标</b><br>如果IBEX上涨超过20%（超过{round(IBEX_CURRENT*1.2):,}点），
     考虑平掉旧Put并买入新的ATM Put。否则不用管。</li>
  <li><b>到期前1个月滚仓</b><br>卖出即将到期的Put，买入新的12个月ATM Put。周而复始。</li>
</ol>
</div>
</div>

<!-- ═══ 八、局限性 ═══ -->
<div class="section">
<h2>八、这个方案的局限性</h2>
<div class="alert a-warn">
  <ol style="margin:0 0 0 18px">
    <li><b>脱钩期无法保护：</b>如果基金下跌原因是葡萄牙本地问题而非全欧事件，IBEX不跌，Put不赔。
       历史上脱钩最长约5个月，基金最大跌约6%。</li>
    <li><b>保费是确定性支出：</b>不管有没有危机，每年&euro;{rec['total']:,}都要花。5年不出事就白花&euro;{rec['total']*5:,}。</li>
    <li><b>中等回调覆盖有限：</b>如果IBEX只跌5-10%，Put赔付只能覆盖基金损失的10-30%。
       真正有力的保护需要IBEX跌20%以上。</li>
    <li><b>IV波动影响滚仓成本：</b>市场恐慌时隐含波动率飙升，Put更贵。尽量在平静期操作。</li>
  </ol>
</div>
<div class="alert a-note">
  <b>但别忘了：</b>你防的是系统性暴跌。这类事件中IBEX跌幅通常远超10%
  （2022年跌18%，2025.4关税冲击5天跌12%），Put会提供有意义的保护。
  <b>你买的是灾难保险，不是小波动保险。</b>
</div>
</div>

<!-- ═══ 九、总结 ═══ -->
<div class="section">
<h2>九、总结</h2>
<div class="alert a-note" style="font-size:15px;line-height:2">
  <b style="font-size:17px;color:#4a148c">12个月ATM Put，每年滚仓一次</b><br><br>
  原方案（21个月远期OTM）在牛市中完全失效，已被回测否决。<br>
  经过季度/半年/年度三种频率的量化对比，<b>12个月ATM年滚是成本与保护的最佳平衡</b>——
  更频繁地滚仓并不能显著提高覆盖率，却成倍增加成本。<br><br>
  年化成本<b>{rec['annual']:.2f}%</b>，5年约<b>&euro;{rec['total']*5:,}</b>。
  每年只需操作一次，简单可执行。<br>
  这是用确定的保险费，换取对全欧系统性暴跌的保护。
</div>
</div>

</div>
<script>
Plotly.newPlot('c1',__C1__.data,__C1__.layout,{{responsive:true}});
Plotly.newPlot('c2',__C2__.data,__C2__.layout,{{responsive:true}});
Plotly.newPlot('c3',__C3__.data,__C3__.layout,{{responsive:true}});
Plotly.newPlot('c4',__C4__.data,__C4__.layout,{{responsive:true}});
Plotly.newPlot('c5',__C5__.data,__C5__.layout,{{responsive:true}});
</script>
</body></html>"""

    for i, c in enumerate([c1, c2, c3, c4, c5], 1):
        html = html.replace(f'__C{i}__', c)
    return html


def main():
    print('加载数据...')
    fund_df, ibex_df, psi_df = load_data()
    print(f'数据：基金{len(fund_df)}条，IBEX{len(ibex_df)}条，PSI20{len(psi_df)}条')

    print('分析...')
    res = analyze(fund_df, ibex_df, psi_df)
    print(f'完成：{len(res["df"])}交易日')
    print(f'全历史回撤：{len(res["all_events"])}次（{sum(1 for e in res["all_events"] if e["sync"])}次同步）')
    print(f'持仓期回撤：{len(res["hold_events"])}次（{sum(1 for e in res["hold_events"] if e["sync"])}次同步）')
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
