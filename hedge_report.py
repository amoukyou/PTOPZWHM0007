"""
葡萄牙基金对冲决策报告 v2 — 基于完整历史数据（2022-2026）
"""

import os, sys, math, json
import numpy as np
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots

sys.stdout.reconfigure(encoding='utf-8')
DATA_DIR = os.path.dirname(os.path.abspath(__file__))

# ═══════ 核心参数（实测数据） ══════════════════════════════
FUND_VALUE      = 651_000       # 当前估值（用户确认，截至2026-03-04）
INITIAL_INV     = 507_000       # 用户确认初始投资
FUND_ENTRY_PRICE= 13.3534       # 2024-07-16 Yahoo NAV（入场日可能有1-2天偏差）
FUND_UNITS      = round(INITIAL_INV / FUND_ENTRY_PRICE)   # ≈37,967份
# 当前NAV：Yahoo最新是2026-02-27的17.583，3月4日市场下跌后FT约17.15
# 两者均可用；报告用用户确认的€651,000为准
FUND_CURR_PRICE = round(FUND_VALUE / FUND_UNITS, 4)       # 反推≈17.15

PSI20_ENTRY_USER= 7_400         # 用户印象
PSI20_ENTRY_ACT = 6_711         # 2024-07-16 实测
PSI20_CURRENT   = 8_862         # 2026-03-04
PSI20_TARGET    = 8_000

IBEX_CURRENT    = 17_062        # 2026-03-04
IBEX_IMPLIED_VOL= 0.185
ECB_RATE        = 0.026
ESTX_CURRENT    = 6_138         # Euro Stoxx 50，2026-03-04
ESTX_IMPLIED_VOL= 0.180

# ── 修正后参数（用Close价格计算，消除时间错位）──────────────
# 之前用Open价导致beta=0.24，现在修正为正确值
BETA_FUND_PSI20 = 0.6271        # 基金 vs PSI20
BETA_FUND_IBEX  = 0.4230        # 基金 vs IBEX35
BETA_FUND_ESTX  = 0.3696        # 基金 vs Euro Stoxx 50
BETA_IBEX_PSI20 = 0.6897        # IBEX35 vs PSI20
BETA_ESTX_PSI20 = 0.5741        # ESTX50 vs PSI20
CORR_FUND_PSI20 = 0.8863        # R²=78.6%
CORR_FUND_IBEX  = 0.6449        # R²=41.6%
CORR_FUND_ESTX  = 0.5775        # R²=33.4%
CORR_IBEX_PSI20 = 0.78          # IBEX35 vs PSI20（西班牙与葡萄牙市场历史估算）

# PSI20=8000 场景推算
PSI20_DROP_PCT  = (PSI20_TARGET - PSI20_CURRENT) / PSI20_CURRENT  # -9.74%
FUND_DROP_PCT   = BETA_FUND_PSI20 * PSI20_DROP_PCT                # -6.10%
FUND_LOSS_EUR   = FUND_VALUE * FUND_DROP_PCT                       # -€39,707
IBEX_AT_8000    = IBEX_CURRENT * (1 + BETA_IBEX_PSI20 * PSI20_DROP_PCT)  # ~15,917
ESTX_AT_8000    = ESTX_CURRENT * (1 + BETA_ESTX_PSI20 * PSI20_DROP_PCT)  # ~5,796

# PSI20=7000 极端场景推算
PSI20_7000      = 7_000
PSI20_DROP_7000 = (PSI20_7000 - PSI20_CURRENT) / PSI20_CURRENT    # -21.01%
FUND_DROP_7000  = BETA_FUND_PSI20 * PSI20_DROP_7000                # -13.17%
FUND_LOSS_7000  = FUND_VALUE * FUND_DROP_7000                      # -€85,722
IBEX_AT_7000    = IBEX_CURRENT * (1 + BETA_IBEX_PSI20 * PSI20_DROP_7000)  # ~14,591

N_CONTRACTS     = 16  # round(FUND_VALUE * BETA_FUND_IBEX / IBEX_CURRENT)
N_CONTRACTS_ESTX= 45  # round(FUND_VALUE * BETA_FUND_ESTX / ESTX_CURRENT)
# ═══════════════════════════════════════════════════════════


def norm_cdf(x):
    return (1 + math.erf(x / math.sqrt(2))) / 2

def bs_put(S, K, T, r, sigma):
    if T <= 0: return max(K - S, 0), -1.0
    d1 = (math.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*math.sqrt(T))
    d2 = d1 - sigma*math.sqrt(T)
    price = K*math.exp(-r*T)*norm_cdf(-d2) - S*norm_cdf(-d1)
    delta = norm_cdf(d1) - 1
    return price, delta


def load_data():
    fund = pd.read_csv(os.path.join(DATA_DIR, 'PTOPZWHM0007_daily_2022-2026.csv'), parse_dates=['Date'])
    fund = fund.sort_values('Date').reset_index(drop=True)

    psi = yf.download('PSI20.LS', start='2022-01-01', end='2026-03-06', progress=False)
    psi.columns = [c[0] for c in psi.columns]
    psi = psi.reset_index()[['Date','Close']].rename(columns={'Close':'psi'})
    psi['Date'] = psi['Date'].dt.normalize()

    ibex = yf.download('^IBEX', start='2022-01-01', end='2026-03-06', progress=False)
    ibex.columns = [c[0] for c in ibex.columns]
    ibex = ibex.reset_index()[['Date','Close']].rename(columns={'Close':'ibex'})
    ibex['Date'] = ibex['Date'].dt.normalize()

    return fund, ibex, psi


def calc_options():
    """计算 IBEX35 和 ESTX50 两套方案"""
    rows = []
    instruments = [
        ('IBEX35', IBEX_CURRENT, IBEX_IMPLIED_VOL, IBEX_AT_8000, N_CONTRACTS),
        ('ESTX50', ESTX_CURRENT, ESTX_IMPLIED_VOL, ESTX_AT_8000, N_CONTRACTS_ESTX),
    ]
    for inst, S, vol, S_at_8000, n in instruments:
        for T_months, exp_label in [(15, 'Jun 2027'), (21, 'Dec 2027')]:
            T = T_months / 12
            for k_pct, k_label in [(1.00,'ATM'), (0.95,'95%'), (0.90,'90%')]:
                K = round(S * k_pct)
                price, _ = bs_put(S, K, T, ECB_RATE, vol)
                total_prem  = price * n
                annual_drag = total_prem / FUND_VALUE / T * 100
                payoff_8000 = max(K - S_at_8000, 0) * n
                coverage    = min(payoff_8000 / abs(FUND_LOSS_EUR) * 100, 100)
                net         = FUND_LOSS_EUR + payoff_8000 - total_prem
                rows.append(dict(
                    inst=inst, exp=exp_label, T=T, k_pct=k_pct, k_label=k_label, K=K,
                    price=round(price,1), n=n,
                    total_prem=round(total_prem), annual_drag=round(annual_drag,2),
                    payoff_8000=round(payoff_8000), coverage=round(coverage,1),
                    net=round(net),
                ))
    return rows


# ─── 图1：基金完整历史 + PSI20 ────────────────────────────────
def chart_full_history(fund_df, psi_df):
    fig = make_subplots(rows=2, cols=1,
        subplot_titles=('基金NAV完整走势（2022年至今）', 'PSI20指数走势（同期）'),
        vertical_spacing=0.10)

    entry_date = pd.Timestamp('2024-07-16')

    # 基金NAV
    fig.add_trace(go.Scatter(
        x=fund_df['Date'], y=fund_df['Open'],
        name='基金NAV', mode='lines',
        line=dict(color='#1f77b4', width=2.5),
        hovertemplate='%{x|%Y-%m-%d}<br>NAV: €%{y:.4f}<extra></extra>',
    ), row=1, col=1)
    # 买入点
    entry_nav = FUND_ENTRY_PRICE
    fig.add_trace(go.Scatter(
        x=[entry_date], y=[entry_nav],
        mode='markers+text', name='你的买入点',
        marker=dict(size=12, color='#e31a1c', symbol='star'),
        text=[f'  买入 €{entry_nav:.2f}<br>  2024-07-16'],
        textposition='middle right',
        textfont=dict(size=11, color='#e31a1c'),
    ), row=1, col=1)
    # 当前NAV水平线
    fig.add_hline(y=FUND_CURR_PRICE, line_dash='dot', line_color='#2ca02c', opacity=0.6,
                  annotation_text=f'当前 €{FUND_CURR_PRICE:.2f}',
                  annotation_position='top right',
                  annotation_font=dict(color='#2ca02c', size=10), row=1, col=1)
    fig.update_yaxes(title_text='NAV (€)', row=1, col=1)

    # PSI20
    fig.add_trace(go.Scatter(
        x=psi_df['Date'], y=psi_df['psi'],
        name='PSI20', mode='lines',
        line=dict(color='#ff7f0e', width=2),
        hovertemplate='%{x|%Y-%m-%d}<br>PSI20: %{y:.0f}<extra></extra>',
    ), row=2, col=1)
    # 买入日PSI20
    psi_entry_row = psi_df[psi_df['Date'] >= entry_date].iloc[0]
    fig.add_trace(go.Scatter(
        x=[psi_entry_row['Date']], y=[psi_entry_row['psi']],
        mode='markers+text', name='买入日PSI20',
        marker=dict(size=12, color='#e31a1c', symbol='star'),
        text=[f'  实际 {psi_entry_row["psi"]:.0f}点\n  (你记忆约7,400)'],
        textposition='middle right',
        textfont=dict(size=11, color='#e31a1c'),
    ), row=2, col=1)
    # PSI20=8000目标线
    fig.add_hline(y=PSI20_TARGET, line_dash='dash', line_color='#d62728', opacity=0.7,
                  annotation_text='担忧线: PSI20=8,000',
                  annotation_position='bottom right',
                  annotation_font=dict(color='#d62728', size=11), row=2, col=1)
    fig.add_hline(y=PSI20_CURRENT, line_dash='dot', line_color='gray', opacity=0.4,
                  annotation_text=f'当前 {PSI20_CURRENT:.0f}',
                  annotation_position='top right',
                  annotation_font=dict(size=10, color='gray'), row=2, col=1)
    fig.update_yaxes(title_text='PSI20点位', row=2, col=1)

    fig.update_layout(template='plotly_white', height=500, showlegend=True,
                      legend=dict(x=0.01, y=0.99),
                      margin=dict(t=50, b=20, l=70, r=30),
                      hovermode='x unified')
    return fig.to_json()


# ─── 图2：到期日损益图（以PSI20为轴）────────────────────────────
def chart_payoff(options):
    psi_range  = np.linspace(5500, 11000, 500)
    ibex_range = IBEX_CURRENT * (1 + BETA_IBEX_PSI20 * (psi_range - PSI20_CURRENT) / PSI20_CURRENT)
    fund_pnl   = FUND_VALUE * BETA_FUND_PSI20 * (psi_range - PSI20_CURRENT) / PSI20_CURRENT

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=psi_range, y=fund_pnl, name='无对冲', mode='lines',
        line=dict(color='#d62728', width=3),
        hovertemplate='PSI20=%{x:.0f}<br>基金损益: €%{y:,.0f}<extra>无对冲</extra>',
    ))

    show = {('Dec 2027','ATM 100%'), ('Dec 2027','95%'), ('Jun 2027','95%'), ('Dec 2027','90%')}
    clr  = {'Dec 2027 ATM 100%':'#1a9641', 'Dec 2027 95%':'#2ca02c',
             'Jun 2027 95%':'#1f77b4',     'Dec 2027 90%':'#78c679'}
    for s in options:
        key = f"{s['exp']} {s['k_label']}"
        if (s['exp'], s['k_label']) not in show: continue
        put_pnl = np.maximum(s['K'] - ibex_range, 0) * s['n'] - s['total_prem']
        fig.add_trace(go.Scatter(
            x=psi_range, y=fund_pnl + put_pnl,
            name=f"{s['exp']} Put {s['k_label']} (€{s['total_prem']:,.0f}权利金)",
            mode='lines', line=dict(color=clr.get(key,'#999'), width=2),
            hovertemplate=f"PSI20=%{{x:.0f}}<br>净损益: €%{{y:,.0f}}<extra>{key}</extra>",
        ))

    fig.add_vline(x=PSI20_TARGET, line_dash='dash', line_color='red', opacity=0.7,
                  annotation_text='PSI20=8,000（担忧线）',
                  annotation_position='top right',
                  annotation_font=dict(color='red', size=11))
    fig.add_vline(x=PSI20_7000, line_dash='dash', line_color='#7b0000', opacity=0.7,
                  annotation_text='PSI20=7,000（极端情形）',
                  annotation_position='bottom right',
                  annotation_font=dict(color='#7b0000', size=11))
    fig.add_vline(x=PSI20_CURRENT, line_dash='dot', line_color='gray', opacity=0.4,
                  annotation_text=f'当前{PSI20_CURRENT}',
                  annotation_position='top left',
                  annotation_font=dict(size=10, color='gray'))
    fig.add_vline(x=PSI20_ENTRY_ACT, line_dash='dot', line_color='#e31a1c', opacity=0.4,
                  annotation_text=f'买入日PSI20={PSI20_ENTRY_ACT}',
                  annotation_position='bottom left',
                  annotation_font=dict(size=10, color='#e31a1c'))
    fig.add_hline(y=0, line_dash='dot', line_color='gray', opacity=0.3)

    fig.update_layout(
        template='plotly_white', height=420, showlegend=True,
        xaxis_title='PSI20点位（到期时预估）',
        yaxis_title='相对今天的损益（€）',
        legend=dict(x=0.01, y=0.01, bgcolor='rgba(255,255,255,0.9)',
                    bordercolor='lightgray', borderwidth=1),
        margin=dict(t=20, b=50, l=70, r=30),
        hovermode='x unified',
    )
    return fig.to_json()


# ─── 图3：基金 vs PSI20 归一化（买入后） ─────────────────────────
def chart_correlation(fund_df, psi_df):
    entry_date = pd.Timestamp('2024-07-16')
    fd = fund_df[fund_df['Date'] >= entry_date].reset_index(drop=True)
    pd2 = psi_df[psi_df['Date'] >= entry_date].reset_index(drop=True)

    merged = fd.merge(pd2, on='Date', how='left').dropna()
    ref_f  = merged['Open'].iloc[0]
    ref_p  = merged['psi'].iloc[0]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=merged['Date'], y=merged['Open']/ref_f*100,
        name=f'基金（买入=100）', mode='lines',
        line=dict(color='#1f77b4', width=2.5),
        hovertemplate='%{x|%Y-%m-%d}<br>基金: %{y:.1f}<extra></extra>',
    ))
    fig.add_trace(go.Scatter(
        x=merged['Date'], y=merged['psi']/ref_p*100,
        name='PSI20（买入=100）', mode='lines',
        line=dict(color='#ff7f0e', width=2, dash='dot'),
        hovertemplate='%{x|%Y-%m-%d}<br>PSI20: %{y:.1f}<extra></extra>',
    ))
    psi_8000_norm = PSI20_TARGET / ref_p * 100
    fig.add_hline(y=psi_8000_norm, line_dash='dash', line_color='red', opacity=0.7,
                  annotation_text=f'PSI20=8000对应归一化值（{psi_8000_norm:.1f}）',
                  annotation_position='top right',
                  annotation_font=dict(color='red', size=10))
    fig.add_hline(y=100, line_dash='dot', line_color='gray', opacity=0.3)

    fig.update_layout(
        template='plotly_white', height=320, showlegend=True,
        xaxis_title='日期', yaxis_title='指数（买入日=100）',
        legend=dict(x=0.01, y=0.99),
        margin=dict(t=20, b=40, l=60, r=30),
        hovermode='x unified',
    )
    return fig.to_json()


# ─── 生成HTML ─────────────────────────────────────────────────
def generate_html(fund_df, ibex_df, psi_df, options):
    c1 = chart_full_history(fund_df, psi_df)
    c2 = chart_payoff(options)
    c3 = chart_correlation(fund_df, psi_df)

    fund_gain = (FUND_CURR_PRICE / FUND_ENTRY_PRICE - 1) * 100
    psi_gain_act  = (PSI20_CURRENT / PSI20_ENTRY_ACT  - 1) * 100
    psi_gain_user = (PSI20_CURRENT / PSI20_ENTRY_USER - 1) * 100

    # 推荐方案：Dec 2027 Put 95%
    rec = next(s for s in options if s['exp']=='Dec 2027' and s['k_label']=='95%')

    def cov_color(pct):
        if pct >= 60: return '#2ca02c'
        if pct >= 30: return '#e65100'
        return '#d62728'

    trows = ''
    last_inst = None
    for s in options:
        is_rec = s['inst']=='IBEX35' and s['exp']=='Dec 2027' and s['k_label']=='95%'
        hi = ' style="background:#e8f5e9;font-weight:600"' if is_rec else ''
        star = ' ★' if is_rec else ''
        c = cov_color(s['coverage'])
        inst_cell = f'<td rowspan="6" style="font-weight:700;background:#f0f4ff;vertical-align:middle">{s["inst"]}</td>' if s['inst'] != last_inst else ''
        last_inst = s['inst']
        trows += f"""<tr{hi}>
          {inst_cell}<td>{s['exp']}{star}</td>
          <td>{s['k_label']} ({s['K']:,}点)</td>
          <td>{s['n']}张</td>
          <td>€{s['price']:,.1f}</td>
          <td><b>€{s['total_prem']:,.0f}</b></td>
          <td><b style="color:#d62728">{s['annual_drag']:.2f}%</b></td>
          <td style="color:{c};font-weight:600">{s['coverage']:.0f}% / €{s['payoff_8000']:,.0f}</td>
          <td>€{s['net']:,.0f}</td>
        </tr>"""

    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>IBEX35 Put对冲方案 | Optimize Portugal Golden Opportunities</title>
<script src="https://cdn.plot.ly/plotly-3.4.0.min.js" crossorigin="anonymous"></script>
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;
     background:#f4f6fb;color:#1a1a2e;font-size:15px;line-height:1.65}}
.page{{max-width:1120px;margin:0 auto;padding:32px 20px 70px}}
h1{{font-size:26px;font-weight:800;color:#1a237e;margin-bottom:6px}}
.meta{{color:#777;font-size:13px;margin-bottom:32px}}
h2{{font-size:17px;font-weight:700;color:#1a237e;margin-bottom:16px;
    padding:6px 12px;border-left:4px solid #1a237e;background:#eef2ff;border-radius:0 6px 6px 0}}
.section{{margin-bottom:40px}}
.cards4{{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin-bottom:18px}}
.cards3{{display:grid;grid-template-columns:repeat(3,1fr);gap:14px;margin-bottom:18px}}
.card{{background:white;border-radius:12px;padding:18px 16px;
       box-shadow:0 2px 10px rgba(0,0,0,0.07)}}
.card .lbl{{font-size:11px;color:#999;text-transform:uppercase;letter-spacing:.6px;margin-bottom:4px}}
.card .val{{font-size:25px;font-weight:800;margin-bottom:3px}}
.card .sub{{font-size:12px;color:#bbb}}
.green .val{{color:#2e7d32}}.red .val{{color:#c62828}}
.blue .val{{color:#1565c0}}.orange .val{{color:#e65100}}
.purple .val{{color:#6a1b9a}}
.alert{{border-radius:10px;padding:15px 18px;margin-bottom:16px;font-size:14px}}
.a-warn{{background:#fff8e1;border-left:5px solid #f9a825}}
.a-info{{background:#e3f2fd;border-left:5px solid #1565c0}}
.a-good{{background:#e8f5e9;border-left:5px solid #388e3c}}
.a-key {{background:#fce4ec;border-left:5px solid #c62828}}
.a-note{{background:#f3e5f5;border-left:5px solid #7b1fa2}}
.chart-box{{background:white;border-radius:12px;padding:18px;
            box-shadow:0 2px 10px rgba(0,0,0,0.07);margin-bottom:18px}}
table{{width:100%;border-collapse:collapse;background:white;
       border-radius:12px;overflow:hidden;
       box-shadow:0 2px 10px rgba(0,0,0,0.07);font-size:14px}}
th{{background:#1a237e;color:white;padding:11px 10px;text-align:center;font-size:12px;font-weight:600}}
td{{padding:10px;text-align:center;border-bottom:1px solid #eee}}
tr:last-child td{{border:none}}
tr:hover td{{background:#f5f5ff}}
.rec{{background:#e8f5e9;border:2px solid #388e3c;border-radius:12px;
      padding:22px 26px;margin-bottom:18px}}
.rec h3{{color:#1b5e20;font-size:17px;margin-bottom:14px}}
.rec-grid{{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}}
.rec-item{{background:white;border-radius:8px;padding:12px 14px}}
.rec-item .rl{{font-size:11px;color:#888;margin-bottom:3px}}
.rec-item .rv{{font-size:21px;font-weight:800;color:#1b5e20}}
.steps ol{{padding-left:22px}}.steps li{{margin-bottom:12px;font-size:14px}}
.steps li b{{color:#1a237e}}
.two-col{{display:grid;grid-template-columns:1fr 1fr;gap:16px}}
.info-card{{background:white;border-radius:10px;padding:16px 18px;
            box-shadow:0 2px 8px rgba(0,0,0,0.06)}}
.info-card h4{{font-size:13px;color:#888;margin-bottom:8px;font-weight:600}}
.info-card p{{font-size:14px;line-height:1.6}}
.tag{{display:inline-block;padding:2px 8px;border-radius:4px;font-size:11px;font-weight:700}}
.tg{{background:#e8f5e9;color:#1b5e20}}.tr{{background:#fdecea;color:#b71c1c}}
.to{{background:#fff3e0;color:#e65100}}
.note-sm{{font-size:12px;color:#aaa;margin-top:8px}}
@media(max-width:700px){{.cards4{{grid-template-columns:1fr 1fr}}.cards3{{grid-template-columns:1fr}}
  .rec-grid{{grid-template-columns:1fr 1fr}}.two-col{{grid-template-columns:1fr}}}}
</style>
</head>
<body>
<div class="page">

<!-- HEADER -->
<h1>IBEX35 Put期权对冲方案</h1>
<p class="meta">
  基金：Optimize Portugal Golden Opportunities Fund（ISIN: PTOPZWHM0007）&nbsp;·&nbsp;
  Yahoo Finance: 0P0001O8MU.F&nbsp;·&nbsp;
  持仓：€651,000 / 37,024份额&nbsp;·&nbsp;
  买入日：2024年7月16日&nbsp;·&nbsp;数据截至2026年3月
</p>

<!-- ── 一、现状 ──────────────────────────────────── -->
<div class="section">
<h2>一、你的现状</h2>
<div class="cards4">
  <div class="card green">
    <div class="lbl">初始投资</div>
    <div class="val">€{INITIAL_INV:,}</div>
    <div class="sub">{FUND_UNITS:,}份 × €{FUND_ENTRY_PRICE:.4f}</div>
  </div>
  <div class="card green">
    <div class="lbl">当前持仓市值</div>
    <div class="val">€{FUND_VALUE:,}</div>
    <div class="sub">推算NAV €{FUND_CURR_PRICE:.2f}</div>
  </div>
  <div class="card purple">
    <div class="lbl">浮盈</div>
    <div class="val">+€{FUND_VALUE-INITIAL_INV:,}</div>
    <div class="sub">基金涨幅 +{fund_gain:.1f}%</div>
  </div>
  <div class="card orange">
    <div class="lbl">你想保护的底线</div>
    <div class="val">PSI20 8,000</div>
    <div class="sub">当前{PSI20_CURRENT:,}，距此−{abs(PSI20_DROP_PCT*100):.1f}%</div>
  </div>
</div>

<div class="alert a-warn">
  <b>⚠ 关于PSI20买入点的修正：</b>
  你记忆中买入时PSI20约7,400点，但实测数据显示2024年7月16日PSI20为
  <b>{PSI20_ENTRY_ACT:,}点</b>（偏差约10%）。
  这意味着PSI20实际涨幅比你认为的更大（+{psi_gain_act:.1f}%，而非你以为的+{psi_gain_user:.1f}%）。
  核心结论不受影响：即使PSI20跌到8,000，你仍远高于买入时的水平。
</div>

<div class="chart-box">
  <div id="c1" style="height:500px"></div>
</div>
</div>

<!-- ── 二、关键发现 ──────────────────────────────── -->
<div class="section">
<h2>二、为什么买IBEX35 Put？逻辑链：基金→PSI20→IBEX35</h2>

<!-- 逻辑链说明 -->
<div class="alert a-info" style="font-size:14px;line-height:1.9;margin-bottom:16px">
  <b>对冲逻辑链（三步）：</b><br>
  <b>① 基金跟PSI20走：</b> 这正是你当初买它的原因。Beta={BETA_FUND_PSI20:.2f}，相关性={CORR_FUND_PSI20:.2f}（R²={CORR_FUND_PSI20**2*100:.0f}%）——PSI20跌10%，基金历史上平均跌约{abs(BETA_FUND_PSI20*10):.1f}%。<br>
  <b>② PSI20无法直接做空：</b> Euronext Lisbon期权流动性极差，买卖价差5-10%，普通投资者无法以合理价格成交。<br>
  <b>③ IBEX35是最佳替代：</b> 西班牙与葡萄牙经济高度联动，IBEX35是IBKR可交易品种中与基金相关性最高的（R²={CORR_FUND_IBEX**2*100:.0f}%）。做空IBEX35≈间接保护了持有PSI20相关资产的敞口。
</div>

<!-- 相关性数据 -->
<div style="display:grid;grid-template-columns:1fr 1fr;gap:14px;margin-bottom:16px">
  <div class="info-card">
    <h4>基金与各指数的实测关联（买入后至今）</h4>
    <p style="line-height:2">
      基金 vs <b>PSI20</b>：Beta=<b>{BETA_FUND_PSI20:.2f}</b>，相关性=<b>{CORR_FUND_PSI20:.2f}</b>，R²=<b>{CORR_FUND_PSI20**2*100:.0f}%</b> <span class="tag tg">最理想</span><br>
      基金 vs <b>IBEX35</b>：Beta=<b>{BETA_FUND_IBEX:.2f}</b>，相关性=<b>{CORR_FUND_IBEX:.2f}</b>，R²=<b>{CORR_FUND_IBEX**2*100:.0f}%</b> <span class="tag tg">可交易最优</span><br>
      基金 vs <b>ESTX50</b>：Beta=<b>{BETA_FUND_ESTX:.2f}</b>，相关性=<b>{CORR_FUND_ESTX:.2f}</b>，R²=<b>{CORR_FUND_ESTX**2*100:.0f}%</b> <span class="tag to">备选</span><br>
      <br>
      <span style="font-size:12px;color:#777">R²=对冲效率：IBEX35只能解释基金{CORR_FUND_IBEX**2*100:.0f}%的波动，
      剩余{100-CORR_FUND_IBEX**2*100:.0f}%是基金独有风险（债券部分、个股基本面等）。
      对冲是不完美的，但它覆盖了最大的那块系统性风险。</span>
    </p>
  </div>
  <div class="info-card">
    <h4>IBEX35 vs PSI20 的联动</h4>
    <p style="line-height:1.9">
      IBEX35 vs PSI20：Beta=<b>{BETA_IBEX_PSI20:.2f}</b>，相关性≈<b>{CORR_IBEX_PSI20:.2f}</b><br>
      即：PSI20跌10%，IBEX35历史上平均跌约<b>{abs(BETA_IBEX_PSI20*10):.1f}%</b><br><br>
      这是put期权保护逻辑的关键：
      你担心PSI20下跌 → PSI20下跌时IBEX35几乎同步下跌 → IBEX35 put期权的内在价值上升 → 赔付补偿基金损失。
      <br><br>
      <span style="font-size:12px;color:#777">注：两者不完全同步，存在"基差风险"。但在大级别下跌（≥10%）时，历史上联动相当稳定。</span>
    </p>
  </div>
</div>

<!-- 双场景测算 -->
<div style="display:grid;grid-template-columns:1fr 1fr;gap:14px;margin-bottom:16px">
  <div class="alert a-warn" style="margin:0">
    <b>场景A：PSI20跌到8,000（你的担忧线）</b><br>
    PSI20下跌 <b>{abs(PSI20_DROP_PCT*100):.1f}%</b>（{PSI20_CURRENT:,}→8,000）<br>
    → 基金预计跌 <b>{abs(FUND_DROP_PCT*100):.1f}%</b>，损失约 <b>€{abs(FUND_LOSS_EUR):,.0f}</b><br>
    → 基金NAV约 €{FUND_CURR_PRICE*(1+FUND_DROP_PCT):.2f}（仍高于买入价€{FUND_ENTRY_PRICE:.2f}）<br>
    → IBEX35对应跌至约 <b>{IBEX_AT_8000:,.0f}点</b>（跌{abs(BETA_IBEX_PSI20*PSI20_DROP_PCT*100):.1f}%）
  </div>
  <div class="alert a-key" style="margin:0">
    <b>场景B：PSI20跌到7,000（极端情形）</b><br>
    PSI20下跌 <b>{abs(PSI20_DROP_7000*100):.1f}%</b>（{PSI20_CURRENT:,}→7,000）<br>
    → 基金预计跌 <b>{abs(FUND_DROP_7000*100):.1f}%</b>，损失约 <b>€{abs(FUND_LOSS_7000):,.0f}</b><br>
    → 基金NAV约 €{FUND_CURR_PRICE*(1+FUND_DROP_7000):.2f}（仍高于买入价€{FUND_ENTRY_PRICE:.2f}）<br>
    → IBEX35对应跌至约 <b>{IBEX_AT_7000:,.0f}点</b>（跌{abs(BETA_IBEX_PSI20*PSI20_DROP_7000*100):.1f}%）
  </div>
</div>

<div class="chart-box">
  <div id="c3" style="height:320px"></div>
</div>
<p class="note-sm" style="padding:0 4px">买入后基金与PSI20归一化走势：两者方向一致，体现了高相关性（R²={CORR_FUND_PSI20**2*100:.0f}%）。</p>
</div>

<!-- ── 三、对冲工具 ──────────────────────────────── -->
<div class="section">
<h2>三、IBKR可交易的对冲工具对比（IBEX35 vs ESTX50）</h2>

<div class="alert a-info" style="margin-bottom:14px">
  <b>PSI20可以替代吗？</b>
  PSI20是最理想的对冲标的（R²=<b>{CORR_FUND_PSI20**2*100:.0f}%</b>），但PSI20期权在Euronext Lisbon流动性<b>极差</b>，
  买卖价差5-10%，普通投资者无法以合理价格成交。所以必须用相关替代品。
</div>

<div style="display:grid;grid-template-columns:repeat(3,1fr);gap:14px;margin-bottom:18px">
  <div class="info-card">
    <h4><span class="tag tr">理想但不可行</span> PSI20 Put</h4>
    <p>
      与基金相关性最高：R²=<b>{CORR_FUND_PSI20**2*100:.0f}%</b><br>
      Beta={BETA_FUND_PSI20:.2f}<br><br>
      但流动性极差，Euronext Lisbon，
      买卖价差巨大，无法高效对冲。
    </p>
  </div>
  <div class="info-card">
    <h4><span class="tag tg">★ 推荐</span> IBEX35 Put（MEFF）</h4>
    <p>
      与基金相关性：R²=<b>{CORR_FUND_IBEX**2*100:.0f}%</b>，Beta={BETA_FUND_IBEX:.2f}<br>
      IBEX35 vs PSI20 相关性≈{CORR_IBEX_PSI20:.2f}<br><br>
      西班牙MEFF交易所，<b>流动性好</b>，IBKR可直接下单。
      合约乘数1€/点，{N_CONTRACTS}张覆盖基金IBEX敞口。
      可买到2027年12月远期，无需滚仓。
    </p>
  </div>
  <div class="info-card">
    <h4><span class="tag to">备选</span> Euro Stoxx 50 Put（Eurex）</h4>
    <p>
      与基金相关性：R²=<b>{CORR_FUND_ESTX**2*100:.0f}%</b>，Beta={BETA_FUND_ESTX:.2f}<br>
      ESTX50 vs PSI20 Beta={BETA_ESTX_PSI20:.2f}<br><br>
      Eurex交易所，流动性极强，合约乘数10€/点。
      但需要{N_CONTRACTS_ESTX}张合约，总权利金更高，
      且与基金相关性低于IBEX35。
    </p>
  </div>
</div>

<div class="alert a-warn">
  <b>为什么IBEX35看起来和基金走势一致，但R²只有{CORR_FUND_IBEX**2*100:.0f}%？</b><br>
  视觉上趋势相似（都受欧洲市场驱动），但日度收益率的"步调"差异很大。
  R²={CORR_FUND_IBEX**2*100:.0f}%意味着IBEX35的涨跌只能解释基金波动的{CORR_FUND_IBEX**2*100:.0f}%，
  剩余{100-CORR_FUND_IBEX**2*100:.0f}%由基金独有因素（葡萄牙企业基本面、债券部分等）决定。
  对冲是不完美的，但它覆盖了系统性风险中最大的那一块。
</div>
</div>

<!-- ── 四、推荐方案 ──────────────────────────────── -->
<div class="section">
<h2>四、具体方案</h2>

<div class="alert a-info" style="margin-bottom:16px">
  <b>合约数量逻辑：</b>
  基金IBEX35 Beta敞口 = €651,000 × {BETA_FUND_IBEX:.3f} = €{int(FUND_VALUE*BETA_FUND_IBEX):,}；
  IBEX35一张合约价值 = {IBEX_CURRENT:,}点 × 1€/点 = €{IBEX_CURRENT:,}；
  合约数 = {int(FUND_VALUE*BETA_FUND_IBEX):,} ÷ {IBEX_CURRENT:,} ≈ <b>{N_CONTRACTS}张</b>。
  这是最小标准对冲单位。
</div>

<div class="rec">
  <h3>★ 推荐方案：Dec 2027 Put 95%（行权价 {next(s for s in options if s["exp"]=="Dec 2027" and s["k_label"]=="95%")["K"]:,}点）</h3>
  <div class="rec-grid">
    <div class="rec-item"><div class="rl">买入数量</div><div class="rv">{rec['n']} 张</div></div>
    <div class="rec-item"><div class="rl">单张权利金（估算）</div><div class="rv">€{rec['price']:,.0f}</div></div>
    <div class="rec-item"><div class="rl">一次性总成本</div><div class="rv">€{rec['total_prem']:,.0f}</div></div>
    <div class="rec-item"><div class="rl">年化损耗（占持仓）</div><div class="rv">{rec['annual_drag']:.2f}% / 年</div></div>
    <div class="rec-item"><div class="rl">PSI20=8000时赔付</div><div class="rv">€{rec['payoff_8000']:,}</div></div>
    <div class="rec-item"><div class="rl">覆盖基金损失比例</div>
      <div class="rv" style="color:{cov_color(rec['coverage'])}">{rec['coverage']:.0f}%</div></div>
  </div>
  <p style="margin-top:14px;font-size:13px;color:#444;line-height:1.8">
    <b>这{rec['n']}张put如何保护你？</b><br>
    • <b>场景A（PSI20=8,000）：</b> IBEX35跌至约{IBEX_AT_8000:,.0f}点，put行权价{rec['K']:,}点，
      赔付=({rec['K']:,}−{IBEX_AT_8000:,.0f})×{rec['n']}张 = <b>€{rec['payoff_8000']:,}</b>。
      基金损失约€{abs(FUND_LOSS_EUR):,.0f}，put覆盖了其中<b>{rec['coverage']:.0f}%</b>，对冲后净损益约€{rec['net']:,}。<br>
    • <b>场景B（PSI20=7,000极端）：</b> IBEX35跌至约{IBEX_AT_7000:,.0f}点，
      赔付=({rec['K']:,}−{IBEX_AT_7000:,.0f})×{rec['n']}张 = <b>€{max(rec['K']-IBEX_AT_7000,0)*rec['n']:,.0f}</b>。
      基金损失约€{abs(FUND_LOSS_7000):,.0f}，put覆盖了其中<b>{min(max(rec['K']-IBEX_AT_7000,0)*rec['n']/abs(FUND_LOSS_7000)*100,100):.0f}%</b>。<br>
    • <b>为什么选95%行权价：</b> 在更深度的下跌中才真正发力（比ATM便宜约40%权利金），
      同时对中等跌幅（PSI20 -10%至-20%）也有部分覆盖。<br>
    • <b>年化损耗仅{rec['annual_drag']:.2f}%</b>（€{rec['total_prem']:,}÷{rec['T']*12:.0f}个月），
      相当于每年为€{FUND_VALUE:,}持仓支付约€{rec['total_prem']/rec['T']:,.0f}保险费，远低于基金1.8%管理费。
  </p>
</div>

<table>
  <tr>
    <th>标的</th><th>到期日</th><th>行权价</th><th>合约数</th>
    <th>单张权利金</th><th>总成本（一次）</th><th>年化损耗</th>
    <th>PSI20=8000时赔付/覆盖率</th><th>对冲后净损益</th>
  </tr>
  {trows}
</table>
<p class="note-sm">
  权利金基于Black-Scholes（隐含波动率18.5%，无风险利率2.6%），
  实际市场隐含波动率和买卖价差可能使实际成本上浮10-20%。
  PSI20=8000赔付按IBEX35对应跌至{IBEX_AT_8000:,.0f}点估算；行权价高于IBEX35到期价时put生效。
</p>
</div>

<!-- ── 五、损益图 ──────────────────────────────────── -->
<div class="section">
<h2>五、到2027年12月到期时——各PSI20点位下你的组合净损益</h2>
<div class="alert a-info" style="margin-bottom:12px;font-size:13px">
  横轴是PSI20到期时的点位，纵轴是相对今天（€{FUND_VALUE:,}）的损益。
  IBEX35对应点位由历史Beta（IBEX vs PSI20 = {BETA_IBEX_PSI20:.2f}）联动推算。
  仅展示IBEX35 Put方案；对冲不完美，实际损益会因两指数背离而有偏差。
</div>
<div class="chart-box">
  <div id="c2" style="height:420px"></div>
</div>
</div>

<!-- ── 六、操作指引 ──────────────────────────────── -->
<div class="section">
<h2>六、如何操作（一步一步）</h2>
<div class="steps">
<ol>
  <li>
    <b>开通欧洲期权交易权限：</b>
    推荐通过 <b>Interactive Brokers（盈透证券）</b> 或 <b>Saxo Bank</b>，
    均可直接接入MEFF（西班牙期权交易所，BME旗下）。
    开户时申请"欧洲期权交易"权限。
  </li>
  <li>
    <b>查找合约：</b>
    搜索标的 <b>IBEX 35 Index Options（代码：OI）</b>，
    到期月选 <b>2027年12月</b>，类型选 <b>Put</b>，
    行权价约 <b>{rec['K']:,}点</b>（95% ATM附近）。
  </li>
  <li>
    <b>挂限价单买入：</b>
    买入 <b>{N_CONTRACTS} 张</b>，参考价 €{rec['price']:,.0f}/张，
    用 <b>Limit Order</b>（限价单），切勿用市价单（价差损失大）。
    总预算约 <b>€{rec['total_prem']:,.0f}</b>（实际可能上浮10-20%）。
  </li>
  <li>
    <b>持有到2027年12月，无需任何操作：</b>
    若到期时IBEX35低于{rec['K']:,}点，期权自动行权赔付差价。
    若IBEX35仍高于行权价，权利金是你的对冲成本（已花完），基金本身仍盈利。
  </li>
  <li>
    <b>每年12月复盘一次：</b>
    若持仓规模大幅变化，可按新规模重新计算合约数。
  </li>
</ol>
</div>
</div>

<!-- ── 七、结论 ──────────────────────────────────── -->
<div class="section">
<h2>七、一句话结论</h2>
<div class="alert a-note" style="font-size:15px;line-height:1.9">
  你买这只基金是因为它高度跟踪PSI20（R²={CORR_FUND_PSI20**2*100:.0f}%）。PSI20没有可操作的期权，
  所以用与PSI20高度联动的<b>IBEX35作为替代对冲工具</b>（R²={CORR_FUND_IBEX**2*100:.0f}%，IBKR可交易中最优）。<br><br>
  推荐方案：<b>{N_CONTRACTS}张 IBEX35 Dec 2027 Put，行权价{rec['K']:,}点</b>，一次性支出约<b>€{rec['total_prem']:,.0f}</b>，年化损耗{rec['annual_drag']:.2f}%。<br><br>
  保护效果：<br>
  · PSI20=8,000（跌{abs(PSI20_DROP_PCT*100):.0f}%）→ 基金损失€{abs(FUND_LOSS_EUR):,.0f} → put赔付€{rec['payoff_8000']:,} → <b>覆盖{rec['coverage']:.0f}%损失</b><br>
  · PSI20=7,000（跌{abs(PSI20_DROP_7000*100):.0f}%）→ 基金损失€{abs(FUND_LOSS_7000):,.0f} → put赔付€{max(rec['K']-IBEX_AT_7000,0)*rec['n']:,.0f} → <b>覆盖{min(max(rec['K']-IBEX_AT_7000,0)*rec['n']/abs(FUND_LOSS_7000)*100,100):.0f}%损失</b><br><br>
  即使PSI20跌到7,000，你的基金NAV约€{FUND_CURR_PRICE*(1+FUND_DROP_7000):.2f}，仍高于你的买入成本€{FUND_ENTRY_PRICE:.2f}。
  这笔put的价值在于：以极低成本（€{rec['total_prem']:,}）为极端风险购买了安心。
</div>
</div>

</div><!-- /page -->
<script>
  Plotly.newPlot('c1',__C1__.data,__C1__.layout,{{responsive:true}});
  Plotly.newPlot('c2',__C2__.data,__C2__.layout,{{responsive:true}});
  Plotly.newPlot('c3',__C3__.data,__C3__.layout,{{responsive:true}});
</script>
</body></html>"""

    html = html.replace('__C1__', c1)
    html = html.replace('__C2__', c2)
    html = html.replace('__C3__', c3)
    return html


def main():
    print('加载数据...')
    fund_df, ibex_df, psi_df = load_data()
    print(f'数据加载完成：基金{len(fund_df)}条，PSI20 {len(psi_df)}条')
    print('计算期权方案...')
    options = calc_options()
    print('生成报告...')
    html = generate_html(fund_df, ibex_df, psi_df, options)
    out = os.path.join(DATA_DIR, 'hedge_report.html')
    with open(out, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f'报告已生成: {out}')
    os.startfile(out)


if __name__ == '__main__':
    main()
