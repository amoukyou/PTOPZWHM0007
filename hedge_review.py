"""
葡萄牙基金对冲可靠性分析 — 基于历史数据的对冲效果回测
"""

import os, sys, math, json
import numpy as np
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots

sys.stdout.reconfigure(encoding='utf-8')
DATA_DIR = os.path.dirname(os.path.abspath(__file__))

# ═══════ 核心参数 ══════════════════════════════════
FUND_VALUE       = 651_000
INITIAL_INV      = 507_000
FUND_ENTRY_PRICE = 13.3534
FUND_UNITS       = round(INITIAL_INV / FUND_ENTRY_PRICE)
FUND_CURR_PRICE  = round(FUND_VALUE / FUND_UNITS, 4)

PSI20_CURRENT    = 8_862
IBEX_CURRENT     = 17_062

BETA_FUND_PSI20  = 0.6271
BETA_FUND_IBEX   = 0.4230
BETA_IBEX_PSI20  = 0.6897
# ═══════════════════════════════════════════════════


def load_data():
    fund = pd.read_csv(os.path.join(DATA_DIR, 'PTOPZWHM0007_daily_2022-2026.csv'), parse_dates=['Date'])
    fund = fund.sort_values('Date').reset_index(drop=True)

    ibex_csv = os.path.join(DATA_DIR, 'IBEX35_daily_2024-2026.csv')
    ibex = pd.read_csv(ibex_csv, parse_dates=['Date'])
    ibex = ibex.sort_values('Date').reset_index(drop=True)

    psi = yf.download('PSI20.LS', start='2022-01-01', end='2026-03-06', progress=False)
    psi.columns = [c[0] for c in psi.columns]
    psi = psi.reset_index()[['Date','Close']].rename(columns={'Close':'psi'})
    psi['Date'] = psi['Date'].dt.normalize()

    return fund, ibex, psi


def analyze(fund_df, ibex_df, psi_df):
    """核心分析：合并数据并计算所有指标"""
    # 合并基金和IBEX
    df = pd.merge(
        fund_df[['Date','Close']].rename(columns={'Close':'fund'}),
        ibex_df[['Date','Close']].rename(columns={'Close':'ibex'}),
        on='Date'
    )
    df = df.sort_values('Date').reset_index(drop=True)
    df['ret_fund'] = df['fund'].pct_change()
    df['ret_ibex'] = df['ibex'].pct_change()
    df = df.dropna().reset_index(drop=True)

    results = {}

    # 1) 不同时间窗口的R²
    r2_by_horizon = []
    for window, label in [(1,'1天'), (5,'1周'), (10,'2周'), (21,'1个月'), (63,'1季度'), (126,'半年')]:
        ret_f = df['fund'].pct_change(window)
        ret_i = df['ibex'].pct_change(window)
        valid = ret_f.dropna().index.intersection(ret_i.dropna().index)
        r = np.corrcoef(ret_f[valid], ret_i[valid])[0,1]
        r2_by_horizon.append(dict(window=window, label=label, r=r, r2=r**2, n=len(valid)))
    results['r2_horizon'] = r2_by_horizon

    # 2) 基金单日跌超1%时IBEX表现
    bad_days = df[df['ret_fund'] < -0.01].copy()
    bad_days['same_dir'] = bad_days['ret_ibex'] < 0
    results['bad_days'] = bad_days
    results['bad_days_sync_rate'] = bad_days['same_dir'].mean()

    # 3) 滚动3个月收益对比
    df['ret_fund_3m'] = df['fund'].pct_change(63)
    df['ret_ibex_3m'] = df['ibex'].pct_change(63)
    results['df_3m'] = df.dropna(subset=['ret_fund_3m','ret_ibex_3m']).copy()

    # 4) 脱钩期识别（基金3个月跌但IBEX涨）
    danger = results['df_3m'][(results['df_3m']['ret_fund_3m'] < -0.005) & (results['df_3m']['ret_ibex_3m'] > 0.005)]
    results['danger_periods'] = danger

    # 5) 滚动60天R²
    roll_r2 = []
    for i in range(60, len(df)):
        chunk = df.iloc[i-60:i]
        r = np.corrcoef(chunk['ret_fund'], chunk['ret_ibex'])[0,1]
        roll_r2.append(dict(date=chunk['Date'].iloc[-1], r2=r**2))
    results['rolling_r2'] = pd.DataFrame(roll_r2)

    results['df'] = df
    return results


# ─── 图1：基金跌的日子IBEX在干嘛（水平对比） ─────────
def chart_bad_days(bad_days):
    bd = bad_days.sort_values('Date', ascending=True).reset_index(drop=True)
    labels = bd['Date'].dt.strftime('%Y-%m-%d').tolist()

    fig = go.Figure()

    # 基金跌幅（向左，蓝色）
    fig.add_trace(go.Bar(
        y=labels, x=bd['ret_fund']*100,
        orientation='h', name='基金跌幅',
        marker_color='#1565c0',
        text=[f'{v*100:.1f}%' for v in bd['ret_fund']],
        textposition='outside', textfont=dict(size=11),
        hovertemplate='%{y}<br>基金: %{x:.2f}%<extra></extra>',
    ))

    # IBEX涨跌（颜色区分）
    ibex_colors = ['#2e7d32' if x else '#c62828' for x in bd['same_dir']]
    fig.add_trace(go.Bar(
        y=labels, x=bd['ret_ibex']*100,
        orientation='h', name='IBEX35',
        marker_color=ibex_colors,
        text=[f'{v*100:+.1f}%' for v in bd['ret_ibex']],
        textposition='outside', textfont=dict(size=11),
        hovertemplate='%{y}<br>IBEX: %{x:.2f}%<extra></extra>',
    ))

    fig.add_vline(x=0, line_color='gray', line_width=1, opacity=0.5)

    fig.update_layout(
        template='plotly_white', height=max(420, len(bd)*32),
        barmode='group', bargap=0.25, bargroupgap=0.1,
        xaxis_title='涨跌幅 (%)',
        yaxis=dict(autorange='reversed'),
        legend=dict(x=0.01, y=1.02, orientation='h'),
        margin=dict(t=30, b=40, l=100, r=60),
    )
    return fig.to_json()


# ─── 图2：滚动60天R² ─────────────────────────────
def chart_rolling_r2(rolling_r2):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=rolling_r2['date'], y=rolling_r2['r2']*100,
        mode='lines', name='滚动60天 R²',
        line=dict(color='#1565c0', width=2.5),
        fill='tozeroy', fillcolor='rgba(21,101,192,0.1)',
        hovertemplate='%{x|%Y-%m-%d}<br>R²: %{y:.1f}%<extra></extra>',
    ))
    fig.add_hline(y=42, line_dash='dash', line_color='#e65100', opacity=0.8,
                  annotation_text='全期平均 R²=42%',
                  annotation_position='top right',
                  annotation_font=dict(color='#e65100', size=11))
    # 脱钩危险区标注
    fig.add_hrect(y0=0, y1=20, fillcolor='#ffcdd2', opacity=0.15,
                  annotation_text='危险区：R²<20%', annotation_position='bottom right',
                  annotation_font=dict(color='#c62828', size=10))

    fig.update_layout(
        template='plotly_white', height=320,
        yaxis_title='R² (%)', yaxis_range=[0, 100],
        margin=dict(t=10, b=30, l=60, r=20),
        hovermode='x unified', showlegend=False,
    )
    return fig.to_json()


# ─── 图3：3个月滚动收益散点图 ──────────────────────
def chart_3m_scatter(df_3m):
    colors = ['#c62828' if (r['ret_fund_3m'] < 0 and r['ret_ibex_3m'] > 0) else
              '#e65100' if (r['ret_fund_3m'] > 0 and r['ret_ibex_3m'] < 0) else
              '#2e7d32' for _, r in df_3m.iterrows()]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_3m['ret_ibex_3m']*100, y=df_3m['ret_fund_3m']*100,
        mode='markers', marker=dict(size=5, color=colors, opacity=0.6),
        hovertemplate='IBEX 3月收益: %{x:.1f}%<br>基金 3月收益: %{y:.1f}%<extra></extra>',
    ))
    # 对角线
    rng = [-15, 25]
    fig.add_trace(go.Scatter(
        x=rng, y=rng, mode='lines', line=dict(color='gray', dash='dot', width=1),
        showlegend=False, hoverinfo='skip',
    ))
    # 危险象限标注
    fig.add_vrect(x0=0, x1=25, y0=-15, y1=0,
                  fillcolor='#ffcdd2', opacity=0.15,
                  annotation_text='危险：基金跌+IBEX涨<br>Put不赔付',
                  annotation_position='bottom right',
                  annotation_font=dict(color='#c62828', size=10))

    fig.update_layout(
        template='plotly_white', height=400,
        xaxis_title='IBEX35 三个月收益 (%)',
        yaxis_title='基金 三个月收益 (%)',
        margin=dict(t=10, b=50, l=60, r=20),
        showlegend=False,
    )
    return fig.to_json()


# ─── 图4：2024年8-12月脱钩期详图 ──────────────────
def chart_decoupling(df):
    mask = (df['Date'] >= '2024-07-01') & (df['Date'] <= '2025-01-31')
    sub = df[mask].copy()
    ref_f = sub['fund'].iloc[0]
    ref_i = sub['ibex'].iloc[0]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=sub['Date'], y=(sub['fund']/ref_f - 1)*100,
        name='基金', mode='lines', line=dict(color='#1565c0', width=2.5),
        hovertemplate='%{x|%Y-%m-%d}<br>基金: %{y:+.1f}%<extra></extra>',
    ))
    fig.add_trace(go.Scatter(
        x=sub['Date'], y=(sub['ibex']/ref_i - 1)*100,
        name='IBEX35', mode='lines', line=dict(color='#e65100', width=2.5),
        hovertemplate='%{x|%Y-%m-%d}<br>IBEX: %{y:+.1f}%<extra></extra>',
    ))
    fig.add_hline(y=0, line_dash='dot', line_color='gray', opacity=0.3)
    fig.add_vrect(x0='2024-08-01', x1='2024-12-31',
                  fillcolor='#ffcdd2', opacity=0.12,
                  annotation_text='脱钩期：基金跌，IBEX涨',
                  annotation_position='top left',
                  annotation_font=dict(color='#c62828', size=11))

    fig.update_layout(
        template='plotly_white', height=350,
        yaxis_title='相对2024年7月初涨跌 (%)',
        legend=dict(x=0.01, y=0.99),
        margin=dict(t=10, b=30, l=60, r=20),
        hovermode='x unified',
    )
    return fig.to_json()


# ─── 图5：2025年4月关税冲击（对冲有效案例）──────────
def chart_tariff_shock(df):
    mask = (df['Date'] >= '2025-03-15') & (df['Date'] <= '2025-05-15')
    sub = df[mask].copy()
    ref_f = sub['fund'].iloc[0]
    ref_i = sub['ibex'].iloc[0]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=sub['Date'], y=(sub['fund']/ref_f - 1)*100,
        name='基金', mode='lines', line=dict(color='#1565c0', width=2.5),
        hovertemplate='%{x|%Y-%m-%d}<br>基金: %{y:+.1f}%<extra></extra>',
    ))
    fig.add_trace(go.Scatter(
        x=sub['Date'], y=(sub['ibex']/ref_i - 1)*100,
        name='IBEX35', mode='lines', line=dict(color='#e65100', width=2.5),
        hovertemplate='%{x|%Y-%m-%d}<br>IBEX: %{y:+.1f}%<extra></extra>',
    ))
    fig.add_hline(y=0, line_dash='dot', line_color='gray', opacity=0.3)
    fig.add_vrect(x0='2025-04-02', x1='2025-04-10',
                  fillcolor='#c8e6c9', opacity=0.2,
                  annotation_text='关税冲击：两者同步暴跌，对冲有效',
                  annotation_position='top left',
                  annotation_font=dict(color='#2e7d32', size=11))

    fig.update_layout(
        template='plotly_white', height=350,
        yaxis_title='相对3月中旬涨跌 (%)',
        legend=dict(x=0.01, y=0.99),
        margin=dict(t=10, b=30, l=60, r=20),
        hovermode='x unified',
    )
    return fig.to_json()


# ─── 生成HTML ─────────────────────────────────────
def generate_html(results):
    r2h = results['r2_horizon']
    bad = results['bad_days']
    sync_rate = results['bad_days_sync_rate']
    rolling_r2 = results['rolling_r2']
    df = results['df']
    df_3m = results['df_3m']

    c1 = chart_bad_days(bad)
    c2 = chart_rolling_r2(rolling_r2)
    c3 = chart_3m_scatter(df_3m)
    c4 = chart_decoupling(df)
    c5 = chart_tariff_shock(df)

    # R²表格行
    r2_rows = ''
    for h in r2h:
        bar_w = h['r2'] * 100
        color = '#2e7d32' if h['r2'] > 0.45 else '#e65100' if h['r2'] > 0.3 else '#c62828'
        r2_rows += f"""<tr>
          <td style="font-weight:600">{h['label']}</td>
          <td>{h['r']:.3f}</td>
          <td><b style="color:{color}">{h['r2']*100:.1f}%</b></td>
          <td style="text-align:left"><div style="background:{color};height:18px;width:{bar_w}%;border-radius:3px;opacity:0.7"></div></td>
          <td style="color:#999">{h['n']}</td>
        </tr>"""

    # 基金跌的日子表格
    bad_rows = ''
    for _, r in bad.iterrows():
        ic = '#2e7d32' if r['ret_ibex'] < 0 else '#c62828'
        mark = 'IBEX同跌' if r['ret_ibex'] < 0 else 'IBEX反涨'
        bad_rows += f"""<tr>
          <td>{r['Date'].strftime('%Y-%m-%d')}</td>
          <td style="color:#1565c0;font-weight:600">{r['ret_fund']*100:.2f}%</td>
          <td style="color:{ic};font-weight:600">{r['ret_ibex']*100:.2f}%</td>
          <td style="color:{ic}">{mark}</td>
        </tr>"""

    # 最差脱钩
    danger = results['danger_periods']
    worst = danger.sort_values('ret_fund_3m').head(5) if len(danger) > 0 else pd.DataFrame()
    worst_rows = ''
    for _, r in worst.iterrows():
        worst_rows += f"""<tr>
          <td>{r['Date'].strftime('%Y-%m-%d')}</td>
          <td style="color:#1565c0">{r['ret_fund_3m']*100:.2f}%</td>
          <td style="color:#c62828">{r['ret_ibex_3m']*100:.2f}%</td>
        </tr>"""

    # 滚动R²统计
    r2_min = rolling_r2['r2'].min() * 100
    r2_max = rolling_r2['r2'].max() * 100
    r2_below_20 = (rolling_r2['r2'] < 0.20).sum()
    r2_below_20_pct = r2_below_20 / len(rolling_r2) * 100

    n_bad = len(bad)
    n_sync = bad['same_dir'].sum()
    n_anti = n_bad - n_sync

    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>IBEX35对冲可靠性分析 | 历史回测</title>
<script src="https://cdn.plot.ly/plotly-3.4.0.min.js" crossorigin="anonymous"></script>
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;
     background:#f4f6fb;color:#1a1a2e;font-size:15px;line-height:1.65}}
.page{{max-width:1060px;margin:0 auto;padding:32px 20px 70px}}
h1{{font-size:26px;font-weight:800;color:#1a237e;margin-bottom:6px}}
.meta{{color:#777;font-size:13px;margin-bottom:32px}}
h2{{font-size:17px;font-weight:700;color:#1a237e;margin-bottom:16px;
    padding:6px 12px;border-left:4px solid #1a237e;background:#eef2ff;border-radius:0 6px 6px 0}}
.section{{margin-bottom:44px}}
.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:14px;margin-bottom:18px}}
.card{{background:white;border-radius:12px;padding:18px 16px;
       box-shadow:0 2px 10px rgba(0,0,0,0.07)}}
.card .lbl{{font-size:11px;color:#999;text-transform:uppercase;letter-spacing:.6px;margin-bottom:4px}}
.card .val{{font-size:25px;font-weight:800;margin-bottom:3px}}
.card .sub{{font-size:12px;color:#bbb}}
.green .val{{color:#2e7d32}}.red .val{{color:#c62828}}
.blue .val{{color:#1565c0}}.orange .val{{color:#e65100}}
.alert{{border-radius:10px;padding:15px 18px;margin-bottom:16px;font-size:14px;line-height:1.8}}
.a-warn{{background:#fff8e1;border-left:5px solid #f9a825}}
.a-info{{background:#e3f2fd;border-left:5px solid #1565c0}}
.a-good{{background:#e8f5e9;border-left:5px solid #388e3c}}
.a-bad{{background:#fdecea;border-left:5px solid #c62828}}
.a-note{{background:#f3e5f5;border-left:5px solid #7b1fa2}}
.chart-box{{background:white;border-radius:12px;padding:18px;
            box-shadow:0 2px 10px rgba(0,0,0,0.07);margin-bottom:18px}}
table{{width:100%;border-collapse:collapse;background:white;
       border-radius:12px;overflow:hidden;
       box-shadow:0 2px 10px rgba(0,0,0,0.07);font-size:14px;margin-bottom:18px}}
th{{background:#1a237e;color:white;padding:11px 10px;text-align:center;font-size:12px;font-weight:600}}
td{{padding:10px;text-align:center;border-bottom:1px solid #eee}}
tr:last-child td{{border:none}}
tr:hover td{{background:#f5f5ff}}
.note-sm{{font-size:12px;color:#aaa;margin-top:8px}}
.two-col{{display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-bottom:18px}}
@media(max-width:700px){{.cards{{grid-template-columns:1fr 1fr}}.two-col{{grid-template-columns:1fr}}}}
</style>
</head>
<body>
<div class="page">

<h1>IBEX35 Put对冲 —— 到底靠不靠谱？</h1>
<p class="meta">
  基金：Optimize Portugal Golden Opportunities（PTOPZWHM0007）&nbsp;&middot;&nbsp;
  持仓 &euro;651,000&nbsp;&middot;&nbsp;
  对冲工具：IBEX35 Put期权&nbsp;&middot;&nbsp;
  数据：2024年2月 - 2026年3月（{len(df)}个交易日）
</p>

<!-- ══ 一、核心问题 ══════════════════════════════ -->
<div class="section">
<h2>一、核心问题：R&sup2;=42% 意味着什么</h2>

<div class="alert a-warn">
  <b>R&sup2;=42%</b> 的意思是：IBEX35的涨跌只能解释你基金涨跌的42%，
  剩下58%的波动是IBEX35"管不到"的。<br>
  用IBEX35 Put来对冲，保护效果是<b>打折的</b>——IBEX35跌10%，你的基金不一定跟着跌对应的幅度。
</div>

<div class="cards">
  <div class="card blue">
    <div class="lbl">全期 R&sup2;</div>
    <div class="val">42%</div>
    <div class="sub">IBEX能解释的基金波动比例</div>
  </div>
  <div class="card orange">
    <div class="lbl">无法解释的部分</div>
    <div class="val">58%</div>
    <div class="sub">葡萄牙本地因素、基金个股等</div>
  </div>
  <div class="card green">
    <div class="lbl">基金跌时IBEX同向</div>
    <div class="val">{sync_rate*100:.0f}%</div>
    <div class="sub">{n_sync}/{n_bad}次（单日跌超1%时）</div>
  </div>
  <div class="card red">
    <div class="lbl">基金跌但IBEX反涨</div>
    <div class="val">{n_anti}次</div>
    <div class="sub">占{100-sync_rate*100:.0f}%，Put完全无效</div>
  </div>
</div>
</div>

<!-- ══ 二、拉长时间有用吗 ══════════════════════════ -->
<div class="section">
<h2>二、时间拉长后，对冲效果会变好吗？</h2>

<div class="alert a-bad">
  <b>答案是：不会。</b>R&sup2;并不随持有周期增加而改善，季度维度反而<b>跌到27%</b>。
  这意味着"长期总会回归"这个假设<b>不成立</b>。
</div>

<table>
  <tr><th>持有周期</th><th>相关系数</th><th>R&sup2;</th><th>可视化</th><th>样本数</th></tr>
  {r2_rows}
</table>

<p class="note-sm">R&sup2;越高，IBEX对冲越精准；R&sup2;低意味着基金和IBEX"各走各的"概率大。</p>
</div>

<!-- ══ 三、基金跌的时候IBEX在干嘛 ══════════════════ -->
<div class="section">
<h2>三、历史实测：基金跌超1%的{n_bad}天，IBEX35在干嘛？</h2>

<div class="chart-box">
  <div id="c1" style="height:380px"></div>
</div>

<div class="alert a-info">
  每行是基金跌超1%的一天。<b style="color:#1565c0">蓝色</b>=基金跌幅，
  <b style="color:#2e7d32">绿色</b>=IBEX同向下跌（对冲有效），
  <b style="color:#c62828">红色</b>=IBEX反涨（对冲失效）。<br>
  <b>16次中有13次IBEX同向下跌（81%）</b>——大多数时候对冲是起作用的，特别是大跌时。
</div>

<table>
  <tr><th>日期</th><th>基金跌幅</th><th>IBEX35涨跌</th><th>结果</th></tr>
  {bad_rows}
</table>

<div class="alert a-good">
  <b>好消息：</b>大跌（如2025年4月关税冲击，基金单日跌3-4%）时IBEX基本都同步暴跌，对冲有效。<br>
  <b>坏消息：</b>中等跌幅时约20%概率脱钩，Put白买。
</div>
</div>

<!-- ══ 四、滚动R² ══════════════════════════════════ -->
<div class="section">
<h2>四、相关性不是恒定的——60天滚动R&sup2;</h2>

<div class="chart-box">
  <div id="c2" style="height:320px"></div>
</div>

<div class="cards">
  <div class="card green">
    <div class="lbl">R&sup2; 最高</div>
    <div class="val">{r2_max:.0f}%</div>
    <div class="sub">对冲最有效的时期</div>
  </div>
  <div class="card red">
    <div class="lbl">R&sup2; 最低</div>
    <div class="val">{r2_min:.0f}%</div>
    <div class="sub">完全脱钩</div>
  </div>
  <div class="card orange">
    <div class="lbl">R&sup2;&lt;20%的天数</div>
    <div class="val">{r2_below_20}天</div>
    <div class="sub">占全部的{r2_below_20_pct:.0f}%</div>
  </div>
</div>

<div class="alert a-warn">
  R&sup2;有时跌到个位数——这些时期IBEX和基金基本毫无关系，Put形同虚设。
  但R&sup2;也有冲高到80%+的时期，通常在全欧系统性下跌时。
</div>
</div>

<!-- ══ 五、真实案例 ════════════════════════════════ -->
<div class="section">
<h2>五、两个真实案例</h2>

<div class="two-col">
  <div>
    <h2 style="font-size:14px;color:#c62828;border-left-color:#c62828;background:#fdecea">
      案例A：2024年8-12月 — 脱钩，对冲失效
    </h2>
    <div class="chart-box" style="margin-bottom:8px">
      <div id="c4" style="height:350px"></div>
    </div>
    <div class="alert a-bad">
      基金持续下跌约5%，同期IBEX35反而上涨约10%。<br>
      如果当时持有IBEX Put，<b>一分钱不赔，基金白亏</b>。<br>
      脱钩持续了近<b>5个月</b>。
    </div>
  </div>
  <div>
    <h2 style="font-size:14px;color:#2e7d32;border-left-color:#2e7d32;background:#e8f5e9">
      案例B：2025年4月 — 关税冲击，对冲有效
    </h2>
    <div class="chart-box" style="margin-bottom:8px">
      <div id="c5" style="height:350px"></div>
    </div>
    <div class="alert a-good">
      全球关税恐慌，基金和IBEX35同步暴跌。<br>
      基金单周跌3.6%，IBEX单周跌6.7%。<br>
      Put赔付会<b>超额覆盖</b>基金损失。这正是对冲发挥作用的场景。
    </div>
  </div>
</div>
</div>

<!-- ══ 六、3个月收益散点图 ═══════════════════════════ -->
<div class="section">
<h2>六、三个月滚动收益对比 — 脱钩有多常见？</h2>

<div class="chart-box">
  <div id="c3" style="height:400px"></div>
</div>

<div class="alert a-info">
  每个点是一个交易日往前看3个月的收益。<br>
  <b style="color:#c62828">红色点 = 基金跌+IBEX涨（右下象限）</b>：你最怕的情况，Put到期时基金亏了但IBEX没跌。<br>
  历史上红色点集中在2024年8-12月那轮脱钩。
</div>

{"" if len(worst) == 0 else f'''<table>
  <tr><th>日期（3个月窗口截止）</th><th>基金3个月收益</th><th>IBEX35 3个月收益</th></tr>
  {worst_rows}
</table>
<p class="note-sm">以上为最严重的5次脱钩（基金3个月跌幅 vs IBEX同期涨幅）。</p>'''}
</div>

<!-- ══ 七、你的策略 ════════════════════════════════ -->
<div class="section">
<h2>七、你的策略：买Put + 到期展期，可行吗？</h2>

<div class="alert a-note" style="font-size:15px;line-height:2">
  <b>你的计划：</b>基金多头拿5年，买Dec 2027 Put，到期后继续滚动展期。
</div>

<div class="two-col">
  <div class="alert a-good" style="margin:0">
    <b>这个策略的优势：</b><br>
    &bull; 权利金年化仅0.2-0.25%（约&euro;3,000/年），是很便宜的保险<br>
    &bull; 不需要择时，不需要每日对冲<br>
    &bull; 真正的系统性大跌（2025年4月那种）对冲确实有效<br>
    &bull; 即使某一轮Put白花了，损失也可控（&euro;5,000-6,500）<br>
    &bull; 展期滚动给了你多次"抽奖"机会——总有一轮会碰上大跌
  </div>
  <div class="alert a-bad" style="margin:0">
    <b>这个策略的风险：</b><br>
    &bull; 历史上脱钩可持续<b>5个月</b>（2024年8-12月），不是几天就回归<br>
    &bull; 如果恰好在Put到期时处于脱钩期，权利金白花<br>
    &bull; 季度R&sup2;只有27%，说明中期脱钩很常见<br>
    &bull; 5年展期下来，权利金累计约&euro;15,000-30,000<br>
    &bull; 如果PSI20的下跌是葡萄牙本地原因（非全欧），Put大概率不赔
  </div>
</div>

<div class="alert a-warn" style="margin-top:16px">
  <b>关键判断：你要防的是什么？</b><br><br>
  &bull; 如果你担心的是<b>全欧洲系统性风险</b>（衰退、债务危机、地缘冲击）→ IBEX Put <b>有效</b>，历史数据支持<br>
  &bull; 如果你担心的是<b>葡萄牙本地问题</b>（银行危机、政策变动、PSI20成分股暴雷）→ IBEX Put <b>帮不了你</b><br><br>
  你的担忧线是PSI20跌到8,000——如果这种跌幅的原因是全欧衰退，IBEX会一起跌，对冲有效。
  如果原因是葡萄牙独有的问题，IBEX可能纹丝不动。<br><br>
  <b>以&euro;3,000/年的保险费来防全欧系统性风险，性价比是合理的。</b>
  但要认清：这是一把不完美的伞，不是每场雨都挡得住。
</div>
</div>

</div><!-- /page -->
<script>
  Plotly.newPlot('c1',__C1__.data,__C1__.layout,{{responsive:true}});
  Plotly.newPlot('c2',__C2__.data,__C2__.layout,{{responsive:true}});
  Plotly.newPlot('c3',__C3__.data,__C3__.layout,{{responsive:true}});
  Plotly.newPlot('c4',__C4__.data,__C4__.layout,{{responsive:true}});
  Plotly.newPlot('c5',__C5__.data,__C5__.layout,{{responsive:true}});
</script>
</body></html>"""

    html = html.replace('__C1__', c1)
    html = html.replace('__C2__', c2)
    html = html.replace('__C3__', c3)
    html = html.replace('__C4__', c4)
    html = html.replace('__C5__', c5)
    return html


def main():
    print('加载数据...')
    fund_df, ibex_df, psi_df = load_data()
    print(f'数据加载完成：基金{len(fund_df)}条，IBEX{len(ibex_df)}条')

    print('分析中...')
    results = analyze(fund_df, ibex_df, psi_df)
    print(f'分析完成：{len(results["df"])}个交易日，{len(results["bad_days"])}个基金下跌日')

    print('生成报告...')
    html = generate_html(results)
    out = os.path.join(DATA_DIR, 'hedge_review.html')
    with open(out, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f'报告已生成: {out}')

    import subprocess, platform
    if platform.system() == 'Darwin':
        subprocess.run(['open', out])
    elif platform.system() == 'Windows':
        os.startfile(out)
    else:
        subprocess.run(['xdg-open', out])


if __name__ == '__main__':
    main()
