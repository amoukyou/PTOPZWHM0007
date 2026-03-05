"""
IBEX35 Put期权对冲分析
- 用Black-Scholes定价不同行权价/期限的put
- 计算需要购买的合约数量（相对基金仓位）
- 展示年化损耗、盈亏平衡点、到期保护收益
"""

import os
import sys
import math
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

sys.stdout.reconfigure(encoding='utf-8')
DATA_DIR = os.path.dirname(os.path.abspath(__file__))

# ── 可调参数 ──────────────────────────────────────────────────────
FUND_VALUE    = 1_000_000   # 基金仓位（欧元），按比例缩放即可
BETA          = 0.194       # 基金对IBEX35的Beta（由hedge_analysis.py算出）
ECB_RATE      = 0.026       # 欧央行无风险利率（约2.6%）
CONTRACT_MULT = 1           # IBEX期权合约乘数：1欧元/点（MEFF标准合约）
# ─────────────────────────────────────────────────────────────────


def norm_cdf(x):
    return (1 + math.erf(x / math.sqrt(2))) / 2


def bs_put(S, K, T, r, sigma):
    """Black-Scholes Put定价，返回(价格, Delta, 隐含年化成本%)"""
    if T <= 0:
        return max(K - S, 0), -1.0, 0
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    price = K * math.exp(-r * T) * norm_cdf(-d2) - S * norm_cdf(-d1)
    delta = norm_cdf(d1) - 1   # put delta为负
    return price, delta, price / S * 100  # 价格占现价%


def build_scenarios(spot, sigma, r=ECB_RATE):
    """
    行权价: ATM(100%), 95%, 90%, 85%, 80%
    期限:   3个月, 6个月, 12个月
    """
    ibex_exposure = FUND_VALUE * BETA          # 需要对冲的IBEX敞口
    rows = []
    for T_months in [3, 6, 12]:
        T = T_months / 12
        for k_pct in [1.00, 0.97, 0.95, 0.92, 0.90, 0.85, 0.80]:
            K = spot * k_pct
            price, delta, pct_spot = bs_put(spot, K, T, r, sigma)

            # 合约数量：用delta对冲（更精确）
            n_contracts_delta = ibex_exposure / (spot * CONTRACT_MULT * abs(delta))
            # 合约数量：按名义敞口（简化）
            n_contracts_notional = ibex_exposure / (spot * CONTRACT_MULT)

            total_premium_delta    = price * n_contracts_delta    * CONTRACT_MULT
            total_premium_notional = price * n_contracts_notional * CONTRACT_MULT

            # 年化损耗 = 总权利金 / 基金价值 / 期限(年)
            annual_drag_delta    = total_premium_delta    / FUND_VALUE / T * 100
            annual_drag_notional = total_premium_notional / FUND_VALUE / T * 100

            # 盈亏平衡：IBEX需要跌多少才能让put回本
            # 到期时put价值 = K - S_T  →  S_T = K - price
            breakeven_spot = K - price
            breakeven_drop = (breakeven_spot - spot) / spot * 100

            rows.append({
                '期限':          f'{T_months}个月',
                '行权价':        int(K),
                '行权价%':       f'{k_pct*100:.0f}% ATM' if k_pct == 1 else f'{k_pct*100:.0f}%',
                'OTM幅度':       f'{(1-k_pct)*100:.0f}%',
                '单张权利金(€)': round(price, 1),
                '权利金/现价':   f'{pct_spot:.2f}%',
                'Put Delta':     round(delta, 3),
                '合约数(Delta对冲)':   round(n_contracts_delta, 1),
                '合约数(名义对冲)':    round(n_contracts_notional, 1),
                '总权利金_Delta(€)':   round(total_premium_delta, 0),
                '总权利金_名义(€)':    round(total_premium_notional, 0),
                '年化损耗%_Delta':     round(annual_drag_delta, 2),
                '年化损耗%_名义':      round(annual_drag_notional, 2),
                '盈亏平衡跌幅':        f'{breakeven_drop:.1f}%',
                '_T': T, '_K': K, '_k_pct': k_pct, '_price': price,
                '_n_notional': n_contracts_notional,
            })
    return pd.DataFrame(rows)


def main():
    # 读取IBEX数据
    ibex = pd.read_csv(os.path.join(DATA_DIR, 'IBEX35_daily_2024-2026.csv'), parse_dates=['Date'])
    ibex = ibex.sort_values('Date').reset_index(drop=True)
    ibex['r'] = ibex['Open'].pct_change()
    sigma_hist = ibex['r'].dropna().std() * (252 ** 0.5)
    spot = ibex['Open'].iloc[-1]
    last_date = ibex['Date'].iloc[-1].strftime('%Y-%m-%d')

    df = build_scenarios(spot, sigma_hist)

    # ── 图表 ──────────────────────────────────────────────────────
    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=(
            '年化损耗率（%/年）—— 不同期限 × 行权价',
            '合约数量（名义对冲，基金100万€）',
            '盈亏平衡：IBEX需要跌多少才能回本权利金',
            '到期日保护效果：IBEX不同跌幅下组合净值变化',
            '权利金总成本（€）vs 对冲缺口说明',
            '综合方案对比表',
        ),
        specs=[
            [{}, {}],
            [{}, {}],
            [{}, {'type': 'table'}],
        ],
        vertical_spacing=0.11, horizontal_spacing=0.09,
        row_heights=[0.33, 0.33, 0.34],
    )

    period_colors = {'3个月': '#1f77b4', '6个月': '#ff7f0e', '12个月': '#2ca02c'}
    k_pcts = sorted(df['_k_pct'].unique(), reverse=True)

    # ── 图1：年化损耗 ────────────────────────────────────────────
    for period, grp in df.groupby('期限', sort=False):
        fig.add_trace(go.Scatter(
            x=grp['行权价%'], y=grp['年化损耗%_名义'],
            mode='lines+markers+text', name=period,
            line=dict(color=period_colors[period], width=2),
            marker=dict(size=8),
            text=[f'{v:.2f}%' for v in grp['年化损耗%_名义']],
            textposition='top center', textfont=dict(size=9),
            hovertemplate='%{x}<br>年化损耗: %{y:.2f}%/年<extra>' + period + '</extra>',
        ), row=1, col=1)
    fig.update_xaxes(title_text='行权价（相对现价%）', row=1, col=1)
    fig.update_yaxes(title_text='年化损耗 (%/年)', row=1, col=1)
    fig.add_annotation(x=0.5, y=1.08, xref='x domain', yref='y domain',
        text=f'<i>IBEX现价={spot:.0f}  历史波动率={sigma_hist*100:.1f}%  无风险利率={ECB_RATE*100:.1f}%</i>',
        showarrow=False, font=dict(size=9, color='gray'), row=1, col=1)

    # ── 图2：合约数量 ────────────────────────────────────────────
    for period, grp in df.groupby('期限', sort=False):
        fig.add_trace(go.Bar(
            x=grp['行权价%'], y=grp['合约数(名义对冲)'],
            name=period, marker_color=period_colors[period],
            text=[f'{v:.0f}张' for v in grp['合约数(名义对冲)']],
            textposition='outside', textfont=dict(size=9),
            hovertemplate='%{x}<br>合约数: %{y:.1f}张<extra>' + period + '</extra>',
            showlegend=False,
        ), row=1, col=2)
    fig.add_annotation(
        text=f'基金100万€ × Beta {BETA} = IBEX敞口 {FUND_VALUE*BETA/1000:.0f}万€<br>'
             f'合约数 ≈ {FUND_VALUE*BETA:.0f} ÷ {spot:.0f}（现价）÷ 1（乘数）≈ {FUND_VALUE*BETA/spot:.0f}张',
        x=0.5, y=-0.22, xref='x2 domain', yref='y2 domain',
        showarrow=False, font=dict(size=10), bgcolor='#fffde7', bordercolor='#f9a825', borderwidth=1,
    )
    fig.update_xaxes(title_text='行权价', row=1, col=2)
    fig.update_yaxes(title_text='合约数（张）', row=1, col=2)
    fig.update_layout(barmode='group')

    # ── 图3：盈亏平衡跌幅 ────────────────────────────────────────
    for period, grp in df.groupby('期限', sort=False):
        # 盈亏平衡跌幅是负数，取绝对值展示
        be_vals = grp['盈亏平衡跌幅'].str.replace('%','').astype(float)
        fig.add_trace(go.Scatter(
            x=grp['行权价%'], y=be_vals,
            mode='lines+markers', name=period,
            line=dict(color=period_colors[period], width=2, dash='dot'),
            marker=dict(size=8),
            text=[f'{v:.1f}%' for v in be_vals],
            textposition='bottom center', textfont=dict(size=9),
            hovertemplate='%{x}<br>IBEX需下跌: %{y:.1f}%才能回本权利金<extra>' + period + '</extra>',
            showlegend=False,
        ), row=2, col=1)
    fig.add_hline(y=0, line_dash='dash', line_color='gray', opacity=0.4, row=2, col=1)
    fig.update_xaxes(title_text='行权价', row=2, col=1)
    fig.update_yaxes(title_text='盈亏平衡跌幅 (%)', row=2, col=1)
    fig.add_annotation(
        text='<i>该线以上 = 权利金亏损；线以下 = 权利金回本，开始真正对冲</i>',
        x=0.5, y=0.02, xref='x3 domain', yref='y3 domain',
        showarrow=False, font=dict(size=9, color='gray'),
    )

    # ── 图4：到期日保护效果（组合净值 vs IBEX跌幅）────────────────
    ibex_drops = np.linspace(-0.40, 0.10, 200)
    ibex_exposure = FUND_VALUE * BETA

    # 假设基金跌幅 = IBEX跌幅 × Beta
    fund_drops = ibex_drops * BETA

    # 无对冲
    portfolio_no_hedge = FUND_VALUE * (1 + fund_drops)
    fig.add_trace(go.Scatter(
        x=ibex_drops * 100, y=(portfolio_no_hedge / FUND_VALUE - 1) * 100,
        mode='lines', name='无对冲', line=dict(color='#d62728', width=2.5),
        hovertemplate='IBEX跌幅: %{x:.1f}%<br>组合损益: %{y:.2f}%<extra>无对冲</extra>',
    ), row=2, col=2)

    # 几个典型方案
    highlight_scenarios = [
        ('12个月', 0.95, '#2ca02c', '12M Put 95%'),
        ('12个月', 0.90, '#1f77b4', '12M Put 90%'),
        ('6个月',  0.95, '#ff7f0e', '6M Put 95%'),
    ]
    for period, k_pct, color, label in highlight_scenarios:
        row_s = df[(df['期限'] == period) & (abs(df['_k_pct'] - k_pct) < 0.001)].iloc[0]
        K     = row_s['_K']
        price = row_s['_price']
        n     = row_s['_n_notional']
        total_prem = price * n * CONTRACT_MULT

        put_payoff_per_contract = np.maximum(K - spot * (1 + ibex_drops), 0)
        total_put_payoff = put_payoff_per_contract * n * CONTRACT_MULT
        portfolio_hedged = FUND_VALUE * (1 + fund_drops) + total_put_payoff - total_prem
        fig.add_trace(go.Scatter(
            x=ibex_drops * 100, y=(portfolio_hedged / FUND_VALUE - 1) * 100,
            mode='lines', name=label, line=dict(color=color, width=2),
            hovertemplate='IBEX跌幅: %{x:.1f}%<br>组合损益: %{y:.2f}%<extra>' + label + '</extra>',
        ), row=2, col=2)

    fig.add_vline(x=0, line_dash='dash', line_color='gray', opacity=0.4, row=2, col=2)
    fig.update_xaxes(title_text='IBEX35到期跌幅 (%)', row=2, col=2)
    fig.update_yaxes(title_text='组合损益 (%)', row=2, col=2)

    # ── 图5：总权利金成本 ─────────────────────────────────────────
    for period, grp in df.groupby('期限', sort=False):
        fig.add_trace(go.Bar(
            x=grp['行权价%'], y=grp['总权利金_名义(€)'],
            name=period, marker_color=period_colors[period],
            text=[f'€{v:,.0f}' for v in grp['总权利金_名义(€)']],
            textposition='outside', textfont=dict(size=9),
            hovertemplate='%{x}<br>总权利金: €%{y:,.0f}<extra>' + period + '</extra>',
            showlegend=False,
        ), row=3, col=1)
    fig.update_xaxes(title_text='行权价', row=3, col=1)
    fig.update_yaxes(title_text='总权利金（€）', row=3, col=1)

    # ── 图6：汇总对比表 ─────────────────────────────────────────
    show_cols = ['期限', '行权价%', '权利金/现价', '合约数(名义对冲)', '总权利金_名义(€)', '年化损耗%_名义', '盈亏平衡跌幅']
    table_df = df[show_cols].copy()
    table_df.columns = ['期限', '行权价', '单张权利金/现价', '合约数', '总权利金(€)', '年化损耗%', '盈亏平衡跌幅']

    # 高亮推荐行（6M/12M, 90-95%）
    row_colors = []
    for _, r_ in df.iterrows():
        if r_['期限'] in ['6个月', '12个月'] and r_['_k_pct'] in [0.95, 0.90]:
            row_colors.append('#e8f5e9')  # 绿色高亮
        else:
            row_colors.append('white')

    fig.add_trace(go.Table(
        header=dict(
            values=[f'<b>{c}</b>' for c in table_df.columns],
            fill_color='#1565c0', font=dict(color='white', size=11),
            align='center', height=28,
        ),
        cells=dict(
            values=[table_df[c].tolist() for c in table_df.columns],
            fill_color=[row_colors] * len(table_df.columns),
            align='center', font=dict(size=10), height=24,
        ),
    ), row=3, col=2)

    # ── 整体布局 ─────────────────────────────────────────────────
    ibex_exposure = FUND_VALUE * BETA
    fig.update_layout(
        title=dict(
            text=(
                f'IBEX35 Put期权对冲方案分析  |  '
                f'基金仓位: {FUND_VALUE/1e4:.0f}万€  ·  '
                f'Beta={BETA}  →  IBEX敞口: {ibex_exposure/1e4:.1f}万€  ·  '
                f'IBEX现价: {spot:.0f}  ·  历史波动率: {sigma_hist*100:.1f}%'
            ),
            font=dict(size=15),
        ),
        template='plotly_white', height=1150, showlegend=True,
        legend=dict(x=0.01, y=0.99, bgcolor='rgba(255,255,255,0.85)',
                    bordercolor='lightgray', borderwidth=1),
        hovermode='x unified',
    )

    out_path = os.path.join(DATA_DIR, 'put_hedge_analysis.html')
    fig.write_html(out_path, include_plotlyjs='cdn')
    print(f'Put对冲分析报告已生成: {out_path}')

    # 打印推荐方案
    print('\n══════════════════════════════════════════')
    print('  推荐方案（绿色高亮行）')
    print('══════════════════════════════════════════')
    recommended = df[df['期限'].isin(['6个月', '12个月']) & df['_k_pct'].isin([0.95, 0.90])]
    for _, r_ in recommended.iterrows():
        print(f"\n  [{r_['期限']}] 行权价 {r_['行权价']} ({r_['行权价%']})")
        print(f"    买入 {r_['合约数(名义对冲)']:.0f} 张 IBEX35 Put")
        print(f"    单张权利金:  €{r_['单张权利金(€)']:.1f}  ({r_['权利金/现价']})")
        print(f"    总权利金:    €{r_['总权利金_名义(€)']:,.0f}")
        print(f"    年化损耗:    {r_['年化损耗%_名义']:.2f}% / 年")
        print(f"    盈亏平衡:    IBEX需下跌 {r_['盈亏平衡跌幅']} 才开始净获益")
    print()
    os.startfile(out_path)


if __name__ == '__main__':
    main()
