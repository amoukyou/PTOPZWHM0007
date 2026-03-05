"""
葡萄牙基金 vs 做空IBEX35 对冲分析
- 计算最优对冲比例 (Beta/OLS)
- 模拟历史对冲组合净值走势
- 对比对冲前后的风险指标
- 滚动Beta分析（对冲比例随时间的变化）
"""

import os
import sys
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

sys.stdout.reconfigure(encoding='utf-8')

DATA_DIR = os.path.dirname(os.path.abspath(__file__))


def load(filename, col_name):
    df = pd.read_csv(os.path.join(DATA_DIR, filename), parse_dates=['Date'])
    return df[['Date', 'Open']].rename(columns={'Open': col_name}).sort_values('Date')


def calc_stats(returns, label):
    ann = 252 ** 0.5
    total = (1 + returns).prod() - 1
    vol = returns.std() * ann
    mean_r = returns.mean() * 252
    sharpe = mean_r / vol if vol > 0 else 0
    cum = (1 + returns).cumprod()
    drawdown = (cum / cum.cummax() - 1)
    max_dd = drawdown.min()
    return {
        '标的': label,
        '总收益率': f'{total*100:.2f}%',
        '年化收益率': f'{mean_r*100:.2f}%',
        '年化波动率': f'{vol*100:.2f}%',
        '夏普比率': f'{sharpe:.2f}',
        '最大回撤': f'{max_dd*100:.2f}%',
    }


def main():
    fund = load('PTOPZWHM0007_daily_2024-2026.csv', 'fund')
    ibex = load('IBEX35_daily_2024-2026.csv', 'ibex')

    df = pd.merge(fund, ibex, on='Date').sort_values('Date').reset_index(drop=True)
    df['r_fund'] = df['fund'].pct_change()
    df['r_ibex'] = df['ibex'].pct_change()
    df = df.dropna().reset_index(drop=True)

    # ── 1. 全期Beta（最优对冲比例）──────────────────────────────
    beta = df['r_fund'].cov(df['r_ibex']) / df['r_ibex'].var()
    corr = df['r_fund'].corr(df['r_ibex'])

    # 对冲组合收益 = 基金收益 - beta * IBEX收益（做空beta份IBEX）
    df['r_hedged'] = df['r_fund'] - beta * df['r_ibex']

    # ── 2. 滚动Beta（30日窗口）──────────────────────────────────
    roll_beta = df['r_fund'].rolling(30).apply(
        lambda x: x.cov(df['r_ibex'].loc[x.index]) / df['r_ibex'].loc[x.index].var()
        if df['r_ibex'].loc[x.index].var() > 0 else np.nan
    )
    # 更高效的方式
    roll_cov  = df['r_fund'].rolling(30).cov(df['r_ibex'])
    roll_var  = df['r_ibex'].rolling(30).var()
    df['roll_beta'] = roll_cov / roll_var

    # 动态对冲：每日用前30日beta
    df['r_hedged_dynamic'] = df['r_fund'] - df['roll_beta'].shift(1) * df['r_ibex']

    # ── 3. 累计净值（起始=100）──────────────────────────────────
    df['nav_fund']           = (1 + df['r_fund']).cumprod() * 100
    df['nav_ibex_short']     = (1 - df['r_ibex']).cumprod() * 100   # 纯做空IBEX
    df['nav_hedged']         = (1 + df['r_hedged']).cumprod() * 100
    df['nav_hedged_dynamic'] = (1 + df['r_hedged_dynamic'].fillna(df['r_fund'])).cumprod() * 100

    # ── 4. 统计对比 ──────────────────────────────────────────────
    stats = pd.DataFrame([
        calc_stats(df['r_fund'],            '原始基金（未对冲）'),
        calc_stats(-df['r_ibex'] * beta,    f'做空IBEX35部分（beta={beta:.3f}）'),
        calc_stats(df['r_hedged'],           '对冲组合（固定beta）'),
        calc_stats(df['r_hedged_dynamic'].fillna(df['r_fund']), '对冲组合（滚动beta）'),
    ])

    # ── 5. 作图 ──────────────────────────────────────────────────
    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=(
            '累计净值走势（起始=100）',
            '滚动30日Beta（做空IBEX35的比例）',
            '对冲前 vs 对冲后 日收益率分布',
            '滚动30日相关性（对冲效果监控）',
            '最大回撤对比',
            '风险指标汇总表',
        ),
        specs=[
            [{}, {}],
            [{}, {}],
            [{}, {'type': 'table'}],
        ],
        vertical_spacing=0.10, horizontal_spacing=0.08,
        row_heights=[0.35, 0.35, 0.30],
    )

    colors = {
        'fund':     '#1f77b4',
        'hedged':   '#2ca02c',
        'dynamic':  '#ff7f0e',
        'ibex_s':   '#d62728',
        'beta':     '#9467bd',
        'corr':     '#8c564b',
    }

    # 图1: 累计净值
    for col, label, color in [
        ('nav_fund',           '原始基金（未对冲）',       colors['fund']),
        ('nav_hedged',         f'对冲组合（固定beta={beta:.2f}）', colors['hedged']),
        ('nav_hedged_dynamic', '对冲组合（滚动beta）',     colors['dynamic']),
        ('nav_ibex_short',     f'纯做空IBEX35（×{beta:.2f}）', colors['ibex_s']),
    ]:
        fig.add_trace(go.Scatter(
            x=df['Date'], y=df[col], mode='lines', name=label,
            line=dict(color=color, width=2),
            hovertemplate=f'<b>{label}</b><br>%{{x|%Y-%m-%d}}<br>净值: %{{y:.2f}}<extra></extra>',
        ), row=1, col=1)
    fig.add_hline(y=100, line_dash='dash', line_color='gray', opacity=0.4, row=1, col=1)
    fig.update_yaxes(title_text='净值', row=1, col=1)

    # 图2: 滚动Beta
    fig.add_trace(go.Scatter(
        x=df['Date'], y=df['roll_beta'], mode='lines', name='滚动30日Beta',
        line=dict(color=colors['beta'], width=1.8),
        hovertemplate='%{x|%Y-%m-%d}<br>Beta: %{y:.3f}<br><i>即：每1元基金需做空%{y:.3f}元IBEX</i><extra></extra>',
        showlegend=False,
    ), row=1, col=2)
    fig.add_hline(y=beta, line_dash='dash', line_color='gray', opacity=0.5, row=1, col=2,
                  annotation_text=f'全期平均Beta={beta:.3f}',
                  annotation_position='top right',
                  annotation_font=dict(size=10))
    fig.update_yaxes(title_text='Beta（做空比例）', row=1, col=2)
    fig.add_annotation(
        x=0.5, y=1.05, xref='x2 domain', yref='y2 domain',
        text=f'<b>当前建议：每持有100万基金，做空 {beta*100:.1f}万元 IBEX35</b>',
        showarrow=False, font=dict(size=11, color=colors['beta']),
        bgcolor='rgba(148,103,189,0.1)', bordercolor=colors['beta'], borderwidth=1,
    )

    # 图3: 日收益率分布直方图
    for r, label, color in [
        (df['r_fund'],    '原始基金', colors['fund']),
        (df['r_hedged'],  '对冲后',   colors['hedged']),
    ]:
        fig.add_trace(go.Histogram(
            x=r * 100, name=label, opacity=0.6,
            marker_color=color, nbinsx=60,
            hovertemplate='日收益率: %{x:.2f}%<br>天数: %{y}<extra></extra>',
        ), row=2, col=1)
    fig.update_layout(barmode='overlay')
    fig.update_xaxes(title_text='日收益率 (%)', row=2, col=1)
    fig.update_yaxes(title_text='天数', row=2, col=1)

    # 图4: 滚动相关性
    roll_corr = df['r_fund'].rolling(30).corr(df['r_ibex'])
    fig.add_trace(go.Scatter(
        x=df['Date'], y=roll_corr, mode='lines', name='基金 vs IBEX 滚动相关性',
        line=dict(color=colors['corr'], width=1.8),
        hovertemplate='%{x|%Y-%m-%d}<br>相关性: %{y:.3f}<extra></extra>',
        showlegend=False,
    ), row=2, col=2)
    fig.add_hline(y=0,    line_dash='dash', line_color='gray',          opacity=0.4, row=2, col=2)
    fig.add_hline(y=0.7,  line_dash='dot',  line_color='#2ca02c',       opacity=0.5, row=2, col=2,
                  annotation_text='强正相关(0.7)', annotation_position='top left',
                  annotation_font=dict(size=9, color='#2ca02c'))
    fig.add_hline(y=-0.7, line_dash='dot',  line_color='#d62728',       opacity=0.5, row=2, col=2,
                  annotation_text='强负相关(-0.7)', annotation_position='bottom left',
                  annotation_font=dict(size=9, color='#d62728'))
    fig.update_yaxes(range=[-1, 1], title_text='相关系数', row=2, col=2)
    fig.add_annotation(
        x=0.5, y=-0.15, xref='x4 domain', yref='y4 domain',
        text='<i>相关性高时对冲效果好，低时对冲效果减弱</i>',
        showarrow=False, font=dict(size=10, color='gray'),
    )

    # 图5: 最大回撤
    fill_colors = {'fund': 'rgba(31,119,180,0.12)', 'hedged': 'rgba(44,160,44,0.12)'}
    for col, label, ckey in [
        ('nav_fund',    '原始基金', 'fund'),
        ('nav_hedged',  '对冲后',   'hedged'),
    ]:
        cum = df[col]
        dd = (cum / cum.cummax() - 1) * 100
        fig.add_trace(go.Scatter(
            x=df['Date'], y=dd, mode='lines', name=f'{label}回撤',
            line=dict(color=colors[ckey], width=1.5), fill='tozeroy',
            fillcolor=fill_colors[ckey],
            hovertemplate=f'<b>{label}</b><br>%{{x|%Y-%m-%d}}<br>回撤: %{{y:.2f}}%<extra></extra>',
        ), row=3, col=1)
    fig.update_yaxes(title_text='回撤 (%)', row=3, col=1)
    fig.update_xaxes(title_text='日期', row=3, col=1)

    # 图6: 统计汇总表
    fig.add_trace(go.Table(
        header=dict(
            values=[f'<b>{c}</b>' for c in stats.columns],
            fill_color='#1f77b4', font=dict(color='white', size=11),
            align='center', height=28,
        ),
        cells=dict(
            values=[stats[c].tolist() for c in stats.columns],
            fill_color=[['#f0f8ff', '#e8f5e9', '#e8f5e9', '#fff3e0'] * 2],
            align='center', font=dict(size=11), height=26,
        ),
    ), row=3, col=2)

    # ── 整体布局 ──────────────────────────────────────────────────
    fig.update_layout(
        title=dict(
            text=f'做空IBEX35对冲葡萄牙基金分析  |  全期对冲比例(Beta)={beta:.3f}  |  全期相关性={corr:.3f}',
            font=dict(size=18),
        ),
        template='plotly_white', height=1100, showlegend=True,
        legend=dict(x=0.01, y=0.99, bgcolor='rgba(255,255,255,0.8)', bordercolor='lightgray', borderwidth=1),
        hovermode='x unified',
    )

    out_path = os.path.join(DATA_DIR, 'hedge_analysis.html')
    fig.write_html(out_path, include_plotlyjs='cdn')
    print(f'对冲分析报告已生成: {out_path}')
    os.startfile(out_path)


if __name__ == '__main__':
    main()
