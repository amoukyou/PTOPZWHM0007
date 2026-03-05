"""
葡萄牙对冲基金 - 历史走势 & 相关性分析
用法:
  1. 单只基金走势: 直接运行，读取同目录下所有 *_daily_*.csv
  2. 多只基金相关性: 放入更多CSV文件，自动计算相关性矩阵

CSV要求: 第1列=Date, 第2列=Open (标准OHLCV格式)
"""

import os
import sys
import glob
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

sys.stdout.reconfigure(encoding='utf-8')

DATA_DIR = os.path.dirname(os.path.abspath(__file__))


def load_fund_data(csv_path):
    """读取CSV，返回(基金名, DataFrame[date, open])"""
    name = os.path.basename(csv_path).split('_daily_')[0]
    df = pd.read_csv(csv_path, parse_dates=['Date'])
    df = df[['Date', 'Open']].rename(columns={'Open': name})
    df = df.sort_values('Date').reset_index(drop=True)
    return name, df


def main():
    csv_files = sorted(glob.glob(os.path.join(DATA_DIR, '*_daily_*.csv')))
    if not csv_files:
        print('未找到CSV文件')
        return

    # 加载所有基金数据
    funds = {}
    merged = None
    for f in csv_files:
        name, df = load_fund_data(f)
        funds[name] = df
        if merged is None:
            merged = df
        else:
            merged = pd.merge(merged, df, on='Date', how='outer')

    merged = merged.sort_values('Date').reset_index(drop=True)
    fund_names = [c for c in merged.columns if c != 'Date']
    n_funds = len(fund_names)

    # --- 构建图表 ---
    if n_funds == 1:
        # 单只基金: 简洁折线图
        fig = go.Figure()
        name = fund_names[0]
        fig.add_trace(go.Scatter(
            x=merged['Date'], y=merged[name],
            mode='lines', name=name,
            line=dict(width=2, color='#1f77b4'),
            hovertemplate='%{x|%Y-%m-%d}<br>价格: %{y:.2f}<extra></extra>'
        ))
        fig.update_layout(
            title=dict(text=f'{name} 历史价格走势', font=dict(size=20)),
            xaxis_title='日期', yaxis_title='开盘价',
            template='plotly_white',
            hovermode='x unified',
            xaxis=dict(
                rangeslider=dict(visible=True),
                rangeselector=dict(buttons=[
                    dict(count=1, label='1M', step='month'),
                    dict(count=3, label='3M', step='month'),
                    dict(count=6, label='6M', step='month'),
                    dict(count=1, label='1Y', step='year'),
                    dict(label='ALL', step='all'),
                ])
            ),
            height=600,
        )
    else:
        # 多只基金: 走势 + 归一化对比 + 相关性热力图
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                '原始价格走势（开盘价）',
                '归一化走势对比（起始日=100，反映涨跌幅）',
                '整体相关性矩阵（基于日收益率，+1完全同向，-1完全反向）',
                '滚动30日相关性（观察相关性随时间变化）',
            ),
            specs=[[{}, {}], [{'type': 'heatmap'}, {}]],
            vertical_spacing=0.14, horizontal_spacing=0.08,
        )
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                  '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

        # 1) 原始价格
        for i, name in enumerate(fund_names):
            fig.add_trace(go.Scatter(
                x=merged['Date'], y=merged[name],
                mode='lines', name=name,
                line=dict(color=colors[i % len(colors)]),
                legendgroup=name,
            ), row=1, col=1)

        # 2) 归一化 (首个非NaN值 = 100)
        norm_df = merged.copy()
        for name in fund_names:
            first_val = norm_df[name].dropna().iloc[0]
            norm_df[name] = norm_df[name] / first_val * 100

        for i, name in enumerate(fund_names):
            fig.add_trace(go.Scatter(
                x=norm_df['Date'], y=norm_df[name],
                mode='lines', name=f'{name} (归一化)',
                line=dict(color=colors[i % len(colors)]),
                legendgroup=name, showlegend=False,
            ), row=1, col=2)

        # 3) 相关性矩阵
        returns = merged[fund_names].pct_change().dropna()
        corr = returns.corr()

        def corr_label(v):
            av = abs(v)
            if av > 0.7:   strength = '强'
            elif av > 0.4: strength = '中'
            else:          strength = '弱'
            direction = '正' if v > 0 else ('负' if v < 0 else '')
            return f'{v:.3f}<br>({strength}{direction}相关)'

        text_labels = [[corr_label(corr.iloc[r, c]) for c in range(len(fund_names))]
                       for r in range(len(fund_names))]

        fig.add_trace(go.Heatmap(
            z=corr.values, x=list(corr.columns), y=list(corr.index),
            colorscale='RdBu_r', zmin=-1, zmax=1,
            text=text_labels,
            texttemplate='%{text}',
            textfont=dict(size=12),
            hovertemplate='%{y} vs %{x}<br>相关系数: %{z:.3f}<extra></extra>',
            showscale=True,
            colorbar=dict(title='相关系数', tickvals=[-1, -0.7, 0, 0.7, 1],
                          ticktext=['-1<br>完全反向', '-0.7<br>强负', '0<br>无关', '0.7<br>强正', '1<br>完全同向']),
        ), row=2, col=1)

        # 4) 滚动相关性 (所有两两组合)
        if n_funds >= 2:
            from itertools import combinations
            pair_colors = ['#d62728', '#1f77b4', '#2ca02c', '#ff7f0e', '#9467bd',
                           '#8c564b', '#e377c2', '#bcbd22', '#17becf', '#7f7f7f']
            dates_for_roll = merged['Date'].iloc[1:].reset_index(drop=True)
            roll_annotations = []
            for idx, (a, b) in enumerate(combinations(fund_names, 2)):
                roll_corr = returns[a].rolling(30).corr(returns[b])
                color = pair_colors[idx % len(pair_colors)]
                label = f'{a} vs {b}'
                fig.add_trace(go.Scatter(
                    x=dates_for_roll,
                    y=roll_corr.values,
                    mode='lines', name=label,
                    line=dict(color=color, width=1.8),
                    hovertemplate=f'<b>{label}</b><br>%{{x|%Y-%m-%d}}<br>30日相关性: %{{y:.3f}}<extra></extra>',
                ), row=2, col=2)
                # 在末尾加标注
                last_valid = roll_corr.dropna()
                if len(last_valid) > 0:
                    roll_annotations.append(dict(
                        x=dates_for_roll.iloc[roll_corr.index.get_loc(last_valid.index[-1])],
                        y=last_valid.iloc[-1],
                        text=f'  {label}<br>  ({last_valid.iloc[-1]:.2f})',
                        showarrow=False,
                        font=dict(color=color, size=10),
                        xref='x4', yref='y4',
                        xanchor='left',
                    ))

            # y=0 基准线
            fig.add_hline(y=0,   line_dash='dash', line_color='gray',   opacity=0.4, row=2, col=2)
            # ±0.7 强相关参考线
            fig.add_hline(y=0.7, line_dash='dot',  line_color='#2ca02c', opacity=0.5, row=2, col=2,
                          annotation_text='强正相关(0.7)', annotation_position='top left',
                          annotation_font=dict(size=9, color='#2ca02c'))
            fig.add_hline(y=-0.7, line_dash='dot', line_color='#d62728', opacity=0.5, row=2, col=2,
                          annotation_text='强负相关(-0.7)', annotation_position='bottom left',
                          annotation_font=dict(size=9, color='#d62728'))
            fig.update_yaxes(range=[-1, 1], title_text='相关系数', row=2, col=2)
            fig.update_xaxes(title_text='日期', row=2, col=2)
            fig.update_layout(annotations=roll_annotations)

        fig.update_layout(
            title=dict(text='葡萄牙基金 - 走势与相关性分析', font=dict(size=20)),
            template='plotly_white', height=900, showlegend=True,
            hovermode='x unified',
        )

    # 输出HTML
    out_path = os.path.join(DATA_DIR, 'fund_analysis.html')
    fig.write_html(out_path, include_plotlyjs='cdn')
    print(f'图表已生成: {out_path}')
    os.startfile(out_path)


if __name__ == '__main__':
    main()
