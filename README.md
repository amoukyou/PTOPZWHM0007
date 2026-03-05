# 葡萄牙基金对冲分析

**Optimize Portugal Golden Opportunities Fund**（ISIN: PTOPZWHM0007）的下行风险对冲方案。

## 在线查看报告

**[点击查看完整对冲报告（hedge_final.html）](https://htmlpreview.github.io/?https://github.com/amoukyou/PTOPZWHM0007/blob/main/hedge_final.html)**

> 如果上面的链接加载慢或图表不显示，请下载 `hedge_final.html` 到本地用浏览器打开。

## 背景

- 持仓：€507,000买入，当前市值€651,000（+28%）
- 买入日：2024年7月16日
- 计划持有5年
- 担忧：俄乌停战、欧盟加息、美欧关税等全欧系统性风险导致基金大跌

## 核心结论

| | 原方案（已否决） | 修正方案（推荐） |
|---|---|---|
| 策略 | 21个月 IBEX35 95% OTM Put | **12个月 IBEX35 ATM Put，年滚** |
| 问题 | 牛市中行权价被甩开，25次回测全部归零 | 行权价每年刷新，不怕牛市漂移 |
| 年化成本 | ~0.9% | ~2.4% |
| 操作频率 | 一次买入不用管 | 每年滚仓一次 |
| 5年总保费 | ~€19,000（但完全无效） | ~€83,000（有效保护） |

## 文件说明

| 文件 | 说明 |
|------|------|
| `hedge_final.py` | **主文件** — 生成完整对冲报告（v4），含5张交互图 |
| `hedge_final.html` | 生成的报告网页，可直接浏览器打开 |
| `hedge_report.py` | 早期报告v1（已弃用） |
| `hedge_review.py` | 早期报告v2（已弃用） |
| `fund_chart.py` | 基金走势图 + 多品种相关性分析 |
| `hedge_analysis.py` | 做空IBEX35对冲分析（固定/滚动Beta） |
| `put_hedge_analysis.py` | Put期权Black-Scholes定价详细计算 |
| `PTOPZWHM0007_daily_*.csv` | 基金NAV历史数据 |
| `IBEX35_daily_2024-2026.csv` | IBEX35指数数据 |
| `开发日志.md` | 项目开发过程记录 |

## 运行

```bash
pip install yfinance pandas numpy plotly
python3 hedge_final.py
```

会自动下载最新IBEX数据，生成 `hedge_final.html` 并打开。
