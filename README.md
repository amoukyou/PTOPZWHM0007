# 葡萄牙基金对冲分析

**Optimize Portugal Golden Opportunities Fund**（ISIN: PTOPZWHM0007）的下行风险对冲方案。

## 在线查看

**[GitHub Pages 报告](https://amoukyou.github.io/PTOPZWHM0007/)**

> 备用：下载 `hedge_final.html` 到本地用浏览器打开。报告每个工作日 UTC 18:30 自动更新（GitHub Actions）。

## 背景

- 持仓：€507,000 买入，当前市值约 €644,000（+27%）
- 买入日：2024年7月16日，计划持有5年
- 担忧：俄乌停战、欧盟加息、美欧关税等全欧系统性风险导致基金大跌
- PSI20 无可用期权 → 用 IBEX35 Put（MEFF Mini 合约）作为替代对冲工具

## 两个方案（页面顶部可切换）

| | 方案A：混合 ATM+OTM | 方案B：纯OTM省钱版 |
|---|---|---|
| 合约配置 | ATM Put ×8 + 90%OTM Put ×20 | 90%OTM Put ×24 |
| 年保费（BS理论值） | ~€16,700（2.6%） | ~€10,000（1.6%） |
| 小跌保护（IBEX -5%） | 有（8张ATM兜底） | 无 |
| 大跌赔付（IBEX -30%） | ~€111,000 | ~€83,000 |
| 适合 | 任何级别下跌都想有保护 | 只防崩盘，接受小跌裸奔 |

> 注意：实际成本受 IV skew 影响，OTM Put 真实价格可能高于 BS 理论值 30-50%。动态滚仓（IBEX涨超10%即触发）会产生额外成本。

## 报告内容

1. **持仓概况** — 基金NAV、PSI20、IBEX35 实时走势（自动抓取）
2. **历史急跌分析** — 11次急跌事件逐一回测，含 IBEX 同步性验证
3. **对冲链路与Beta** — 含条件Beta实证分析（急跌时敏感度比全样本高~25%）
4. **混合行权价策略** — 4种配置对比（纯ATM/混合/进取/纯OTM）
5. **推荐方案** — A/B切换，含PSI20场景表和损益图
6. **操作步骤** — IBKR下单指引，含 skew 和滚仓成本警告
7. **局限性** — Event #10/11 教训、R²=42%、条件Beta、滚仓隐性成本

## 文件说明

| 文件 | 说明 |
|------|------|
| `hedge_final.py` | **主文件（v19）** — 生成完整报告，含实时数据抓取和交互图表 |
| `hedge_final.html` | 生成的报告网页 |
| `docs/index.html` | GitHub Pages 副本（自动同步） |
| `.github/workflows/update-report.yml` | 每日自动更新（工作日 UTC 18:30） |
| `PTOPZWHM0007_daily_*.csv` | 基金NAV历史数据 |

## 运行

```bash
pip install yfinance pandas numpy plotly
python3 hedge_final.py
```

自动抓取最新基金NAV（FT Markets）和指数数据（Yahoo Finance），生成 `hedge_final.html` 并打开。

---

## English Summary

Downside hedge analysis for **Optimize Portugal Golden Opportunities Fund** (ISIN: PTOPZWHM0007) — a Portuguese equity fund with €507K initial investment, currently valued at ~€644K.

**The problem**: No options available on the PSI20 index (which the fund tracks closely, R²=79%). We use **IBEX35 Put options** (MEFF Mini contracts, tradable via IBKR) as a cross-hedge, leveraging the Beta relationship between PSI20 and IBEX35.

**Two plans** (toggle in the report header):
- **Plan A** (Mixed ATM+OTM): ATM×8 + 90%OTM×20 — covers both small and large drops, ~€16,700/yr
- **Plan B** (Pure OTM): 90%OTM×24 — crash-only protection, ~€10,000/yr

**Key findings**: All 11 historical fund crashes (>3% weekly) saw IBEX decline in sync. Conditional Beta during crashes (~0.53) is ~25% higher than full-sample Beta (0.423). Split-leg rolling strategy saves ~€6,700/yr in rolling costs.

The report page supports **Chinese/English toggle** — click the "EN" button in the top navigation bar.

**[View Live Report](https://amoukyou.github.io/PTOPZWHM0007/)**
