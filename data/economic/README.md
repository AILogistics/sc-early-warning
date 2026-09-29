# 港口吞吐量数据录入指南

**目标**:为 3 个港口填充月度集装箱吞吐量(TEU)数据,替换合成代理,用于 PAI 相关性验证(论文 Section 5.3)。

## 文件位置

```
data/economic/
├── rotterdam_throughput_monthly.csv
├── los_angeles_throughput_monthly.csv
└── shanghai_throughput_monthly.csv
```

每文件格式:`month,throughput_teu` —— 填入月度 TEU 数字即可。

## 数据源(按可得性排序)

### Rotterdam(最容易,英文)
- **官方**:Port of Rotterdam Authority 年度报告
  - https://www.portofrotterdam.com/en/the-port/port-facts-and-figures
  - 月度数据在 "Container throughput" PDF 里
- **备选**:Eurostat 海运统计(免费)
  - https://ec.europa.eu/eurostat/web/maritime-transport

### Los Angeles / Long Beach(最容易,英文)
- **Port of LA**: https://www.portoflosangeles.org/business/statistics/container-statistics
- **Port of LB**: https://www.polb.com/business/statistics/monthly-container-statistics
- 直接网页表格,可复制粘贴

### Shanghai(中等,中文)
- **上海国际港务集团**: https://www.sipg.com.cn(中文,需翻译/确认)
- **备选**:UNCTAD 海运统计年报(年度,不是月度)

## 最少填多少?

**推荐**:至少 24 个月(2022-01 到 2023-12),覆盖 Shanghai E2/E5 + Rotterdam E6 事件窗口。

**最低**:12 个月(2022-01 到 2022-12),Shanghai E2 封控期间。

## 填好后

CSV 填好后,告诉我,我会:
1. 重跑 PAI 相关性分析(用真实数据)
2. 重绘 Figure 13(PAI-Throughput 散点回归图)
3. 更新论文 Section 4.4 和 5.3 的 [TODO]
4. 重算核心标准 [3] 的真实 r 值

## 时间预估

每港口约 15-30 分钟(取决于数据源),3 港口共约 1 小时。
