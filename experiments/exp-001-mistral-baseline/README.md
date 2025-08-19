# Experiment report

**Folder:** `experiments/exp-001-mistral-baseline`

- **Model:** mistral
- **Prompt:** baseline
- **Rows:** 6
- **Unique videos:** 6

## Quick stats
- Label distribution: DEBUNKING:2, MISINFO:3, NO_MISINFO:1
- Avg confidence: **0.95**
- Latency (sec) — avg: **32.53**, min: 13.24, max: 88.53

## Plots
![Label Distribution](./label_distribution.png)
![Confidence Histogram](./confidence_hist.png)
![Latency Histogram](./latency_hist.png)
![Top Keywords](./keywords_top.png)

## Files
- `results.csv` — raw outputs
- `label_distribution.png` — label counts
- `confidence_hist.png` — confidence histogram
- `latency_hist.png` — latency histogram
- `keywords_top.csv` — top keyword counts
- `keywords_top.png` — top keyword bar chart
