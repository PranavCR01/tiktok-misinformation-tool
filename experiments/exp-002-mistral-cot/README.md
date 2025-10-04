# Experiment Report - Chain of Thought (COT)

**Folder:** `experiments/exp-002-mistral-cot`

- **Model:** mistral
- **Prompt:** cot (Chain of Thought)
- **Temperature:** 0.0
- **Goal:** Test if structured reasoning improves misinformation detection accuracy

## Experiment Hypothesis
Chain of Thought prompting will improve classification accuracy and consistency by forcing the model to:
1. Analyze content systematically
2. Evaluate source credibility 
3. Verify factual claims
4. Consider context appropriately
5. Provide reasoned final classification

## Quick Stats (To be filled after run)
- **Rows:** TBD
- **Unique videos:** TBD
- **Label distribution:** TBD
- **Avg confidence:** TBD
- **Latency (sec) — avg:** TBD, min: TBD, max: TBD

## Comparison with Baseline (exp-001)
- **Accuracy changes:** TBD
- **Confidence calibration:** TBD
- **Latency impact:** TBD
- **Keyword quality:** TBD

## COT Prompt Structure
The Chain of Thought prompt guides the model through 5 reasoning steps:
1. **Content Analysis** - Understanding main topics and claims
2. **Source Evaluation** - Assessing credibility indicators  
3. **Claim Verification** - Identifying verifiable facts
4. **Context Assessment** - Determining intent (spreading vs. debunking)
5. **Final Reasoning** - Systematic classification decision

## Plots (Generated after run)
- Label Distribution: `./label_distribution.png`
- Confidence Histogram: `./confidence_hist.png` 
- Latency Histogram: `./latency_hist.png`
- Top Keywords: `./keywords_top.png`

## Files
- `results.csv` — raw model outputs
- `config.yaml` — experiment configuration
- `NOTES.md` — detailed experiment notes
- `command.txt` — run command for reproducibility