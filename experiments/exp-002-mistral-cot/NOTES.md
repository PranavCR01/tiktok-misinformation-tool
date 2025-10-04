# exp-002 — mistral chain of thought
- goal: test Chain of Thought (COT) prompting for improved reasoning in misinformation detection
- prompt: cot, temp=0.0
- ASR: faster-whisper (CPU) if available, else local whisper
- hypothesis: COT prompting will improve accuracy by forcing the model to reason through its classification step-by-step
- observations (fill after run):
  - label distribution:
  - avg latency:
  - avg confidence:
  - comparison with baseline (exp-001):
  - issues:

## COT Prompt Strategy
The Chain of Thought prompt guides the model through:
1. Content analysis (what is being said?)
2. Source evaluation (who is saying it? any credentials mentioned?)
3. Claim verification (are there verifiable facts being made?)
4. Context assessment (is this correcting misinformation or spreading it?)
5. Final classification with reasoning

Expected benefits:
- More consistent classifications
- Better handling of edge cases
- Improved confidence calibration
- More informative keywords extraction