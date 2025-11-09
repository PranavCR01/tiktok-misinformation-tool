# Gemma Experiments

This folder contains experiments using the Gemma model for misinformation detection.

## Available Configurations

1. Baseline (`exp-003-gemma-baseline/`)
   - Basic fact-checking prompt
   - No examples or special instructions

2. Few-shot (`exp-003-gemma-fewshot/`)
   - Includes example cases to guide the model
   - May provide more consistent results

3. Reasoned (`exp-003-gemma-reasoned/`)
   - Includes step-by-step thinking instruction
   - May provide more thorough analysis

## Running the Experiments

First, make sure you have Gemma available in Ollama:
```bash
ollama pull gemma
```

Then run each experiment:
```bash
# Baseline approach
python3 scripts/run_mistral_batch.py --config experiments/exp-003-gemma-baseline/config.yaml

# Few-shot approach
python3 scripts/run_mistral_batch.py --config experiments/exp-003-gemma-fewshot/config.yaml

# Reasoned approach
python3 scripts/run_mistral_batch.py --config experiments/exp-003-gemma-reasoned/config.yaml
```

## Analyzing Results

After running the experiments, analyze the results:
```bash
python3 scripts/analyze_experiment.py experiments/exp-003-gemma-{baseline,fewshot,reasoned}/config.yaml
```