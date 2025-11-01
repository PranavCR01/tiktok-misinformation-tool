# Experiment Templates

This folder contains templates for creating new experiments.

## Quick Start - Create New Experiment

```bash
# From project root
bash experiments/_templates/create_new_experiment.sh exp-YOUR-NAME

# Example:
bash experiments/_templates/create_new_experiment.sh exp-002-fewshot-mistral
```

This will create:
- `experiments/exp-YOUR-NAME/config.yaml` (pre-configured)
- `experiments/exp-YOUR-NAME/NOTES.md` (structured notes template)
- `experiments/exp-YOUR-NAME/command.txt` (exact command to run)

## Manual Setup

If you prefer to create experiments manually:

1. **Copy template:**
   ```bash
   cp experiments/_templates/config.template.yaml experiments/YOUR-EXP/config.yaml
   ```

2. **Edit configuration:**
   - Update `out:` path to match your experiment folder
   - Choose prompt type: `baseline`, `fewshot`, or `reasoned`
   - Set temperature (0.0 recommended for reproducibility)
   - Add descriptive notes

3. **Run experiment:**
   ```bash
   python scripts/run_mistral_batch.py --config experiments/YOUR-EXP/config.yaml
   ```

4. **Generate analysis:**
   ```bash
   python scripts/analyze_experiment.py --exp-dir experiments/YOUR-EXP
   ```

## Experiment Naming Convention

Use descriptive names that indicate what you're testing:

- `exp-001-mistral-baseline` - Baseline with Mistral
- `exp-002-mistral-fewshot` - Few-shot prompting
- `exp-003-llama-comparison` - Different model comparison
- `exp-004-multilingual-test` - Non-English content
- `exp-005-high-temp` - Temperature variation

## Configuration Options

### Prompt Types
- **baseline**: Standard classification (fastest, consistent)
- **fewshot**: Includes examples (better calibration)
- **reasoned**: Chain-of-thought (slowest, most detailed)

### Model Options (Ollama)
- `mistral` - 7B params, fast, good accuracy
- `llama2` - Meta's model, 7B-70B variants
- `gemma` - Google's efficient model
- `phi` - Microsoft's small but capable model

### Temperature Settings
- `0.0` - Deterministic (recommended for reproducibility)
- `0.1-0.3` - Slightly variable but consistent
- `0.5-0.7` - Balanced creativity
- `1.0+` - More creative (not recommended for research)

## Best Practices

1. **One variable at a time**: Only change one parameter per experiment
2. **Document everything**: Use NOTES.md to record observations
3. **Version your data**: Note which video samples you used
4. **Commit results**: Git commit each completed experiment
5. **Compare systematically**: Use same videos across experiments when testing prompts/models
