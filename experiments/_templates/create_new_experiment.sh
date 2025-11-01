#!/bin/bash
# Quick script to create a new experiment folder

if [ -z "$1" ]; then
    echo "Usage: ./create_new_experiment.sh <experiment-name>"
    echo "Example: ./create_new_experiment.sh exp-002-fewshot-test"
    exit 1
fi

EXP_NAME=$1
EXP_DIR="experiments/$EXP_NAME"

# Check if experiment already exists
if [ -d "$EXP_DIR" ]; then
    echo "Error: Experiment '$EXP_NAME' already exists!"
    exit 1
fi

echo "Creating new experiment: $EXP_NAME"

# Create directory
mkdir -p "$EXP_DIR"

# Copy config template
cp experiments/_templates/config.template.yaml "$EXP_DIR/config.yaml"

# Update output path in config
sed -i '' "s|YOUR-EXP-NAME|$EXP_NAME|g" "$EXP_DIR/config.yaml"

# Create NOTES.md
cat > "$EXP_DIR/NOTES.md" << NOTES
# $EXP_NAME

## Goal
(Describe what you're testing in this experiment)

## Configuration
- Model: (e.g., mistral, llama2)
- Prompt: (baseline | fewshot | reasoned)
- Temperature: (0.0 - 1.0)
- Dataset: (describe your video samples)

## Hypothesis
(What do you expect to happen?)

## Status
- [ ] Config customized
- [ ] Videos prepared
- [ ] Batch experiment run
- [ ] Analysis generated
- [ ] Results reviewed

## Observations
(Fill after running experiment)

### Quantitative Results
- Label distribution:
- Avg confidence:
- Avg latency:

### Qualitative Findings
(Any interesting patterns, errors, or insights)

## Next Steps
(What to try next based on these results)
NOTES

# Create command.txt
echo "python scripts/run_mistral_batch.py --config experiments/$EXP_NAME/config.yaml" > "$EXP_DIR/command.txt"

echo "✓ Experiment folder created: $EXP_DIR"
echo ""
echo "Next steps:"
echo "1. Edit $EXP_DIR/config.yaml with your settings"
echo "2. Add video files to data/videos/"
echo "3. Run: bash $EXP_DIR/command.txt"
echo "4. Analyze: python scripts/analyze_experiment.py --exp-dir $EXP_DIR"
