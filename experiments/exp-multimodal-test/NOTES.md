# exp-multimodal-test — Multimodal Content Extraction Test

## Goal
Test complete multimodal pipeline combining:
- Audio transcript (Whisper/Faster-Whisper)
- On-screen text (EasyOCR)

## Configuration
- Model: Mistral (via Ollama)
- Prompt: baseline
- Temperature: 0.0
- Audio: Enabled (faster-whisper)
- Visual: Enabled (EasyOCR, 1 fps sampling)
- Languages: English

## Hypothesis
Combining audio + visual content will provide richer context for classification
compared to audio-only approach.

## Status
- [x] Config created
- [x] Video prepared (test1.mp4)
- [ ] Batch experiment run
- [ ] Results reviewed
- [ ] Comparison with audio-only results

## Expected Outcomes
- More comprehensive content extraction
- Potentially better classification accuracy
- Ability to detect text-based misinformation missed by audio

## Observations
(To be filled after run)
