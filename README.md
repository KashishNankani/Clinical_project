# Clinical ASR Project

An end-to-end Automatic Speech Recognition (ASR) pipeline for clinical audio, built using OpenAI Whisper. Transcribes doctor-patient conversations, evaluates accuracy using Word Error Rate (WER), and applies domain-specific auto-corrections.

## Results
- Average WER: 0.11

## Project Structure
```
clinical_asr_project/
├── data/
│   └── dataset_manifest.json
├── scripts/
│   ├── record_dataset.py
│   ├── transcribe_audio.py
│   ├── evaluate_wer.py
│   └── asr_pipeline.py
├── outputs/
│   ├── predictions/
│   ├── evaluations/
│   ├── alignments/
│   └── pipeline/
├── requirements.txt
└── README.md
```

## Setup
```bash
pip install -r requirements.txt
```

## Usage
Run the full pipeline with one command:
```bash
python scripts/asr_pipeline.py
```

## Pipeline Steps
1. **Transcribe** — Whisper transcribes audio files with segments
2. **Evaluate** — WER calculated, auto-corrections applied
3. **Output** — Clean JSON with raw + corrected transcripts and segments

## Tech Stack
- OpenAI Whisper (small model)
- jiwer (WER evaluation)
- NLTK (lemmatization)
- PyTorch
