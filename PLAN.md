# PLAN — finishing zero-label-lab

Legend: `[x]` done, `[ ]` to do, `⛔` needs the owner's explicit OK (paid Meta API call or action only the owner can take).

## Phase 0 — environment and hygiene
- [ ] 0.1 Convert `requirements.txt` from UTF-16 to UTF-8, remove the duplicate `imageio-ffmpeg` line, pin it
- [ ] 0.2 Translate French docstrings, comments and CLI messages in `zerolabel/` to English (behaviour unchanged)
- [ ] 0.3 Check that torch / transformers / peft / trl / scikit-learn install under Python 3.15 (dry-run resolves; real install in Phase 4), document the result
- [ ] 0.4 `pytest` tests (no API call): `sam_parse`, `evaluate.match/score`, `report_agent.check`, `measure`
- [ ] 0.5 Single reproducible entry point `python -m zerolabel.reproduce` that rebuilds every result from cached files (never calls the API)
- [ ] 0.6 Push phase 0

## Phase 1 — report agent
- [ ] 1.1 Style rules in the prompt (plain sentences, no raw field names, 2 decimals, no number outside the measurements, size-bias sentence on WARNING)
- [ ] 1.2 Stronger verifier: raw field names, glued words (e.g. `withoutdetection`), mandatory size-bias sentence when WARNING; tests
- [ ] 1.3 ⛔ Re-run Spark on the 5 levels (command + cost estimate, wait for OK)
- [ ] 1.4 Push phase 1

## Phase 2 — solid evaluation
- [ ] 2.1 Recall per radius bin from ground truth (SAM and OpenCV) + figure
- [ ] 2.2 Matching-threshold sensitivity (4, 8, 12 px)
- [ ] 2.3 Failure gallery: frames with missed drops highlighted
- [ ] 2.4 Generate 5 levels × 4 seeds locally, run OpenCV on all of them
- [ ] 2.5 ⛔ SAM on the new videos (cost estimate, cache every raw response, never pay twice)
- [ ] 2.6 Correlation estimated coverage vs true recall (Pearson, Spearman, bootstrap CI) + figure
- [ ] 2.7 ⛔ SAM variability: 2–3 identical calls on one video
- [ ] 2.8 Push phase 2

## Phase 3 — real droplet videos
- [ ] 3.1 Check licence and content of Zenodo 14731289 and GitHub `JoanaPNeto/Droplets`
- [ ] 3.2 Download only 1–2 small files, run OpenCV guard + `measure.py`
- [ ] 3.3 ⛔ SAM on the real videos
- [ ] 3.4 Compare with manual measurements if available
- [ ] 3.5 Feasibility note: open SAM 3 weights (licence, hardware) — proposal only
- [ ] 3.6 Push phase 3

## Phase 4 — small report-writing LLM
- [ ] 4.1 Check model licences (Qwen2.5-0.5B-Instruct, Llama 3.2 1B), pick one
- [ ] 4.2 Dataset: measurements (real + synthetic, perturbed) → deterministic template reports (never Spark text); split by video
- [ ] 4.3 Verifier-based reward function + tests
- [ ] 4.4 SFT (LoRA) + GRPO (TRL) training script
- [ ] 4.5 Evaluation script: VERIFIED rate, invented-number rate, WARNING accuracy, length (base / SFT / SFT+GRPO / Spark if allowed)
- [ ] 4.6 Tiny CPU smoke run locally
- [ ] 4.7 Colab/Kaggle notebook ready to run; ⛔ the owner runs it on a GPU and sends back the results
- [ ] 4.8 Push phase 4

## Phase 5 — showcase
- [ ] 5.1 Final README (problem, method, pipeline diagram, results, figures in `assets/`, LLM, limits, responsible AI, Meta tools, reproduction, licences)
- [ ] 5.2 Final `requirements.txt`, code clean-up, check that no secret is in the Git history
- [ ] 5.3 Demo video plan (2–3 min) + 3 CV lines FR/EN
- [ ] 5.4 Push phase 5

## Findings from reading the repo (2026-11-11)
- Repo is at `Desktop\Projects\personal\zero-label-lab` (not `Desktop\Projects\zero-label-lab`).
- Working tree clean: no uncommitted README; `README.md` only contains the title.
- `requirements.txt` is UTF-16 encoded and lists `imageio-ffmpeg` twice.
- Code comments and CLI messages are partly in French.
- Benchmark numbers in `outputs/benchmark.md` match the owner's notes.
- Only one SAM raw response per level is cached: the earlier level-3 run (recall 0.410) is not on disk.
- Spark reports exist for levels 1 and 3 only; the typo `withoutdetection` is not in these cached reports.
- SAM usage in raw responses reports `video_frames_processed: 150` and 0 tokens; the per-frame price is not in the repo.
- No secret found in the Git history; `.env` was never committed.
