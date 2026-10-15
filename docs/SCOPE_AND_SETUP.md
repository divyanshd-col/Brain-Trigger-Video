# Brain-Trigger-Video — Scope, Requirements & Setup

Status: **scoping** (nothing built yet). Every claim below is from reading the
source code of `facebookresearch/tribev2` and `Square-Zero-Labs/tribev2` at
their current `main`, unless marked *unverified*.

---

## 1. What this actually is (one paragraph, no hype)

**TRIBE v2** (Meta FAIR, 2026) is a *brain-encoding* model: you give it a
video / audio / text stimulus and it predicts the fMRI response of an
**average** human brain, **once per second**, on ~20,484 cortical points
(fsaverage5 mesh). It does **not** measure attention, emotion, liking,
retention or virality. Any "engagement score" is a number *we* derive from
the predicted brain map; it is a hypothesis, not a fact.

The **Square Zero Labs** notebook is a fork of Meta's repo that:
1. Pulls 5 high-view + 5 low-view videos from a YouTube channel (`yt-dlp`)
2. Turns each **thumbnail** into a 6-second silent clip
3. Runs TRIBE in **video-only** mode (skips Llama → no gated model needed)
4. Averages predictions into 5 hand-picked ROI groups (visual, auditory/speech,
   language, attention/control, motor) using the HCP atlas
5. Correlates those scores with view counts (n = 10 — its own README says
   "descriptive, not causal")

## 2. The evidence check (read before building anything)

| Claim | Evidence | Implication |
|---|---|---|
| TRIBE predicts brain response to natural video well | Meta paper (arXiv 2605.04326), 2–3× over linear baselines | Model is real and good *at what it was trained for* |
| TRIBE-derived signal predicts YouTube engagement | **Null result.** arXiv 2607.01400: 48 videos, per-second signal vs "Most Replayed" heatmap, partial r = **+0.058** (95% CI −0.04 to 0.15), not better than a plain loudness/motion baseline | Do **not** treat output as an engagement predictor until validated on *our* data |
| Thumbnail → brain score predicts CTR/views | Not tested beyond n=10; static image as video is out-of-distribution | Exploratory only |

**Consequence for the plan:** the highest-value step is not building a UI; it
is a cheap validation against our own retention data (Phase 2). If no signal
→ stop and save the build cost.

## 3. Hard requirements

### 3.1 Accounts & access
| Item | Why | Needed for |
|---|---|---|
| Hugging Face account + access token (read) | Download `facebook/tribev2` weights (~1 GB) + feature extractors | Everything |
| Approved access to `meta-llama/Llama-3.2-3B` on HF (gated; approval usually quick but not instant) | Text/language features from speech transcript | Full mode only. **Video-only mode skips this** |
| Google account with Colab (Pro recommended) **or** a cloud GPU (Modal / RunPod / Lambda / GCP) | Inference needs a CUDA GPU | Everything |

### 3.2 Compute
- **GPU: required.** The code falls back to CPU, but it runs V-JEPA2 ViT-g,
  DINOv2-large, Wav2Vec-BERT, Llama-3.2-3B and WhisperX large-v3, so CPU is not
  practical.
- VRAM: *unverified*. A 16 GB T4 may be tight for full mode; a 24 GB L4 or
  40 GB A100 is the safe choice. **Phase 0 measures this** instead of guessing.
- Disk: ~25–40 GB for model weights + feature cache (*unverified estimate*).
- Python ≥ 3.11, `torch>=2.5.1,<2.7`, `numpy==2.2.6`, `ffmpeg`, `uv` (WhisperX
  is run through `uvx`).

### 3.2a Compute decision: pay-per-use → **Modal** (decided)

| | Modal (chosen) | RunPod pod | Colab Pro |
|---|---|---|---|
| Billing | Per second, **only while a video is processing**; scales to zero | Per hour **while the pod is on**, even when idle | Monthly units |
| Free allowance | $30/month credit on the Starter plan | None | — |
| Price (Oct 2026, verify at checkout) | L40S ≈ $1.95/h, L4 ≈ $0.80/h, A100-80GB ≈ $2.10–2.50/h, plus a small CPU/RAM charge | L4/A40 ≈ $0.35–0.45/h, A100 ≈ $1.19–1.39/h | — |
| Reuse for Phase 3 web app | **Same function becomes the API**; no rewrite | Must rebuild as a service | Not deployable |
| Failure mode | Cold start reloads weights (mitigated: `tribe-cache` Volume) | Forgetting to stop the pod burns money | Session dies, dependency resets |

Why Modal wins on effort vs. output:
- The $30 credit likely covers all of Phase 0–2: about 15 L40S hours.
- No idle-billing risk.
- The Phase 0 code is already the Phase 3 backend.

RunPod is cheaper per hour but only pays off at sustained, high volume, which we
don't have evidence for yet.

GPU choice: start on **L40S (48 GB)** so the first run can't fail on memory.
`run_stats.json` records peak VRAM. If it is under ~20 GB, switch `GPU = "L4"`
in `tribe_modal.py` and cut cost by about 60%.

**Cost per video = `gpu_seconds_per_video_minute` (measured in Phase 0) × video
minutes × GPU $/s.** That number isn't known yet. Phase 0 exists to measure it.

Data note: uploaded videos are processed on Modal's cloud and cached in the
`tribe-cache` volume. For unreleased marketing content, check that this is
acceptable under company policy. Delete cached inputs with
`modal volume rm tribe-cache inputs -r`.

### 3.3 Input constraints (from code)
- Video formats: `.mp4 .avi .mkv .mov .webm`; audio: `.wav .mp3 .flac .ogg`.
- Output resolution: **1 prediction per second**. A 30 s reel gives 30 rows,
  not frame-level detail.
- Predictions are **shifted 5 s back** for hemodynamic lag. The first ~5 s
  of the output need careful reading, and the hook is in those 5 s.
- Seconds with no events are dropped by default (`remove_empty_segments=True`).
- **Language:** speech transcription (`eventstransforms.py`) supports only
  English, French, Spanish, Dutch and Chinese, and **defaults to English**.
  **Hindi and Hinglish are not supported.** For non-English videos, either:
  (a) run **video + audio only** (drop text features), or
  (b) patch the WhisperX call to `--language hi`. That is untested, and the
  model was trained mostly on English stimuli, so its predictions would be
  out-of-distribution.
- Long videos are chunked into 30–60 s pieces internally. Runtime grows
  roughly linearly with length (*unverified*).

### 3.4 Licence — **blocker to clear first**
TRIBE v2 code and weights are **CC BY-NC 4.0 (non-commercial)**. Using it to
optimise a for-profit company's marketing or content is very likely
"commercial use". **Get a legal/compliance sign-off before anything leaves
personal R&D.** Llama 3.2 has its own licence (the Llama Community License),
which also needs to be accepted.

## 4. Phased plan (effort vs. output, stop-gates included)

| Phase | What | Effort | Output | Gate to continue |
|---|---|---|---|---|
| **0. Smoke test** | Run Meta's Colab on **one** of our videos, unchanged. Record GPU type, VRAM peak, minutes per minute of video | ~2 h | Proof it runs + real cost numbers | It runs on the GPU we can afford |
| **1. "Upload & see" notebook** | Fork the notebook: upload a video → per-second timeline for the 5 ROI groups + brain snapshots at peaks/dips + CSV | ~1 day | Usable personal tool | — |
| **2. Validate (most important)** | Take 20–40 of **our** published videos that have retention curves (YouTube Analytics audience retention, Meta/IG hold-rate, ThruPlay). Correlate per-second ROI signals with retention drops, controlling for position-in-video (the step the null-result paper says matters) | ~2–3 days | A yes/no answer: does this predict *our* audience? | Partial correlation clearly > loudness/motion baseline |
| **3. Team web app** | Gradio/Streamlit front-end on a serverless GPU (e.g. Modal); upload → report; A/B mode comparing two cuts/hooks/thumbnails | ~1 week | Shareable internal tool | Only if Phase 2 passes **and** licence cleared |

**Where the leverage is:** use the tool for **relative A/B comparison**
(hook A vs. hook B, thumbnail A vs. B, same video with and without captions).
Don't use it for absolute scores. Comparing two inputs under the same model
cancels much of the model's bias, so it is the use the evidence can best
support.

## 5. Setup — Phase 0 on Modal (recommended, ~30 min)

Script: [`tribe_modal.py`](../tribe_modal.py). It was import-checked and linted
but **has not run on a GPU yet**. The first run is the test.

1. Hugging Face: create a **read** token. For `--mode full`, also request
   access at `https://huggingface.co/meta-llama/Llama-3.2-3B`.
2. Modal: sign up at modal.com, then on your laptop:
   ```bash
   pip install modal
   modal setup                                         # browser login
   modal secret create huggingface HF_TOKEN=hf_xxx     # your HF token
   ```
3. Run on one short video (30–90 s) first:
   ```bash
   modal run tribe_modal.py --video my_video.mp4 --mode av
   ```
   - `--mode av` = video + audio, any language, no Llama needed (**start here**)
   - `--mode full` = adds the English transcript and Llama text features
   - `--mode video` = visuals only (thumbnail-style)
4. The first run is slow because it builds the image and downloads weights and
   the HCP atlas into the volume. **Run the same command a second time.** Use
   that second `run_stats.json` for cost numbers, since the first includes
   one-off downloads.
5. Outputs land in `outputs/<video>-<mode>/`: `timeline.png`,
   `groups_per_second.csv`, `roi_per_second.csv`, `preds.npy`,
   `run_stats.json`.

What to look at: in `timeline.png`, do the brain-region lines differ from the
dashed loudness/motion baselines? If they only mirror the baselines, the model
adds nothing for this video.

## 5b. Fallback — Phase 0 on Colab (step by step)

1. Create a HF account → Settings → Access Tokens → create a **read** token.
2. Open `https://huggingface.co/meta-llama/Llama-3.2-3B` → accept the licence
   → wait for "access granted" (skip if going video-only).
3. Open Meta's notebook in Colab:
   `https://colab.research.google.com/github/facebookresearch/tribev2/blob/main/tribe_demo.ipynb`
   (or the Square Zero Labs fork, whose dependency pins fix Colab breakages:
   `https://colab.research.google.com/github/Square-Zero-Labs/tribev2/blob/main/tribe_demo.ipynb`).
4. Runtime → Change runtime type → **GPU** (pick L4/A100 if available).
5. Run the install cell → **Runtime → Restart session** → continue below it.
6. Log in to HF in a new cell:
   ```python
   from huggingface_hub import login; login()   # paste token
   ```
7. Replace the sample video with your own:
   ```python
   from google.colab import files
   uploaded = files.upload()                # pick your .mp4; saved to cwd
   video_path = next(iter(uploaded))
   df = model.get_events_dataframe(video_path=video_path)
   preds, segments = model.predict(events=df)
   ```
   For video-only (no Llama, any language), load the model with
   `config_update={"data": {"features_to_use": ["video"]}}` (this is what the
   Square Zero Labs notebook does). Then build a `Video` event directly
   instead of calling `get_events_dataframe`, which would run transcription.
8. Note: GPU name (`!nvidia-smi`), peak VRAM
   (`torch.cuda.max_memory_allocated()/1e9`), wall-clock time, and video length.

## 6. Open questions (answers change the build)

1. **Use case:** thumbnails, short-form reels/ads (<60 s), or long-form
   YouTube? Phase 1 is a different tool for each.
2. **Language** of the voice-over: English, Hindi, Hinglish?
3. **Decision it should drive:** pick between variants before publishing, or
   diagnose after publishing? Pre-publish favours A/B mode; post-publish
   favours a timeline that sits next to the retention curve.
4. **Ground truth we own:** do we have per-second retention data (YouTube
   Studio / Meta Ads export) for ≥20 past videos? Without it, Phase 2 cannot
   happen.
5. ~~Compute budget~~ → **Decided: pay-per-use, Modal** (see 3.2a).
6. **Users:** just you, or a team? This decides whether Phase 3 is needed.
7. **Licence:** has legal seen CC BY-NC 4.0 for this use?

## Sources
- Meta TRIBE v2 code: https://github.com/facebookresearch/tribev2
- Weights: https://huggingface.co/facebook/tribev2
- Demo: https://aidemos.atmeta.com/tribev2
- Paper: https://arxiv.org/abs/2605.04326
- Square Zero Labs fork + thumbnail notebook: https://github.com/Square-Zero-Labs/tribev2
- Null-result engagement study: https://arxiv.org/abs/2607.01400
