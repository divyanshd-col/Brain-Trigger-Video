"""Phase 0 smoke test: run TRIBE v2 on one of your videos on a pay-per-use Modal GPU.

Usage (from your laptop):
    modal run tribe_modal.py --video path/to/video.mp4 --mode av

Modes:
    full   video + audio + text. Speech is transcribed as ENGLISH; needs Llama-3.2-3B access.
    av     video + audio, no transcript. Works for any language (Hindi/Hinglish).
    video  visual stream only (what the Square Zero Labs thumbnail notebook uses).

Writes to ./outputs/<video-name>-<mode>/:
    preds.npy          (seconds x 20484) predicted cortical response
    roi_per_second.csv per-second mean for each of the 180 HCP regions
    groups_per_second.csv  5 region groups + loudness/motion baselines, per second
    timeline.png       the groups and baselines plotted over time
    run_stats.json     GPU, peak VRAM, time per stage -> real cost per minute of video
"""

import hashlib
import io
import json
import time
from pathlib import Path

import modal

TRIBE_FORK = "git+https://github.com/Square-Zero-Labs/tribev2.git@71b17ffc61a9a28b24d41ced3507a7fdb09a7ead"
GPU = "L40S"  # 48 GB: safe first run. Drop to "L4" (24 GB) if run_stats shows peak VRAM < ~20 GB.
CACHE = Path("/cache")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "ffmpeg")
    .pip_install(
        f"tribev2 @ {TRIBE_FORK}",
        # Same pins the Square Zero Labs notebook uses to get a working install.
        "torch==2.6.0",
        "torchvision==0.21.0",
        "torchaudio==2.6.0",
        "transformers==5.17.0",
        "numpy==2.2.6",
        "scipy==1.15.3",
        "exca==0.5.20",
        "pandas>=2.2.3,<3",
        "matplotlib>=3.9,<4",
        "uv",  # TRIBE runs WhisperX through `uvx`
    )
    .env(
        {
            "HF_HOME": str(CACHE / "hf"),
            "UV_CACHE_DIR": str(CACHE / "uv"),
            "MNE_DATA": str(CACHE / "mne"),  # HCP atlas download, ~once
            "MPLBACKEND": "Agg",
        }
    )
)

app = modal.App("brain-trigger-video", image=image)
volume = modal.Volume.from_name("tribe-cache", create_if_missing=True)

# Hand-picked HCP region groups, copied from the Square Zero Labs thumbnail notebook.
# They are a convenience for comparison, not canonical neuroscience labels.
ROI_GROUPS = {
    "visual": {
        "V1", "V2", "V3", "V4", "V3A", "V3B", "V6", "V6A", "V7", "LO1", "LO2", "LO3",
        "MT", "MST", "FFC", "PIT", "PHA1", "PHA2", "PHA3", "VMV1", "VMV2", "VMV3", "VVC",
    },
    "auditory_speech": {
        "A1", "A4", "A5", "LBelt", "MBelt", "PBelt", "RI", "STGa", "TA2",
        "STSda", "STSdp", "STSva", "STSvp",
    },
    "language_semantic": {
        "44", "45", "47l", "p47r", "IFJa", "IFJp", "IFSa", "IFSp",
        "TGd", "TGv", "TE1a", "TE1p", "TE2a", "TE2p",
    },
    "attention_control": {
        "FEF", "PEF", "IP0", "IP1", "IP2", "LIPd", "LIPv", "AIP", "MIP", "VIP",
        "8Av", "8Ad", "8BL", "8C", "9-46d", "9-46v", "SCEF",
    },
    "motor_somato": {
        "1", "2", "3a", "3b", "4", "5m", "5mv", "5L", "5hv",
        "6a", "6d", "6ma", "6mp", "6r", "6v", "24dd", "24dv",
    },
}

MODE_FEATURES = {"full": None, "av": ["video", "audio"], "video": ["video"]}


def baselines_per_second(video_path: Path, n_seconds: int):
    """Loudness (audio RMS) and motion (mean frame difference) per second.

    The published null result (arXiv 2607.01400) found TRIBE no better than these
    two numbers at predicting replays, so every TRIBE signal is reported next to them.
    """
    import numpy as np
    from moviepy import VideoFileClip

    clip = VideoFileClip(str(video_path))
    loudness = np.full(n_seconds, np.nan)
    if clip.audio is not None:
        sr = 16000
        audio = clip.audio.to_soundarray(fps=sr)
        audio = audio.mean(axis=1) if audio.ndim == 2 else audio
        for s in range(n_seconds):
            chunk = audio[s * sr : (s + 1) * sr]
            if len(chunk):
                loudness[s] = float(np.sqrt(np.mean(chunk**2)))

    fps = 4
    diffs = [[] for _ in range(n_seconds)]
    prev = None
    for i, frame in enumerate(clip.iter_frames(fps=fps)):
        gray = frame.mean(axis=2)
        second = i // fps
        if prev is not None and second < n_seconds:
            diffs[second].append(float(np.abs(gray - prev).mean()))
        prev = gray
    motion = np.array([np.mean(d) if d else np.nan for d in diffs])
    clip.close()
    return loudness, motion


@app.function(
    gpu=GPU,
    volumes={str(CACHE): volume},
    secrets=[modal.Secret.from_name("huggingface")],
    timeout=60 * 60,
)
def analyse(video_bytes: bytes, filename: str, mode: str) -> dict:
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import torch
    from moviepy import VideoFileClip
    from tribev2.demo_utils import TribeModel, get_audio_and_text_events
    from tribev2.utils import get_hcp_labels

    stats = {"mode": mode, "gpu": torch.cuda.get_device_name(0), "seconds": {}}
    tick = time.time()

    def lap(name):
        nonlocal tick
        stats["seconds"][name] = round(time.time() - tick, 1)
        tick = time.time()

    # Content-addressed path so a re-run of the same video reuses cached features.
    digest = hashlib.sha256(video_bytes).hexdigest()[:16]
    video_path = CACHE / "inputs" / f"{digest}{Path(filename).suffix.lower()}"
    video_path.parent.mkdir(parents=True, exist_ok=True)
    video_path.write_bytes(video_bytes)
    with VideoFileClip(str(video_path)) as clip:
        duration = float(clip.duration)
    stats["video_seconds"] = round(duration, 1)

    torch.cuda.reset_peak_memory_stats()
    config_update = None
    if MODE_FEATURES[mode] is not None:
        config_update = {"data": {"features_to_use": MODE_FEATURES[mode]}}
    model = TribeModel.from_pretrained(
        "facebook/tribev2", cache_folder=CACHE / "features", config_update=config_update
    )
    lap("load_model")

    event = {
        "type": "Video",
        "filepath": str(video_path),
        "start": 0.0,
        "timeline": digest,
        "subject": "default",
    }
    if mode == "full":
        events = model.get_events_dataframe(video_path=video_path)
    elif mode == "av":
        events = get_audio_and_text_events(pd.DataFrame([event]), audio_only=True)
    else:
        events = pd.DataFrame([{**event, "duration": duration}])
    lap("build_events")

    preds, segments = model.predict(events=events)
    lap("predict")
    stats["peak_vram_gb_torch"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
    stats["note_vram"] = "WhisperX (full mode) runs in a subprocess and is not counted here."

    # Per-region means: one atlas load, then vectorised (tribev2's summarize_by_roi
    # reloads the atlas for every region, which is very slow per timestep).
    labels = get_hcp_labels(mesh="fsaverage5", combine=False, hemi="both")
    roi_df = pd.DataFrame({name: preds[:, idx].mean(axis=1) for name, idx in labels.items()})
    starts = [getattr(s, "start", np.nan) for s in segments]
    roi_df.insert(0, "start_sec", starts)

    groups_df = pd.DataFrame({"start_sec": starts})
    for group, members in ROI_GROUPS.items():
        groups_df[group] = roi_df[[c for c in roi_df.columns if c in members]].mean(axis=1)
    groups_df["whole_cortex_mean"] = preds.mean(axis=1)

    loudness, motion = baselines_per_second(video_path, int(np.ceil(duration)))
    sec_idx = pd.Series(starts).fillna(-1).astype(int).clip(lower=0, upper=len(loudness) - 1)
    groups_df["baseline_loudness"] = loudness[sec_idx.to_numpy()]
    groups_df["baseline_motion"] = motion[sec_idx.to_numpy()]
    lap("summarise")

    # z-score each line so they share one axis; this is for shape, not magnitude.
    fig, ax = plt.subplots(figsize=(12, 5))
    x = groups_df["start_sec"]
    for col in [*ROI_GROUPS, "baseline_loudness", "baseline_motion"]:
        y = groups_df[col]
        z = (y - y.mean()) / (y.std() or 1)
        ax.plot(x, z, label=col, linestyle="--" if col.startswith("baseline") else "-")
    ax.set_xlabel("video second")
    ax.set_ylabel("z-score within this video")
    ax.set_title(f"{filename} — mode={mode} (predicted average brain, not measured)")
    ax.legend(fontsize=8, ncol=4)
    ax.grid(alpha=0.2)
    png = io.BytesIO()
    fig.savefig(png, dpi=150, bbox_inches="tight")
    plt.close(fig)

    stats["total_gpu_seconds"] = round(sum(stats["seconds"].values()), 1)
    stats["gpu_seconds_per_video_minute"] = round(
        stats["total_gpu_seconds"] / max(duration / 60, 1e-6), 1
    )
    stats["timesteps_kept"] = int(preds.shape[0])
    volume.commit()

    npy = io.BytesIO()
    np.save(npy, preds)
    return {
        "preds.npy": npy.getvalue(),
        "roi_per_second.csv": roi_df.to_csv(index=False).encode(),
        "groups_per_second.csv": groups_df.to_csv(index=False).encode(),
        "timeline.png": png.getvalue(),
        "run_stats.json": json.dumps(stats, indent=2).encode(),
    }


@app.local_entrypoint()
def main(video: str, mode: str = "av"):
    if mode not in MODE_FEATURES:
        raise SystemExit(f"--mode must be one of {list(MODE_FEATURES)}")
    path = Path(video)
    out_dir = Path("outputs") / f"{path.stem}-{mode}"
    out_dir.mkdir(parents=True, exist_ok=True)
    results = analyse.remote(path.read_bytes(), path.name, mode)
    for name, data in results.items():
        (out_dir / name).write_bytes(data)
    print((out_dir / "run_stats.json").read_text())
    print(f"Saved outputs to {out_dir}/")
