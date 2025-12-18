import os, json, math, urllib.request
from pathlib import Path
import numpy as np
import cv2
import torch
import imageio.v2 as imageio
from cotracker.predictor import CoTrackerPredictor
from huggingface_hub import hf_hub_download

OUT_DIR = Path("dataset_drag2live")
CKPT_DIR = Path("checkpoints")
CKPT_PATH = CKPT_DIR / "cotracker2.pth"

RAW_DIR = Path("raw_cloud_video")

RES = 256
TARGET_FPS = 8

T = 25
STRIDE = 16
GRID_SIZE = 16

TOPK_LIST = [32]
MAKE_FLIP = False

MAX_CLIPS_PER_VIDEO = 40


CKPT_URL = "https://dl.fbaipublicfiles.com/cotracker/cotracker2.pth"

def ensure_ckpt():
    return hf_hub_download(repo_id="facebook/cotracker", filename="cotracker2.pth")


def read_and_resample(video_path):
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or math.isnan(fps):
        fps = 30.0
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (RES, RES), interpolation=cv2.INTER_AREA)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    cap.release()
    if len(frames) == 0:
        return None
    step = max(1, int(round(fps / TARGET_FPS)))
    frames = frames[::step]
    return frames

def make_clips(frames):
    clips = []
    n = len(frames)
    starts = list(range(0, n - T + 1, STRIDE))
    if len(starts) == 0:
        return clips
    if MAX_CLIPS_PER_VIDEO is not None and len(starts) > MAX_CLIPS_PER_VIDEO:
        idx = np.linspace(0, len(starts) - 1, MAX_CLIPS_PER_VIDEO)
        idx = np.round(idx).astype(int).tolist()
        starts = [starts[i] for i in idx]
    for s in starts:
        clips.append(frames[s:s + T])
    return clips


def to_tensor(frames):
    x = np.stack(frames, axis=0)
    x = torch.from_numpy(x).float() / 255.0
    x = x.permute(0, 3, 1, 2).unsqueeze(0)
    return x

def motion_scores(tracks, vis):
    t = tracks.shape[0]
    n = tracks.shape[1]
    v = vis.astype(np.float32)
    p = tracks.astype(np.float32)
    dp = p[1:] - p[:-1]
    dv = v[1:] * v[:-1]
    speed = np.linalg.norm(dp, axis=-1) * dv
    score = speed.sum(axis=0) / (dv.sum(axis=0) + 1e-6)
    score[np.isnan(score)] = 0.0
    return score

def select_topk(tracks, vis, k):
    scores = motion_scores(tracks, vis)
    idx = np.argsort(-scores)[:k]
    return tracks[:, idx, :], vis[:, idx], idx

def save_mp4(path, frames):
    imageio.mimsave(str(path), frames, fps=TARGET_FPS)

def flip_h(frames, tracks):
    f2 = [np.ascontiguousarray(f[:, ::-1, :]) for f in frames]
    tr = tracks.copy()
    tr[..., 0] = (RES - 1) - tr[..., 0]
    return f2, tr

def main():
    ckpt = ensure_ckpt()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cotracker = torch.hub.load("facebookresearch/co-tracker", "cotracker3_offline").to(device).eval()

    (OUT_DIR / "videos").mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "tracks").mkdir(parents=True, exist_ok=True)

    index_path = OUT_DIR / "index.jsonl"
    if index_path.exists():
        index_path.unlink()

    meta = {
        "resolution": [RES, RES],
        "fps": TARGET_FPS,
        "frames_per_clip": T,
        "stride": STRIDE,
        "grid_size": GRID_SIZE,
        "topk_list": TOPK_LIST,
        "flip": MAKE_FLIP
    }

    sample_id = 0
    for vp in sorted(RAW_DIR.glob("*.mp4")):
        frames = read_and_resample(vp)
        if frames is None or len(frames) < T:
            continue
        clips = make_clips(frames)
        for ci, clip in enumerate(clips):
            vid_base = f"{vp.stem}_c{ci:03d}"
            video_out = OUT_DIR / "videos" / f"{vid_base}.mp4"
            save_mp4(video_out, clip)

            x = to_tensor(clip).to(device)
            with torch.no_grad():
                pred_tracks, pred_visibility = cotracker(x, grid_size=GRID_SIZE)

            tracks = pred_tracks.squeeze(0).detach().cpu().numpy()
            vis = pred_visibility.squeeze(0).squeeze(-1).detach().cpu().numpy().astype(np.uint8)

            tracks_path = OUT_DIR / "tracks" / f"{vid_base}.npz"
            np.savez_compressed(
                tracks_path,
                tracks=tracks.astype(np.float16),
                vis=vis,
                res=np.array([RES, RES], dtype=np.int32),
                fps=np.int32(TARGET_FPS),
                T=np.int32(T)
            )

            for k in TOPK_LIST:
                topk_tr, topk_vis, topk_idx = select_topk(tracks, vis, k)
                rec = {
                    "id": sample_id,
                    "video": str(video_out.as_posix()),
                    "tracks_npz": str(tracks_path.as_posix()),
                    "topk_k": int(k),
                    "topk_idx": topk_idx.tolist()
                }
                with open(index_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                sample_id += 1

            if MAKE_FLIP:
                clip_f, tracks_f = flip_h(clip, tracks)
                vid_base_f = f"{vid_base}_flip"
                video_out_f = OUT_DIR / "videos" / f"{vid_base_f}.mp4"
                save_mp4(video_out_f, clip_f)

                tracks_path_f = OUT_DIR / "tracks" / f"{vid_base_f}.npz"
                np.savez_compressed(
                    tracks_path_f,
                    tracks=tracks_f.astype(np.float16),
                    vis=vis,
                    res=np.array([RES, RES], dtype=np.int32),
                    fps=np.int32(TARGET_FPS),
                    T=np.int32(T)
                )

                for k in TOPK_LIST:
                    topk_tr, topk_vis, topk_idx = select_topk(tracks_f, vis, k)
                    rec = {
                        "id": sample_id,
                        "video": str(video_out_f.as_posix()),
                        "tracks_npz": str(tracks_path_f.as_posix()),
                        "topk_k": int(k),
                        "topk_idx": topk_idx.tolist()
                    }
                    with open(index_path, "a", encoding="utf-8") as f:
                        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    sample_id += 1

    with open(OUT_DIR / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"done. samples={sample_id} index={index_path}")

if __name__ == "__main__":
    main()
