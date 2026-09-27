"""Per-frame facial geometry from EngageNet video, via MediaPipe FaceLandmarker.

WHY GEOMETRY RATHER THAN PIXELS

EngageNet's lowest class is defined behaviourally - a subject who "frequently glances away from the
screen". That is a claim about where someone is looking over time, not about how their face looks in
any one frame. Eleven scalars per frame express it directly, at roughly 600 bytes per serving cycle
against ~3.3 MB of face crops, with no pixels leaving the browser.

WHAT COMES OUT, AND FROM WHERE

    yaw, pitch, roll   Euler angles decomposed from the 4x4 facial transformation matrix
    ear_l, ear_r       eye aspect ratio per eye; ear_mean their average
    mouth_open         lip separation normalised by mouth width, so it is scale-free
    gaze_x, gaze_y     iris centre offset from the eye-corner midpoint, normalised by eye width
    motion             mean inter-frame landmark displacement - a proxy for fidgeting
    face_found         1.0 when a face was detected in that frame, else 0.0

`face_found` is emitted deliberately even though it is not a facial measurement. On Validation it
correlates -0.43 with the label: clips where no face is ever found are 100% low-engagement against
16% for fully-detected clips. That is partly real (turning fully away defeats the detector, which is
itself disengagement) and partly a corpus property. Recording it lets the analysis quantify the
confound instead of absorbing it silently into the other ten channels.

FRAME SAMPLING - 10 FRAMES AT 1 FPS

This is a train/serve parity contract, not a tuning choice. The deployed pipeline samples at 1 fps
over a 10-second window; training on a different spacing would measure a different thing. An earlier
model in this project drifted to 1.93 s against a served 0.67 s with nothing checking it.

Decode is the bottleneck, not MediaPipe - measured at 2.52 s against 0.53 s per clip. So the sampler
switches strategy by cost: sequential grab()/retrieve() for short clips, keyframe seeking for long
ones. Two EngageNet subjects (subject_13, subject_45) are encoded at 1000 fps / 10,000 frames, which
makes sequential decode pathological; seeking handles them in normal time.

CHECKPOINTING

This script previously wrote a single npz at the end. A Colab runtime was lost after ~2 hours of
extraction and every clip had to be redone. It now writes a partial npz every `--checkpoint-every`
clips and `--resume` skips whatever the output already holds, so an interrupted run costs minutes.

    python extract_geometry.py --clips-dir /content/raw/Validation \
                               --out /content/geom/Validation.npz --resume
"""
from __future__ import annotations

import argparse
import os
import pathlib
import sys
import time

import cv2
import numpy as np

FEATURE_NAMES = ["yaw", "pitch", "roll", "ear_l", "ear_r", "ear_mean",
                 "mouth_open", "gaze_x", "gaze_y", "motion", "face_found"]
N_FRAMES = 10
FPS = 1.0

# MediaPipe FaceLandmarker canonical indices.
EYE_L = [33, 160, 158, 133, 153, 144]     # outer, upper x2, inner, lower x2
EYE_R = [362, 385, 387, 263, 373, 380]
IRIS_L, IRIS_R = 468, 473
LIP_UP, LIP_DN, LIP_L, LIP_R = 13, 14, 61, 291

# Above this many frames to walk sequentially, seek instead. Decode dominates cost, and the two
# 1000 fps subjects would otherwise take ~20 s per clip.
SEEK_IF_FRAMES_OVER = 1200


def _euler_from_matrix(m):
    """Yaw/pitch/roll in degrees from the rotation block of the 4x4 transformation matrix."""
    r = np.asarray(m, dtype=np.float64)[:3, :3]
    sy = float(np.sqrt(r[0, 0] ** 2 + r[1, 0] ** 2))
    if sy > 1e-6:
        pitch = np.arctan2(r[2, 1], r[2, 2])
        yaw = np.arctan2(-r[2, 0], sy)
        roll = np.arctan2(r[1, 0], r[0, 0])
    else:                                  # gimbal lock
        pitch = np.arctan2(-r[1, 2], r[1, 1])
        yaw = np.arctan2(-r[2, 0], sy)
        roll = 0.0
    return [float(np.degrees(a)) for a in (yaw, pitch, roll)]


def _ear(pts, idx):
    """Eye aspect ratio: mean vertical lid separation over horizontal eye width."""
    p = pts[idx]
    horiz = np.linalg.norm(p[0] - p[3])
    if horiz < 1e-8:
        return np.nan
    return float((np.linalg.norm(p[1] - p[5]) + np.linalg.norm(p[2] - p[4])) / (2.0 * horiz))


def _gaze(pts):
    """Iris centre offset from the eye-corner midpoint, normalised by eye width.

    A proxy, not a calibrated gaze vector: it cannot say where on the screen someone is looking,
    only how far their irises sit from centre. That is enough for the label, and it needs no
    per-user calibration step, which matters for a system that meets each learner once.
    """
    if len(pts) <= max(IRIS_L, IRIS_R):
        return [np.nan, np.nan]
    out = []
    for iris, eye in ((IRIS_L, EYE_L), (IRIS_R, EYE_R)):
        outer, inner = pts[eye[0]], pts[eye[3]]
        w = np.linalg.norm(outer - inner)
        if w < 1e-8:
            return [np.nan, np.nan]
        out.append((pts[iris][:2] - (outer[:2] + inner[:2]) / 2.0) / w)
    g = np.mean(out, axis=0)
    return [float(g[0]), float(g[1])]


def _mouth(pts):
    """Lip separation over mouth width - scale-free, so camera distance does not enter."""
    w = np.linalg.norm(pts[LIP_L] - pts[LIP_R])
    if w < 1e-8:
        return np.nan
    return float(np.linalg.norm(pts[LIP_UP] - pts[LIP_DN]) / w)


def sample_frames(path, n=N_FRAMES, fps=FPS):
    """n frames spaced 1/fps apart. Sequential when cheap, seeking when not."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return []
    native = cap.get(cv2.CAP_PROP_FPS) or 30.0
    if not np.isfinite(native) or native <= 0:
        native = 30.0
    step = max(1, int(round(native / fps)))
    frames = []
    if step * (n - 1) > SEEK_IF_FRAMES_OVER:
        for i in range(n):
            cap.set(cv2.CAP_PROP_POS_MSEC, i * 1000.0 / fps)
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(frame)
    else:
        want = {i * step for i in range(n)}
        i = 0
        while len(frames) < n:
            if not cap.grab():
                break
            if i in want:
                ok, frame = cap.retrieve()
                if ok:
                    frames.append(frame)
            i += 1
    cap.release()
    return frames


def clip_features(frames, landmarker, mp_image_cls, mp_fmt):
    """(N_FRAMES, 11) float32, NaN-padded when a frame is missing or has no face."""
    out = np.full((N_FRAMES, len(FEATURE_NAMES)), np.nan, dtype=np.float32)
    prev = None
    for i in range(min(len(frames), N_FRAMES)):
        rgb = cv2.cvtColor(frames[i], cv2.COLOR_BGR2RGB)
        res = landmarker.detect(mp_image_cls(image_format=mp_fmt, data=rgb))
        if not res.face_landmarks:
            out[i, FEATURE_NAMES.index("face_found")] = 0.0
            prev = None
            continue
        pts = np.array([[p.x, p.y, p.z] for p in res.face_landmarks[0]], dtype=np.float64)

        yaw = pitch = roll = np.nan
        if getattr(res, "facial_transformation_matrixes", None):
            yaw, pitch, roll = _euler_from_matrix(res.facial_transformation_matrixes[0])
        el, er = _ear(pts, EYE_L), _ear(pts, EYE_R)
        gx, gy = _gaze(pts)
        motion = float(np.mean(np.linalg.norm(pts[:, :2] - prev[:, :2], axis=1))) if prev is not None else np.nan

        out[i] = [yaw, pitch, roll, el, er, np.nanmean([el, er]),
                  _mouth(pts), gx, gy, motion, 1.0]
        prev = pts
    return out


def save(dest, ids, feats):
    arr = np.asarray(feats, dtype=np.float32)
    cov = np.nan_to_num(arr[:, :, FEATURE_NAMES.index("face_found")]).mean(axis=1) if len(arr) else arr
    np.savez_compressed(dest, feats=arr, clip_id=np.array(ids),
                        coverage=np.asarray(cov, dtype=np.float32),
                        feature_names=np.array(FEATURE_NAMES))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clips-dir", help="directory of .mp4; omit if using --file-list")
    ap.add_argument("--file-list", help="text file, one clip path per line - the way shards are "
                                        "passed on Windows, where symlinks need admin rights")
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="/content/face_landmarker.task")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--resume", action="store_true",
                    help="skip clips already present in --out and append to it")
    ap.add_argument("--checkpoint-every", type=int, default=100)
    ap.add_argument("--threads", type=int, default=1,
                    help="OpenCV/BLAS threads per worker. Keep at 1 when running several workers: "
                         "MediaPipe threads internally, and oversubscribing cores made two "
                         "concurrent extractors 6x SLOWER than one on a 2-core box.")
    a = ap.parse_args()

    if not a.clips_dir and not a.file_list:
        raise SystemExit("need --clips-dir or --file-list")

    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(var, str(a.threads))
    cv2.setNumThreads(a.threads)

    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision

    landmarker = mp_vision.FaceLandmarker.create_from_options(
        mp_vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=a.model),
            running_mode=mp_vision.RunningMode.IMAGE,
            num_faces=1,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=True))

    if a.file_list:
        clips = [pathlib.Path(l.strip()) for l in
                 pathlib.Path(a.file_list).read_text(encoding="utf-8").splitlines() if l.strip()]
    else:
        clips = sorted(pathlib.Path(a.clips_dir).glob("*.mp4"))
    dest = pathlib.Path(a.out)
    dest.parent.mkdir(parents=True, exist_ok=True)

    ids, feats, seen = [], [], set()
    if a.resume and dest.exists():
        d = np.load(dest, allow_pickle=False)
        ids = [str(c) for c in d["clip_id"]]
        feats = list(d["feats"])
        seen = set(ids)
        print(f"  resuming: {len(seen)} clips already in {dest.name}")

    todo = [c for c in clips if c.stem not in seen]
    if a.limit:
        todo = todo[:a.limit]
    print(f"  {len(todo)} clips to do of {len(clips)} in {a.clips_dir}")

    unreadable, t0 = 0, time.time()
    for k, c in enumerate(todo, 1):
        frames = sample_frames(c)
        if not frames:
            unreadable += 1
        ids.append(c.stem)
        feats.append(clip_features(frames, landmarker, mp.Image, mp.ImageFormat.SRGB))
        if k % a.checkpoint_every == 0:
            save(dest, ids, feats)
            el = time.time() - t0
            print(f"    {k}/{len(todo)}  {el/k:.2f}s/clip  "
                  f"eta {(len(todo)-k)*el/k/60:.0f}m  [checkpointed]", flush=True)

    save(dest, ids, feats)
    # An empty list gives shape (0,), not (0, N_FRAMES, 11), so the summary below would raise on a
    # 1-D array. That happens routinely under --resume: a worker whose shard is already complete has
    # nothing to do and must exit cleanly rather than crash the run it was meant to continue.
    if not feats:
        print(f"  nothing to do; {dest} already holds {len(ids)} clips")
        return 0
    arr = np.asarray(feats, dtype=np.float32)
    ff = np.nan_to_num(arr[:, :, FEATURE_NAMES.index("face_found")])
    print(f"  wrote {dest}  {arr.shape} | unreadable {unreadable} | "
          f"face in all {N_FRAMES} frames: {100*(ff.mean(axis=1) == 1).mean():.1f}% | "
          f"zero frames: {100*(ff.mean(axis=1) == 0).mean():.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
