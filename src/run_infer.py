# src/run_infer.py — outfit split (upper/lower/shoes) via MediaPipe pose + person mask

import os, sys, time, argparse
from pathlib import Path
import numpy as np
import cv2

# Optional deps
try:
    from ultralytics import YOLO
except Exception:
    YOLO = None

try:
    import mediapipe as mp
    MP_POSE = mp.solutions.pose
    MP_DRAW = mp.solutions.drawing_utils
    MP_SEG  = mp.solutions.selfie_segmentation
except Exception:
    mp = MP_POSE = MP_DRAW = MP_SEG = None


def dbg(*a): print(*a, flush=True)
def ensure_dir(p: Path): p.mkdir(parents=True, exist_ok=True)

def save_image(path: Path, img):
    ensure_dir(path.parent)
    ok = cv2.imwrite(str(path), img)
    if not ok: raise IOError(f"Gagal simpan: {path}")
    dbg("✅", path)

def read_image_strict(pth: str):
    p = Path(pth).expanduser()
    try: p = p.resolve(strict=False)
    except: pass
    if not p.exists(): return None, None
    try:
        data = np.fromfile(str(p), dtype=np.uint8)
        img  = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if img is not None: return img, str(p)
    except: pass
    img = cv2.imread(str(p), cv2.IMREAD_COLOR)
    if img is not None: return img, str(p)
    try:
        from PIL import Image; pil = Image.open(str(p)).convert("RGB")
        return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR), str(p)
    except: return None, None


# ---------- Pose + Seg ----------
def person_mask_mediapipe(img_bgr):
    """Binary mask person ukuran sama dengan gambar (0/255)."""
    if MP_SEG is None: return None
    h, w = img_bgr.shape[:2]
    with MP_SEG.SelfieSegmentation(model_selection=1) as seg:
        res = seg.process(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    if res.segmentation_mask is None: return None
    m = (res.segmentation_mask > 0.5).astype(np.uint8) * 255
    return cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)

def pose_landmarks(img_bgr):
    """Kembalikan dict landmark {name:(x,y,vis)} dalam pixel; None jika gagal."""
    if MP_POSE is None: return None
    h, w = img_bgr.shape[:2]
    with MP_POSE.Pose(static_image_mode=True, min_detection_confidence=0.5) as pose:
        res = pose.process(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    if not res.pose_landmarks: return None
    lm = res.pose_landmarks.landmark
    def px(i): return int(lm[i].x * w), int(lm[i].y * h), lm[i].visibility
    idx = MP_POSE.PoseLandmark
    names = {
        "l_shoulder": idx.LEFT_SHOULDER,
        "r_shoulder": idx.RIGHT_SHOULDER,
        "l_hip": idx.LEFT_HIP,
        "r_hip": idx.RIGHT_HIP,
        "l_ankle": idx.LEFT_ANKLE,
        "r_ankle": idx.RIGHT_ANKLE,
    }
    return {k: px(int(v)) for k, v in names.items()}, res.pose_landmarks

def split_regions(mask_person, lms, h, w):
    """Bikin mask upper/lower/shoes pakai y dari bahu/hip/ankle; intersect dengan person mask."""
    def yof(*keys, default):  # median y dengan fallback
        ys = [lms[k][1] for k in keys if lms.get(k, (0,0,0))[2] > 0.3]
        return int(np.median(ys)) if ys else int(default)

    y_sh = yof("l_shoulder", "r_shoulder", default=0.30*h)
    y_hp = yof("l_hip", "r_hip",         default=0.55*h)
    y_an = yof("l_ankle", "r_ankle",     default=0.90*h)

    # batas aman
    y_sh = np.clip(y_sh, 0, h-1); y_hp = np.clip(y_hp, 0, h-1); y_an = np.clip(y_an, 0, h-1)

    upper = np.zeros((h,w), np.uint8); upper[:max(y_hp, int(0.5*h)), :] = 255
    lower = np.zeros((h,w), np.uint8); lower[max(y_hp, int(0.45*h)):y_an, :] = 255
    shoes = np.zeros((h,w), np.uint8); shoes[max(y_an- int(0.05*h), int(0.80*h)):, :] = 255

    # intersect dengan person mask
    upper &= mask_person; lower &= mask_person; shoes &= mask_person
    return upper, lower, shoes

def crop_from_mask(img, mask, pad=8):
    ys, xs = np.where(mask>0)
    if len(ys)==0: return None
    y1,y2 = max(0, ys.min()-pad), min(img.shape[0], ys.max()+pad)
    x1,x2 = max(0, xs.min()-pad), min(img.shape[1], xs.max()+pad)
    return img[y1:y2, x1:x2]


# ---------- Main pipeline ----------
def run(image_path, out_dir, yolo_conf=0.35):
    dbg(">>> Outfit split start")
    img, resolved = read_image_strict(image_path)
    if img is None:
        dbg("❌ Gagal membaca gambar:", image_path); return
    h, w = img.shape[:2]
    ensure_dir(Path(out_dir))
    save_image(Path(out_dir)/"copy_input.jpg", img)

    # 1) (opsional) deteksi YOLO untuk debugging
    if YOLO is not None:
        try:
            model = YOLO("yolov8n.pt")
            yres = model.predict(source=[img], conf=yolo_conf, iou=0.5, verbose=False)
            save_image(Path(out_dir)/"result.jpg", yres[0].plot())
        except Exception as e:
            dbg("⚠️ YOLO skip:", e)

    # 2) person mask
    pm = person_mask_mediapipe(img)
    if pm is None:
        # fallback: gunakan seluruh gambar (kurang bagus)
        dbg("⚠️ SelfieSegmentation tidak tersedia → pakai full image sebagai mask.")
        pm = np.ones((h,w), np.uint8)*255

    # 3) pose landmarks
    lms, raw_lm = pose_landmarks(img)
    if lms is None:
        dbg("⚠️ Pose tidak terdeteksi, pakai fallback batas proporsi.")
        lms = {}

    # 4) bagi upper/lower/shoes
    upper, lower, shoes = split_regions(pm, lms, h, w)

    # 5) simpan overlay & crops
    overlay = img.copy()
    overlay[upper>0] = (0,255,0)   # upper = hijau
    overlay[lower>0] = (255,0,0)   # lower = biru
    overlay[shoes>0] = (0,0,255)   # shoes = merah
    vis = cv2.addWeighted(img, 0.5, overlay, 0.5, 0)

    # gambar pose (jika ada) di atas overlay
    if raw_lm is not None and MP_DRAW is not None:
        MP_DRAW.draw_landmarks(vis, raw_lm, MP_POSE.POSE_CONNECTIONS)

    save_image(Path(out_dir)/"mask_overlay.jpg", vis)

    cu = crop_from_mask(img, upper);  cl = crop_from_mask(img, lower);  cs = crop_from_mask(img, shoes)
    if cu is not None: save_image(Path(out_dir)/"upper.jpg", cu)
    else: dbg("⚠️ upper kosong")
    if cl is not None: save_image(Path(out_dir)/"lower.jpg", cl)
    else: dbg("⚠️ lower kosong")
    if cs is not None: save_image(Path(out_dir)/"shoes.jpg", cs)
    else: dbg("⚠️ shoes kosong")

    dbg("🏁 Selesai split outfit")


def parse_args():
    ap = argparse.ArgumentParser("Outfit splitter (upper/lower/shoes)")
    ap.add_argument("image_path")
    ap.add_argument("output_dir")
    ap.add_argument("--conf", type=float, default=0.35)
    return ap.parse_args()

def main():
    dbg(">>> run_infer.py")
    args = parse_args()
    run(args.image_path, args.output_dir, yolo_conf=args.conf)

if __name__ == "__main__":
    main()
