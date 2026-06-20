
import cv2
import numpy as np

ARCFACE_DST = np.array([
    [38.2946, 51.6963],
    [73.5318, 51.5014],
    [56.0252, 71.7366],
    [41.5493, 92.3655],
    [70.7299, 92.2041],
], dtype=np.float32)

_face_app_cache = {}


def get_face_app(model_name: str = 'buffalo_l', det_thresh: float = 0.30):

    cache_key = (model_name, det_thresh)
    if cache_key not in _face_app_cache:
        from insightface.app import FaceAnalysis
        app = FaceAnalysis(name=model_name, providers=['CPUExecutionProvider'])
        app.prepare(ctx_id=-1, det_thresh=det_thresh)
        _face_app_cache[cache_key] = app
    return _face_app_cache[cache_key]


def normalize_embedding(emb: np.ndarray) -> np.ndarray:
    if emb is None:
        return emb
    norm = np.linalg.norm(emb)
    return emb / norm if norm > 0 else emb


def crop_face(img: np.ndarray, face, output_size=(112, 112)) -> np.ndarray:
    try:
        x1, y1, x2, y2 = face.bbox.astype(int)
        h, w = img.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        crop = img[y1:y2, x1:x2]
        if crop.size == 0:
            return cv2.resize(img, output_size)
        return cv2.resize(crop, output_size)
    except Exception:
        return cv2.resize(img, output_size)


def align_face(img: np.ndarray, face, output_size=(112, 112)) -> np.ndarray:

    kps = None
    for attr in ('kps', 'landmark', 'keypoints'):
        kps = getattr(face, attr, None)
        if kps is not None:
            break
    if kps is not None and len(kps) >= 5:
        try:
            src = np.array(kps[:5], dtype=np.float32)
            M, _ = cv2.estimateAffinePartial2D(src, ARCFACE_DST, method=cv2.LMEDS)
            if M is not None:
                return cv2.warpAffine(img, M, output_size,
                                       flags=cv2.INTER_LINEAR,
                                       borderMode=cv2.BORDER_REPLICATE)
        except Exception:
            pass
    if kps is not None and len(kps) >= 3:
        try:
            src = np.array(kps[:3], dtype=np.float32)
            dst = ARCFACE_DST[:3].copy()
            M = cv2.getAffineTransform(src, dst)
            return cv2.warpAffine(img, M, output_size,
                                   flags=cv2.INTER_LINEAR,
                                   borderMode=cv2.BORDER_REPLICATE)
        except Exception:
            pass
    return crop_face(img, face, output_size)


def augment_face_images(aligned: np.ndarray) -> list:
    return [
        aligned,
        cv2.flip(aligned, 1),
        cv2.convertScaleAbs(aligned, alpha=1.15, beta=20),
        cv2.convertScaleAbs(aligned, alpha=0.85, beta=-20),
        cv2.GaussianBlur(aligned, (3, 3), 0),
        cv2.convertScaleAbs(aligned, alpha=1.1, beta=0),
    ]
