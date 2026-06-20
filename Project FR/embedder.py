
import logging

import cv2
import numpy as np

from face_utils import align_face, get_face_app, normalize_embedding
from settings import DEEPFACE, EMBEDDING, INSIGHTFACE

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

VALID_BACKENDS = ('insightface', 'deepface')


def get_backend() -> str:
    backend = EMBEDDING.get('backend', 'insightface')
    if backend not in VALID_BACKENDS:
        raise ValueError(f"Unknown EMBEDDING['backend']: {backend!r}. Use one of {VALID_BACKENDS}.")
    return backend


def get_recognition_threshold() -> float:
    return DEEPFACE['recognition_thresh'] if get_backend() == 'deepface' else INSIGHTFACE['recognition_thresh']

def embed_from_face_object(face):
    if get_backend() != 'insightface':
        return None
    emb = getattr(face, 'embedding', None)
    return normalize_embedding(emb) if emb is not None else None


def embed_from_image(aligned_bgr_img: np.ndarray):
    backend = get_backend()
    resized = cv2.resize(aligned_bgr_img, (112, 112))

    if backend == 'insightface':
        app = get_face_app(INSIGHTFACE['model_name'], INSIGHTFACE['det_thresh'])
        faces = app.get(resized)
        if not faces:
            return None
        return normalize_embedding(faces[0].embedding)

    # backend == 'deepface'
    try:
        from deepface import DeepFace
    except ImportError:
        logger.error("EMBEDDING['backend'] is 'deepface' but the deepface package isn't installed. "
                      "Run: pip install -r requirements-deepface.txt")
        return None
    try:
        result = DeepFace.represent(
            img_path=resized,
            model_name=DEEPFACE['model_name'],
            detector_backend='skip',    
            enforce_detection=False,
            align=False,
        )
    except Exception as exc:
        logger.warning(f'DeepFace embedding failed: {exc}')
        return None
    if not result:
        return None
    return normalize_embedding(np.array(result[0]['embedding'], dtype=np.float32))

def embed_for_recognition(face, frame: np.ndarray):
    emb = embed_from_face_object(face)
    if emb is not None:
        return emb
    aligned = align_face(frame, face)
    return embed_from_image(aligned)
