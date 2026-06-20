
import json
import logging
import os

import cv2
import numpy as np
from sklearn.metrics.pairwise import cosine_distances

from embedder import embed_for_recognition, get_backend, get_recognition_threshold
from face_utils import get_face_app
from settings import CAMERA, INSIGHTFACE, PATHS
from user_store import load_names

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def _check_backend_matches_training() -> bool:
    if not os.path.exists(PATHS['backend_meta_file']):
        print('Warning: no training metadata found; cannot verify embedding backend. Proceeding anyway.')
        return True
    with open(PATHS['backend_meta_file'], 'r', encoding='utf-8') as fs:
        meta = json.load(fs)
    trained_backend = meta.get('backend')
    current_backend = get_backend()
    if trained_backend != current_backend:
        print(f"ERROR: the saved model was trained with backend='{trained_backend}', but "
              f"settings.EMBEDDING['backend'] is currently '{current_backend}'.")
        print('These embedding spaces are not compatible with each other. '
              'Either switch the setting back, or retrain ("Train model" in the menu).')
        return False
    return True


def load_model():
    if not (os.path.exists(PATHS['prototypes_file']) and os.path.exists(PATHS['prototype_ids_file'])):
        print('Model files not found. Train the model first.')
        return None
    if not _check_backend_matches_training():
        return None
    prototypes = np.load(PATHS['prototypes_file'])
    proto_ids = np.load(PATHS['prototype_ids_file'])
    names = load_names()
    return prototypes, proto_ids, names


def recognize_embedding(embedding, prototypes: np.ndarray, proto_ids: np.ndarray, names: dict):
    if embedding is None or prototypes.size == 0:
        return None, 'Unknown', 0.0
    distances = cosine_distances([embedding], prototypes)[0]
    best_idx = int(np.argmin(distances))
    score = 1.0 - float(distances[best_idx])
    if score >= get_recognition_threshold():
        user_id = int(proto_ids[best_idx])
        return user_id, names.get(str(user_id), f'User {user_id}'), score
    return None, 'Unknown', score


def _draw_result(frame, face, name: str, score: float) -> None:
    x1, y1, x2, y2 = face.bbox.astype(int)
    color = (0, 255, 0) if name != 'Unknown' else (0, 0, 255)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    cv2.putText(frame, f'{name} ({score:.2f})', (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)


def _run_recognition_loop(cap, window_title: str, model) -> None:
    prototypes, proto_ids, names = model
    app = get_face_app(INSIGHTFACE['model_name'], INSIGHTFACE['det_thresh'])
    print('Press ESC to stop.')
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print('No frame received...')
                break
            faces = app.get(frame)
            for face in faces:
                embedding = embed_for_recognition(face, frame)
                _, name, score = recognize_embedding(embedding, prototypes, proto_ids, names)
                _draw_result(frame, face, name, score)
            cv2.imshow(window_title, frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


def recognize_live(video_source=None) -> None:
    model = load_model()
    if model is None:
        return
    video_source = CAMERA['index'] if video_source is None else video_source
    print(f'Connecting to: {video_source}')

    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        print(f'Could not open video source: {video_source}')
        print('Tips: confirm the camera/stream is reachable and not already in use by another app.')
        return
    if hasattr(cv2, 'CAP_PROP_BUFFERSIZE'):
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA['width'])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA['height'])

    _run_recognition_loop(cap, 'Face Recognition', model)


def recognize_video_file() -> None:
    video_path = input('Enter video file path: ').strip()
    if not os.path.exists(video_path):
        print('Video file not found.')
        return
    model = load_model()
    if model is None:
        return
    video = cv2.VideoCapture(video_path)
    if not video.isOpened():
        print('Could not open video file.')
        return
    _run_recognition_loop(video, 'Video Recognition', model)
