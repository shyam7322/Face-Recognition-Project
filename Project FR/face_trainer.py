
import json
import logging
import os
import time

import cv2
import numpy as np

from embedder import embed_from_image, get_backend
from face_utils import normalize_embedding
from settings import PATHS

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

VALID_EXTENSIONS = ('.jpg', '.jpeg', '.png')


def get_images_and_labels(image_dir: str):
    paths, ids = [], []
    if not os.path.exists(image_dir):
        return paths, ids
    for folder_name in os.listdir(image_dir):
        folder_path = os.path.join(image_dir, folder_name)
        if not os.path.isdir(folder_path):
            continue
        try:
            uid = int(folder_name.rsplit('_', 1)[1]) if '_' in folder_name else int(folder_name)
        except ValueError:
            logger.warning(f'Skipping unparseable folder: {folder_name}')
            continue
        for fname in os.listdir(folder_path):
            if fname.lower().endswith(VALID_EXTENSIONS):
                paths.append(os.path.join(folder_path, fname))
                ids.append(uid)
    return paths, ids


def compute_prototypes(embeddings_array: np.ndarray, ids_array: np.ndarray,
                        min_samples_for_filtering: int = 5, outlier_percentile: float = 15.0):
    unique_ids = np.unique(ids_array)
    proto_embs, proto_ids = [], []
    for uid in unique_ids:
        mask = ids_array == uid
        user_embs = embeddings_array[mask]
        if len(user_embs) >= min_samples_for_filtering:
            centroid = normalize_embedding(user_embs.mean(axis=0))
            sims = user_embs @ centroid
            threshold = np.percentile(sims, outlier_percentile)
            kept = user_embs[sims >= threshold]
            logger.info(f'  ID={uid}: kept {len(kept)}/{mask.sum()} after outlier removal')
            user_embs = kept
        proto_embs.append(normalize_embedding(user_embs.mean(axis=0)))
        proto_ids.append(uid)
    return np.array(proto_embs), np.array(proto_ids)


def _save_backend_metadata(num_images: int, num_users: int) -> None:
    meta = {
        'backend': get_backend(),
        'trained_at': time.strftime('%Y-%m-%dT%H:%M:%S'),
        'num_images': num_images,
        'num_users': num_users,
    }
    os.makedirs(os.path.dirname(PATHS['backend_meta_file']), exist_ok=True)
    with open(PATHS['backend_meta_file'], 'w', encoding='utf-8') as fs:
        json.dump(meta, fs, indent=4)


def train(image_dir: str = None) -> None:
    image_dir = image_dir or PATHS['image_dir']
    paths, ids = get_images_and_labels(image_dir)
    if not paths:
        print('No training images found.')
        return

    print(f'Found {len(paths)} images from {len(set(ids))} users.')
    print(f'Extracting embeddings using backend="{get_backend()}" ...')

    embeddings, valid_ids = [], []
    for idx, (path, uid) in enumerate(zip(paths, ids), start=1):
        img = cv2.imread(path)
        if img is None:
            logger.warning(f'Cannot read: {path}')
            continue
        emb = embed_from_image(img)
        if emb is None:
            logger.warning(f'No embedding extracted from: {path}')
            continue
        embeddings.append(emb)
        valid_ids.append(uid)
        if idx % 20 == 0:
            print(f'  Processed {idx}/{len(paths)}...')

    if not embeddings:
        print('No valid embeddings extracted.')
        return

    embs_array = np.array(embeddings)
    ids_array = np.array(valid_ids)

    os.makedirs(os.path.dirname(PATHS['trainer_file']), exist_ok=True)
    np.save(PATHS['trainer_file'], embs_array)
    np.save(PATHS['ids_file'], ids_array)
    print(f'Saved {len(embeddings)} raw embeddings.')

    print('Computing per-user prototypes with outlier filtering...')
    proto_embs, proto_ids = compute_prototypes(embs_array, ids_array)
    np.save(PATHS['prototypes_file'], proto_embs)
    np.save(PATHS['prototype_ids_file'], proto_ids)

    _save_backend_metadata(num_images=len(embeddings), num_users=len(proto_ids))
    print(f'Saved {len(proto_embs)} prototypes. Training complete (backend="{get_backend()}").')


if __name__ == '__main__':
    train()
