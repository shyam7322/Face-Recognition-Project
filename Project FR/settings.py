import os

_SETTINGS_DIR = os.path.dirname(os.path.abspath(__file__))   
PROJECT_ROOT = os.path.dirname(_SETTINGS_DIR)               

def _p(relative_path: str) -> str:
    return os.path.join(PROJECT_ROOT, relative_path)


PATHS = {
    'image_dir':          _p('dataset/images'),
    'names_file':         _p('dataset/names.json'),
    'trainer_file':       _p('dataset/embeddings.npy'),
    'ids_file':            _p('dataset/ids.npy'),
    'prototypes_file':     _p('dataset/prototypes.npy'),
    'prototype_ids_file':  _p('dataset/prototype_ids.npy'),
    'backend_meta_file':   _p('dataset/backend.json'),
}

CAMERA = {
    'index':  0,
    'width':  640,
    'height': 480,
}

FACE_DETECTION = {
    'min_size': (80, 80),   
}

TRAINING = {
    'samples_needed': 120,  
}


EMBEDDING = {
    'backend': 'insightface',   # 'insightface' or 'deepface'
}

INSIGHTFACE = {
    'model_name':          'buffalo_l',
    'det_thresh':           0.30,   # minimum detector confidence to accept a face
    'recognition_thresh':   0.30,   # minimum cosine similarity to call it a match
}

DEEPFACE = {
    'model_name':           'Facenet512',
    'recognition_thresh':   0.25,   # minimum cosine similarity to call it a match
}
