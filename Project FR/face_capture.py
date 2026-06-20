
import logging
import os

import cv2

from face_utils import align_face, augment_face_images, get_face_app
from settings import CAMERA, FACE_DETECTION, INSIGHTFACE, PATHS, TRAINING
from user_store import ensure_directories, get_next_image_index, get_next_user_id, load_names, save_names

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

VALID_EXTENSIONS = ('.jpg', '.jpeg', '.png')


def initialize_camera(camera_index: int):
    cam = cv2.VideoCapture(camera_index)
    if not cam.isOpened():
        logger.error('Could not open camera.')
        return None
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA['width'])
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA['height'])
    return cam


def select_best_face(faces):
    if not faces:
        return None
    return max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))


def save_capture(aligned, user_folder: str, file_index: int, use_augmentation: bool):
    if use_augmentation:
        variants = augment_face_images(aligned)
        for offset, img in enumerate(variants):
            cv2.imwrite(os.path.join(user_folder, f'face_{file_index + offset}.jpg'), img)
        return file_index + len(variants), len(variants)
    cv2.imwrite(os.path.join(user_folder, f'face_{file_index}.jpg'), aligned)
    return file_index + 1, 1


def capture_user_images(app=None):
    ensure_directories()
    names = load_names()

    user_name = input('Enter user name: ').strip()
    if not user_name:
        print('Name cannot be empty.')
        return

    use_augmentation = input('Enable augmentation? Multiplies each capture x6 [y/N]: ').strip().lower() == 'y'

    user_id = get_next_user_id()
    user_folder = os.path.join(PATHS['image_dir'], f'{user_name}_{user_id}')
    os.makedirs(user_folder, exist_ok=True)

    if app is None:
        logger.info('Loading InsightFace detector...')
        app = get_face_app(INSIGHTFACE['model_name'], INSIGHTFACE['det_thresh'])

    cam = initialize_camera(CAMERA['index'])
    if cam is None:
        print('Camera initialization failed.')
        return

    existing = len([f for f in os.listdir(user_folder) if f.lower().endswith(VALID_EXTENSIONS)])
    file_idx = get_next_image_index(user_folder)
    captured = 0
    min_w, min_h = FACE_DETECTION['min_size']

    print(f'Capturing for "{user_name}" (ID={user_id}). Existing: {existing}/{TRAINING["samples_needed"]}')
    print('Press ESC to stop early.')

    try:
        while (existing + captured) < TRAINING['samples_needed']:
            ret, frame = cam.read()
            if not ret:
                continue

            faces = app.get(frame)
            qualified = [f for f in faces if getattr(f, 'det_score', 1.0) >= INSIGHTFACE['det_thresh']]
            best = select_best_face(qualified)

            if best is not None:
                x1, y1, x2, y2 = best.bbox.astype(int)
                w, h = x2 - x1, y2 - y1
                if w >= min_w and h >= min_h:
                    aligned = align_face(frame, best, output_size=(112, 112))
                    file_idx, written = save_capture(aligned, user_folder, file_idx, use_augmentation)
                    captured += written
                    total = existing + captured
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f'Captured {total}/{TRAINING["samples_needed"]}',
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                else:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
                    cv2.putText(frame, 'Move closer to the camera',
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            else:
                cv2.putText(frame, 'No face detected',
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            cv2.imshow('Capture', frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break
    finally:

        cam.release()
        cv2.destroyAllWindows()

    if captured > 0:
        names[str(user_id)] = user_name
        save_names(names)
        print(f'Saved {captured} images for "{user_name}" -> {user_folder}')
    else:
        print('No new images captured. User not registered.')


if __name__ == '__main__':
    capture_user_images()
