
import json
import logging
import os
import re
import shutil

from settings import PATHS

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

VALID_EXTENSIONS = ('.jpg', '.jpeg', '.png')


def ensure_directories() -> None:
    os.makedirs(PATHS['image_dir'], exist_ok=True)
    if not os.path.exists(PATHS['names_file']):
        save_names({})


def load_names() -> dict:
    if not os.path.exists(PATHS['names_file']):
        return {}
    try:
        with open(PATHS['names_file'], 'r', encoding='utf-8') as fs:
            content = fs.read().strip()
            return json.loads(content) if content else {}
    except Exception as exc:
        logger.warning(f'Could not read names file: {exc}')
        return {}


def save_names(names: dict) -> None:
    os.makedirs(os.path.dirname(PATHS['names_file']), exist_ok=True)
    with open(PATHS['names_file'], 'w', encoding='utf-8') as fs:
        json.dump(names, fs, indent=4, ensure_ascii=False)


def _parse_user_id(folder_name: str):
    try:
        return int(folder_name.rsplit('_', 1)[1]) if '_' in folder_name else int(folder_name)
    except ValueError:
        return None


def get_next_user_id() -> int:
    image_dir = PATHS['image_dir']
    if not os.path.exists(image_dir):
        return 1
    ids = [0]
    for folder_name in os.listdir(image_dir):
        if not os.path.isdir(os.path.join(image_dir, folder_name)):
            continue
        uid = _parse_user_id(folder_name)
        if uid is not None:
            ids.append(uid)
    return max(ids) + 1


def get_next_image_index(user_folder: str) -> int:
    if not os.path.exists(user_folder):
        return 1
    pattern = re.compile(r'face_(\d+)\.(?:jpg|jpeg|png)$', re.IGNORECASE)
    max_index = 0
    for filename in os.listdir(user_folder):
        match = pattern.match(filename)
        if match:
            max_index = max(max_index, int(match.group(1)))
    return max_index + 1


def find_user_folder(user_id: int):
    image_dir = PATHS['image_dir']
    if not os.path.exists(image_dir):
        return None
    for folder_name in os.listdir(image_dir):
        if folder_name == str(user_id) or folder_name.endswith(f'_{user_id}'):
            path = os.path.join(image_dir, folder_name)
            if os.path.isdir(path):
                return path
    return None


def list_users() -> list:

    names = load_names()
    image_dir = PATHS['image_dir']
    users = []
    if not os.path.exists(image_dir):
        return users
    for folder_name in sorted(os.listdir(image_dir)):
        folder_path = os.path.join(image_dir, folder_name)
        if not os.path.isdir(folder_path):
            continue
        uid = _parse_user_id(folder_name)
        if uid is None:
            continue
        count = len([f for f in os.listdir(folder_path) if f.lower().endswith(VALID_EXTENSIONS)])
        users.append({'id': uid, 'name': names.get(str(uid), folder_name), 'images': count, 'folder': folder_path})
    return users


def print_users() -> None:
    users = list_users()
    print('\nUsers:')
    if not users:
        print('  No users found')
        return
    for u in users:
        print(f"  ID={u['id']} Name={u['name']} Images={u['images']}")


def delete_user(user_id: int, confirm: bool = True) -> bool:
    names = load_names()
    folder = find_user_folder(user_id)
    deleted = False
    if folder:
        if confirm:
            answer = input(f'Confirm delete {os.path.basename(folder)}? [y/N]: ').strip().lower()
            if answer != 'y':
                print('Cancelled')
                return False
        shutil.rmtree(folder)
        deleted = True
    if str(user_id) in names:
        del names[str(user_id)]
        save_names(names)
    return deleted
