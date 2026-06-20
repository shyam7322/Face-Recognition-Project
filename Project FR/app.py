
import logging

from face_capture import capture_user_images
from face_recognizer import recognize_live, recognize_video_file
from face_trainer import train as train_model
from settings import CAMERA
from user_store import delete_user, ensure_directories, print_users

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def prompt_delete_user() -> None:
    print_users()
    user_id_text = input('Enter user ID to delete: ').strip()
    if not user_id_text.isdigit():
        print('Invalid ID')
        return
    deleted = delete_user(int(user_id_text))
    print('Deleted.' if deleted else 'User not found or not deleted.')


def prompt_recognize() -> None:
    print('\n1. Use local camera')
    print('2. Use IP camera stream')
    selection = input('Select: ').strip()
    if selection == '2':
        video_source = input('Enter RTSP/HTTP URL: ').strip()
        print(f'Connecting to IP stream: {video_source}')
        recognize_live(video_source)
    else:
        print(f'Using local camera (index {CAMERA["index"]})')
        recognize_live(CAMERA['index'])


MENU = {
    '1': ('Capture new user images', capture_user_images),
    '2': ('Train model', train_model),
    '3': ('Recognize live (camera/IP)', prompt_recognize),
    '4': ('Recognize video file', recognize_video_file),
    '5': ('List users', print_users),
    '6': ('Delete user', prompt_delete_user),
    '0': ('Exit', None),
}


def main() -> None:
    ensure_directories()
    while True:
        print('\nFaceID CLI Menu:')
        for key, (text, _) in MENU.items():
            print(f'  {key}. {text}')
        choice = input('Select option: ').strip()
        if choice == '0':
            print('Exiting.')
            break
        action = MENU.get(choice)
        if action is None:
            print('Invalid selection.')
            continue
        action[1]()


if __name__ == '__main__':
    main()
