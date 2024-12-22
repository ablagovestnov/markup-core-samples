from pathlib import Path
import cv2
from drill_core_process import DrillCoreProcess

def find_max_side(images_dir):
    """Находит максимальный размер стороны среди изображений."""
    max_side = 0
    for image_path in images_dir.glob("*.JPG"):
        image = cv2.imread(str(image_path))
        if image is not None:
            max_side = max(max_side, max(image.shape[:2]))
    print(f'Found max side: {max_side}')
    return max_side

def main(name, debug=False):
    print(f'Hi, {name}')

    images_dir = Path('input')

    if debug:
        images_dir = Path('test_input')

    if not images_dir.exists():
        print(f"Directory {images_dir} not found!")
        return

    jpegs = list(images_dir.glob("*.JPG"))
    if not jpegs:
        print("No JPG files found in the directory.")
        return

    dcp = DrillCoreProcess()
    max_side = find_max_side(images_dir)

    # Используем новый метод process_images для обработки всех изображений.
    dcp.process_images(
        images=[str(jpeg_path) for jpeg_path in jpegs],
        max_side=max_side,
        resize_to=1024,
        purpose='train',  # Можно задать общий purpose или разделить внутри метода.
        debug=debug
    )

if __name__ == '__main__':
    # Добавлена возможность запуска с режимом отладки.
    import sys
    debug_mode = '--debug' in sys.argv
    main('Ultron Almighty', debug=debug_mode)
