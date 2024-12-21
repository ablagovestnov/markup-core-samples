import math
import random
from functools import reduce

import cv2
import numpy as np
import pandas as pd
from test import PointsClusterer

class DrillCoreProcess:
    def __init__(self, images_path='images/'):
        self.images_path = images_path

    def process_images(self, images, metrics=None, max_side=0, resize_to=512, purpose='train', debug=False):

        for idx, image_path in enumerate(images):
            file_name = image_path.split('/')[-1]
            print(f"Processing image {idx + 1}/{len(images)}: {file_name}")
            self.process_image(image_path, file_name, metrics, max_side, resize_to, purpose, debug)

    def process_image(self, image_name='', file_name='', metrics=None, max_side=0, resize_to=512, purpose='train', debug=False):
        if not image_name:
            return

        metrics = metrics or {'start': 0, 'end': 0}

        location = image_name
        original = cv2.imread(location)
        if original is None:
            # Проверяется, удалось ли загрузить изображение. Если нет, выводится сообщение об ошибке.
            print(f"Error: Cannot read image {image_name}")
            return

        saved = original.copy()
        original = original[350:]

        # Загружается изображение в градациях серого и обрезается верхняя часть.
        image = cv2.imread(location, cv2.IMREAD_GRAYSCALE)[350:]
        # Применяется размытие Гаусса для уменьшения шума перед выделением контуров.
        blur = cv2.GaussianBlur(image, (5, 5), 0)
        # Выполняется выделение границ на изображении с помощью алгоритма Canny.
        edges = cv2.Canny(blur, 105, 200, None, 3)
        if debug:
            print("Edges detected")
            cv2.imshow("Edges", edges)

        # Обнаруживаются линии с использованием преобразования Хафа.
        lines = cv2.HoughLines(edges, 1, np.pi / 180, 150, None, 0, 0)

        if lines is None:
            print("No lines detected.")
            return

        # Обнаруженные линии обрабатываются и фильтруются для дальнейшей кластеризации.
        resulted_lines = self.extract_lines(lines)
        if debug:
            print(f"Extracted lines: {resulted_lines}")

        # Кластеризуются обработанные линии для группировки близких линий.
        clusters = self.cluster_lines(resulted_lines)
        if debug:
            print(f"Clusters found: {clusters}")

        if len(clusters) < 2:
            print("ERROR: not enough clusters")
            return

        # Фильтруются кластеры, чтобы исключить пересечения и оставить значимые области.
        filtered_clusters = self.filter_clusters(clusters)
        if debug:
            print(f"Filtered clusters: {filtered_clusters}")

        if len(filtered_clusters) < 2:
            print("ERROR: not enough filtered clusters")
            return

        # Извлекаются области изображения, соответствующие буровым кернам.
        drill_core_samples = self.extract_core_samples(original, filtered_clusters, saved.shape[1])
        if debug:
            print(f"Extracted core samples: {drill_core_samples}")

        if len(drill_core_samples) < 2 or len(drill_core_samples) > 4:
            print("ERROR: wrong core number")
            return

        # Масштабируется изображение до указанного размера.
        resized_image = self.resize_image(saved, resize_to)
        # Создаются файлы разметки для обнаруженных кернов в формате YOLO.
        self.fill_labeling_file(file_name, resized_image, drill_core_samples, resize_to, purpose)

        # Сохраняется обработанное изображение.
        cv2.imwrite(f'yolo_data/images/{purpose}/resized_{file_name}', resized_image)
        if debug:
            print(f"Processed image saved as: yolo_data/images/{purpose}/resized_{file_name}")
            cv2.imshow("Resized Image", resized_image)
            cv2.waitKey(0)
        print(f"Processed and saved: {file_name}")

    def extract_lines(self, lines):
        """
        Извлекает координаты горизонтальных линий на основе параметров преобразования Хафа.

        Линии преобразуются из полярной системы координат в декартовую, фильтруются по длине и ориентации.
        Возвращает список координат Y, соответствующих найденным горизонтальным линиям.
        """
        resulted_lines = []
        for i in range(len(lines)):
            rho, theta = lines[i][0]
            # Преобразуем параметры линии из полярной системы координат в декартовую
            a, b = math.cos(theta), math.sin(theta)
            x0, y0 = a * rho, b * rho
            # Определяем точки линии, экстраполируя её на большое расстояние
            pt1 = (int(x0 + 1000 * (-b)), int(y0 + 1000 * (a)))
            pt2 = (int(x0 - 1000 * (-b)), int(y0 - 1000 * (a)))
            # Фильтруем линии: оставляем только горизонтальные линии достаточной длины
            if abs(pt1[0] - pt2[0]) > 1000 and -0.001 < a < 0.001:
                resulted_lines.append(round((pt1[1] + pt2[1]) / 2))
        return resulted_lines

    def cluster_lines(self, lines):
        # Создаётся объект кластеризации для группировки линий.
        clusterer = PointsClusterer(lines)
        clusters = clusterer.my_cluster(75)
        # Отфильтровываются пустые и незначительные кластеры.
        clusters = list(filter(lambda c: len(c) > 0, clusters))
        # Обработка только тех кластеров, которые содержат минимум два элемента
        clusters = [
            (min(c), max(c), round(sum(c) / len(c))) for c in clusters if len(c) >= 2 and abs(c[0] - c[1]) > 15
        ]
        # Кластеры сортируются по средней позиции.
        clusters.sort(key=lambda c: c[2])
        return clusters

    def filter_clusters(self, clusters):
        # Фильтруются кластеры, чтобы исключить пересечения.
        filtered_clusters = [clusters[0]]
        for i in range(1, len(clusters)):
            if abs(filtered_clusters[-1][1] - clusters[i][0]) >= 200:
                filtered_clusters.append(clusters[i])
        return filtered_clusters

    def extract_core_samples(self, original, clusters, width):
        drill_core_samples = []
        for i in range(len(clusters) - 1):
            # Создаются области, соответствующие кернам, с учётом ширины изображения.
            drill_core_samples.append(
                (
                    (25, 350 + clusters[i][1]),
                    (width - 25, 350 + clusters[i + 1][0])
                )
            )
        return drill_core_samples

    def resize_image(self, image, resize_to):
        # Масштабируется изображение с сохранением пропорций.
        ratio = resize_to / image.shape[1]
        return cv2.resize(image, (resize_to, round(ratio * image.shape[0])))

    def fill_labeling_file(self, file_name, img, cores, resize_to, purpose):
        yolo_string = []
        for core in cores:
            # Вычисляются нормализованные координаты меток для формата YOLO.
            x_center_abs = round(core[0][0]) + (round(core[1][0]) - round(core[0][0])) / 2
            y_center_abs = round(core[0][1]) + (round(core[1][1]) - round(core[0][1])) / 2
            image_width, image_height = img.shape[1], img.shape[0]
            width_of_label_abs = round(core[1][0]) - round(core[0][0])
            height_of_label_abs = round(core[1][1]) - round(core[0][1])

            x_center_norm = x_center_abs / image_width
            y_center_norm = y_center_abs / image_height
            width_norm = width_of_label_abs / image_width
            height_norm = height_of_label_abs / image_height

            yolo_string.append(f'0 {x_center_norm} {y_center_norm} {width_norm} {height_norm}')

        # Сохраняется файл разметки для изображения.
        with open(f'yolo_data/labels/{purpose}/resized_{file_name[:-4]}.txt', 'w') as f:
            f.write('\n'.join(yolo_string))
