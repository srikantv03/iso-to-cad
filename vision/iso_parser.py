import cv2
import numpy as np
from stl import mesh
import matplotlib.pyplot as plt
import tornado.ioloop
import tornado.web
import base64
from model.cube import Cube
from model.model_file import ModelFile


class IsoParser:
    # mathematical constants
    TAN_30_DEGREES = 1.73 / 3

    # operational constants
    OUTPUT_HEIGHT = 312
    OUTPUT_WIDTH = 540
    MIN_SCALE_THRESHOLD = 1
    MAX_SCALE_THRESHOLD = 4
    SCALE_THRESHOLD_INCREMENT = 0.1
    THRESHOLD_MULTIPLIER = 0.8
    ALPHA = 0.05

    def __init__(self, uri, model_file_name, iso_template):
        self.uri = uri
        self.img = self.__readb64(uri)
        self.iso_template = iso_template
        self.model_file_name = model_file_name

    def __readb64__(self, uri):
        encoded_data = uri.split(",")[1]
        img = cv2.imdecode(
            np.fromstring(base64.b64decode(encoded_data), np.uint8), cv2.IMREAD_COLOR
        )
        return img

    def create_iso_file(self, display_plots=False):
        template = cv2.imread(self.iso_template)

        template_matching_method = cv2.TM_CCOEFF_NORMED

        front_faces = list()

        max_val = 1
        current_max = 0

        scale = self.MIN_SCALE_THRESHOLD
        current_max_scale = self.MIN_SCALE_THRESHOLD

        while scale < self.MAX_SCALE_THRESHOLD:
            scale += self.SCALE_THRESHOLD_INCREMENT

            temp_image = cv2.resize(self.img, (0, 0), fx=scale, fy=scale)
            res = cv2.matchTemplate(temp_image, template, template_matching_method)
            _, max_val, _, max_loc = cv2.minMaxLoc(res)

            if max_val > current_max:
                current_max = max_val
                current_max_scale = scale

        resized_img = cv2.resize(
            self.img, (0, 0), fx=current_max_scale, fy=current_max_scale
        )
        res = cv2.matchTemplate(resized_img, template, template_matching_method)
        _, max_val, _, max_loc = cv2.minMaxLoc(res)
        threshold = self.THRESHOLD_MULTIPLIER * max_val

        while max_val > threshold:
            _, max_val, _, max_loc = cv2.minMaxLoc(res)
            if max_val > threshold:
                front_faces.append(max_loc)
                plt.plot(max_loc[0], max_loc[1], "og", markersize=10)

            start_row = (
                max_loc[1] - self.OUTPUT_HEIGHT // 2
                if max_loc[1] - self.OUTPUT_HEIGHT // 2 >= 0
                else 0
            )
            end_row = (
                max_loc[1] + self.OUTPUT_HEIGHT // 2 + 1
                if max_loc[1] + self.OUTPUT_HEIGHT // 2 + 1 <= res.shape[0]
                else res.shape[0]
            )
            start_col = (
                max_loc[0] - self.OUTPUT_WIDTH // 2
                if max_loc[0] - self.OUTPUT_WIDTH // 2 >= 0
                else 0
            )
            end_col = (
                max_loc[0] + self.OUTPUT_WIDTH // 2 + 1
                if max_loc[0] + self.OUTPUT_WIDTH // 2 + 1 <= res.shape[1]
                else res.shape[0]
            )
            res[start_row:end_row, start_col:end_col] = 0

        front_matrix = []

        for i in range(len(front_faces)):
            front_faces[i] = (
                front_faces[i][0],
                front_faces[i][1] - front_faces[i][0] * 1.73 / 3,
            )

        y_locations = set()
        x_locations = set()

        for i in range(0, len(front_faces)):
            is_in_x = False
            is_in_y = False
            for val in y_locations:
                if (
                    abs(front_faces[i][1] - val)
                    < self.ALPHA * (front_faces[i][1] + val) / 2
                ):
                    is_in_y = True
                    break

            for val in x_locations:
                if (
                    abs(front_faces[i][0] - val)
                    < self.ALPHA * (front_faces[i][0] + val) / 2
                ):
                    is_in_x = True
                    break

            if is_in_x == False:
                x_locations.add(front_faces[i][0])
            if is_in_y == False:
                y_locations.add(front_faces[i][1])

        sorted_x_locations = sorted(list(x_locations))
        sorted_y_locations = sorted(list(y_locations))

        for x in range(len(sorted_x_locations)):
            front_matrix.append([])
            for y in range(len(sorted_y_locations)):
                front_matrix[x].append(0)

        for face_location in front_faces:
            location = [0, 0]
            for xl in range(len(sorted_x_locations)):
                if (
                    abs(face_location[0] - sorted_x_locations[xl])
                    < self.ALPHA * (face_location[0] + sorted_x_locations[xl]) / 2
                ):
                    location[0] = xl
            for yl in range(len(sorted_y_locations)):
                if (
                    abs(face_location[1] - sorted_y_locations[yl])
                    < self.ALPHA * (face_location[1] + sorted_y_locations[yl]) / 2
                ):
                    location[1] = yl

            front_matrix[location[0]][location[1]] = 1

        cubes = []
        for x in range(len(front_matrix)):
            for y in range(len(front_matrix[x])):
                if front_matrix[x][y] == 1:
                    cubes.append(Cube((x, y, 0)))

        model_file = ModelFile(cubes)
        model_file.gen_file(self.model_file_name)

        Cube.reset_cube_count()

        if display_plots:
            plt.show()
            cv2.waitKey(0)
