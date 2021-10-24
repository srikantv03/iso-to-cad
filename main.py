# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import io
import math

import cv2
import numpy as np
from stl import mesh
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import tornado.ioloop
import tornado.web
from io import BytesIO
import base64

class CADHandler(tornado.web.RequestHandler):
    def post(self):
        readFile(self.get_body_argument("img"))
        f = BytesIO(open("cube.stl", "rb").read()).read()

        print(type(f))
        self.write(f)

class Cube:
    numCubes = 0
    # Create the mesh
    def __init__(self, midPos):
        print(Cube.numCubes)
        self.vertices = np.array([ \
            [-1, -1, -1],
            [+1, -1, -1],
            [+1, +1, -1],
            [-1, +1, -1],
            [-1, -1, +1],
            [+1, -1, +1],
            [+1, +1, +1],
            [-1, +1, +1]])
        # Define the 12 triangles composing the cube
        self.faces = np.array([ \
            [0, 3, 1],
            [1, 3, 2],
            [0, 4, 7],
            [0, 7, 3],
            [4, 5, 6],
            [4, 6, 7],
            [5, 1, 2],
            [5, 2, 6],
            [2, 3, 6],
            [3, 7, 6],
            [0, 1, 5],
            [0, 5, 4]])
        for i in range(len(self.vertices)):
            for l in range(3):
                self.vertices[i][l] -= 2 * midPos[l]

        for j in range(len(self.faces)):
            for k in range(3):
                self.faces[j][k] += Cube.numCubes * 8

        Cube.numCubes += 1

    def getVertices(self):
        return self.vertices

    def getFaces(self):
        return self.faces

class StlFile:
    cubes = []
    def __init__(self, cubes = []):
        self.cubes = cubes

    def genFile(self):
        allFaces = None
        allVertices = None

        for i in range(len(self.cubes)):
            object = self.cubes[i]
            if i == 0:
                allFaces = object.getFaces()
                allVertices = object.getVertices()
            else:
                f = object.getFaces()
                v = object.getVertices()
                allFaces = np.concatenate([allFaces, f])
                allVertices = np.concatenate([allVertices, v])

        cube = mesh.Mesh(np.zeros(allFaces.shape[0], dtype=mesh.Mesh.dtype))
        for i, f in enumerate(allFaces):
            for j in range(3):
                cube.vectors[i][j] = allVertices[f[j], :]

        cube.save('cube.stl')

    # def returnObjURI(self):


def readb64(uri):
   encoded_data = uri.split(',')[1]
   nparr = np.fromstring(base64.b64decode(encoded_data), np.uint8)
   img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
   return img

def readFile(inimg):
    print(inimg)
    h = 312
    w = 540
    template = cv2.imread("template.png")
    img = readb64(inimg)
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # edges = cv2.Canny(gray, 75, 150)

    method = cv2.TM_CCOEFF_NORMED

    # cv2.imshow("Template Matching", result)
    # cv2.imshow("Edges", edges)
    # cv2.imshow("Image", img)

    frontFaces = list()

    # fake out max_val for first run through loop
    max_val = 1
    scale = 1
    max_scale = 1
    current_max = 0
    temp_image = cv2.resize(img, (0, 0), fx=1, fy=1)
    while scale < 4:
        scale += .1
        temp_image = cv2.resize(img, (0, 0), fx=scale, fy=scale)
        res = cv2.matchTemplate(temp_image, template, method)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        if max_val > current_max:
            current_max = max_val
            max_scale = scale
    image = cv2.resize(readb64(inimg), (0, 0), fx=max_scale, fy=max_scale)
    plt.imshow(image)
    img = cv2.resize(img, (0, 0), fx=max_scale, fy=max_scale)
    res = cv2.matchTemplate(img, template, method)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    threshold = .8 * max_val
    while max_val > threshold:
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        print(max_val)
        if max_val > threshold:
            frontFaces.append(max_loc)
            plt.plot(max_loc[0], max_loc[1], "og", markersize=10)
        start_row = max_loc[1] - h // 2 if max_loc[1] - h // 2 >= 0 else 0
        end_row = max_loc[1] + h // 2 + 1 if max_loc[1] + h // 2 + 1 <= res.shape[0] else res.shape[0]
        start_col = max_loc[0] - w // 2 if max_loc[0] - w // 2 >= 0 else 0
        end_col = max_loc[0] + w // 2 + 1 if max_loc[0] + w // 2 + 1 <= res.shape[1] else res.shape[0]
        res[start_row: end_row, start_col: end_col] = 0

    frontMatrix = [[1]]
    lastLoc = (0,0)

    for i in range(len(frontFaces)):
        frontFaces[i] = (frontFaces[i][0], frontFaces[i][1] - frontFaces[i][0] * 1.73/3)

    print(frontFaces)


    for i in range(1, len(frontFaces)):
        val = frontFaces[i]
        # same x loc
        print(abs(val[0] - val[1]))
        print(0.5 * (val[0] + val[1])/2)
        if abs(frontFaces[i - 1][0] - val[0]) < 0.05 * (frontFaces[i - 1][0] + val[1])/2:
            tempMatrix = []
            for i in range(len(frontMatrix[lastLoc[0]])):
                tempMatrix.append(0)
            tempMatrix[lastLoc[1]] = 1
            lastLoc = (len(frontMatrix) - 1, lastLoc[1])
            frontMatrix.append(tempMatrix)

        elif abs(frontFaces[i - 1][1] - val[1]) < .35 * abs((frontFaces[i - 1][0] - val[0])):
            for i in range(len(frontMatrix)):
                frontMatrix[i].append(0)
            frontMatrix[lastLoc[0]][len(frontMatrix[0]) - 1] = 1
            lastLoc = (lastLoc[0], len(frontMatrix[0]) - 1)

    cubes = []
    for x in range(len(frontMatrix)):
        for y in range(len(frontMatrix[x])):
            if frontMatrix[x][y] == 1:
                cubes.append(Cube((x, y, 0)))


    print(len(cubes))
    stlFile = StlFile(cubes)
    stlFile.genFile()

    plt.show()
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def make_app():
    return tornado.web.Application([
        (r"/", CADHandler),
    ])

if __name__ == "__main__":
    app = make_app()
    app.listen(8888)
    tornado.ioloop.IOLoop.current().start()
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
