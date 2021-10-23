# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import cv2
import numpy as np
from stl import mesh
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

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

def readFile():
    template = cv2.imread("template.png")
    img = cv2.imread("image.png")
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # edges = cv2.Canny(gray, 75, 150)

    result = cv2.matchTemplate(img, template, cv2.TM_SQDIFF_NORMED)

    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

    print(len(result[0]))

    print(min_loc)
    # cv2.imshow("Template Matching", result)
    # cv2.imshow("Edges", edges)
    # cv2.imshow("Image", img)
    image = mpimg.imread("image.png")
    plt.imshow(image)
    plt.plot(min_loc[0], min_loc[1], "og", markersize=10)
    plt.plot(min_loc[0]+180, min_loc[1]+329, "og", markersize=10)# og:shorthand for green circle
    plt.show()

    cv2.waitKey(0)
    cv2.destroyAllWindows()

def writeStl():
    cubes = [Cube((0,0,0)), Cube((0,0,-1)), Cube((0,-1,-1)), Cube((1,-1,-1))]
    stlFile = StlFile(cubes)
    stlFile.genFile()

readFile()
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
