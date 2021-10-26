import numpy as np

VERTEX_MESH_COMPOSITION = np.array(
    [
        [-1, -1, -1],
        [+1, -1, -1],
        [+1, +1, -1],
        [-1, +1, -1],
        [-1, -1, +1],
        [+1, -1, +1],
        [+1, +1, +1],
        [-1, +1, +1],
    ]
)

FACE_MESH_COMPOSITION = np.array(
    [
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
        [0, 5, 4],
    ]
)


class Cube:
    # Global tracker
    num_cubes = 0

    def __init__(self, middle_position):
        self.vertices = VERTEX_MESH_COMPOSITION
        # Define the 12 triangles composing the cube
        self.faces = FACE_MESH_COMPOSITION
        for i in range(len(self.vertices)):
            for l in range(3):
                self.vertices[i][l] -= 2 * middle_position[l]

        for j in range(len(self.faces)):
            for k in range(3):
                self.faces[j][k] += Cube.num_cubes * 8

        Cube.num_cubes += 1

    def get_vertices(self):
        return self.vertices

    def get_faces(self):
        return self.faces

    def reset_cube_count():
        num_cubes = 0
