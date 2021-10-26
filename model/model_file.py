import numpy as np
from stl import mesh


class ModelFile:
    cubes = []

    def __init__(self, cubes=[]):
        self.cubes = cubes

    def gen_file(self, file_name: str):
        all_faces = None
        all_vertices = None

        for i in range(len(self.cubes)):
            object = self.cubes[i]
            if not i:
                all_faces = object.getFaces()
                all_vertices = object.getVertices()
            else:
                f = object.getFaces()
                v = object.getVertices()
                all_faces = np.concatenate([all_faces, f])
                all_vertices = np.concatenate([all_vertices, v])

        cube = mesh.Mesh(np.zeros(all_faces.shape[0], dtype=mesh.Mesh.dtype))
        for i, f in enumerate(all_faces):
            for j in range(3):
                cube.vectors[i][j] = all_vertices[f[j], :]

        cube.save(f"{file_name}.stl")
