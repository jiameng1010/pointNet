from pyntcloud import PyntCloud
import pyembree
import numpy as np
import trimesh
from trimesh import sample, ray, triangles
from trimesh.ray.ray_pyembree import RayMeshIntersector
import pandas as pd


cloud = PyntCloud.from_file("/home/mjia/Documents/ShapeCompletion/test.ply")
sample = cloud.get_sample(name='mesh_random_sampling', as_PyntCloud=True, n=1024)
sample.plot(mesh=True)

mesh = trimesh.load('../ModelNet40/desk/train/desk_0008.off')

mesh.show()