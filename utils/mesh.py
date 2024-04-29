import numpy as np
import torch
import kaolin

from utils.icosphere import icosphere


class Mesh:
    def __init__(self, vertices, faces, mesh_name="Mesh", device="cuda:0"):
        """
        :param vertices: numpy array or torch tensor with shape [num_vertices, 3]
        :param faces: numpy array or torch tensor with shape [num_faces, 3]
        :param mesh_name: name of the mesh
        :param device: pytorch device to store the mesh data

        The mesh class has the following attributes:
        - vertices: torch tensor with shape [num_vertices, 3]
        - faces: torch tensor with shape [num_faces, 3]
        - edges: torch tensor with shape [num_edges, 2]
        - edge_index: torch tensor with shape [2, 2 * num_edges] used for message passing
        - face_normals: torch tensor with shape [num_faces, 3]
        - vertex_normals: torch tensor with shape [num_vertices, 3]
        - laplacian_matrix: torch sparse tensor with shape [num_vertices, num_vertices]
        - Nv, Nf, Ne: number of vertices, faces, and edges in the mesh
        """
        self.mesh_name = mesh_name

        if isinstance(vertices, np.ndarray):
            self.vertices = torch.tensor(vertices, dtype=torch.float32, device=device)
        else:
            self.vertices = vertices.to(device)

        if isinstance(faces, np.ndarray):
            self.faces = torch.tensor(faces, dtype=torch.int64, device=device)
        else:
            self.faces = faces.to(device)

        with torch.no_grad():
            # Extract edges from faces of the mesh
            edges = torch.column_stack([self.faces, torch.roll(self.faces, shifts=-1, dims=1)])
            edges = edges.reshape(-1, 2)
            edges, _ = torch.sort(edges, dim=1)
            edges = torch.unique(edges, dim=0)
            edges_idx = torch.argsort(edges, dim=0)[:, 0]
            self.edges = edges[edges_idx]

            bi_edges = torch.cat([edges, edges[:, [1, 0]]], dim=0)
            # bi_edges = np.sort(bi_edges, axis=1)
            sorted_ids = torch.argsort(bi_edges, dim=0, descending=False)[:, 0]
            self.edge_index = bi_edges[sorted_ids].T

        # Normalize the mesh so that it fits into a unit sphere
        self.normalize_into_unit_sphere()

        # Compute the normals for faces and vertices of the mesh
        self._compute_normals()

        # Compute the Laplacian matrix of the mesh
        self._compute_laplacian()

        self.Nv, self.Nf, self.Ne = self.vertices.shape[0], self.faces.shape[0], self.edges.shape[0]

    @staticmethod
    def load_from_obj(obj_path, subdivision_iter=0, **kwargs):
        """Load a mesh from an obj file."""
        # We use kaolin to load the mesh.
        # Other libraries like trimesh create duplicate vertices if a vertex has several normals.
        mesh = kaolin.io.obj.import_mesh(obj_path, with_normals=True)
        mesh_name = obj_path.split('/')[-1].split(".")[0]

        vertices, faces = mesh.vertices, mesh.faces

        if subdivision_iter > 0:
            vertices, faces = kaolin.ops.mesh.subdivide_trianglemesh(vertices.unsqueeze(0),
                                                                     faces,
                                                                     subdivision_iter)
            vertices, faces = vertices.squeeze(0), faces

        return Mesh(vertices, faces, mesh_name=mesh_name, **kwargs)

    @staticmethod
    def load_icosphere(subdivision_freq=2 ** 6, **kwargs):
        """
        Load an icosphere mesh with a given number of subdivisions.

        :param subdivision_freq: Subdivision frequency

        :return: Mesh object of the icosphere
        the mesh will have 12 + 10 * (subdivision_freq**2 -1) vertices and 20 * (subdivision_freq**2) faces
        Setting subdivision_freq=2**n will create an icosphere sphere with n recursive subdivisions.
        """
        vertices, faces = icosphere(nu=subdivision_freq)
        return Mesh(vertices, faces, mesh_name=f"Icosphere", **kwargs)

    @torch.no_grad()
    def normalize_into_unit_sphere(self):
        """Normalize the vertices of the mesh into a unit sphere."""
        center = torch.mean(self.vertices, dim=0)
        scale = torch.max(torch.norm(self.vertices - center, p=2, dim=1))
        self.vertices = (self.vertices - center) / scale

    @torch.no_grad()
    def _compute_normals(self):
        """Compute the normals for faces and vertices of the mesh."""
        vertex_normals = torch.zeros_like(self.vertices)
        vertices_faces = self.vertices[self.faces]

        faces_normals = torch.cross(
            vertices_faces[:, 2] - vertices_faces[:, 1],
            vertices_faces[:, 0] - vertices_faces[:, 1],
            dim=1,
        )
        self.face_normals = torch.nn.functional.normalize(
            faces_normals, eps=1e-6, dim=1
        )

        # NOTE: this is already applying the area weighting as the magnitude
        # of the cross product is 2 x area of the triangle.
        vertex_normals = vertex_normals.index_add(
            0, self.faces[:, 0], faces_normals
        )
        vertex_normals = vertex_normals.index_add(
            0, self.faces[:, 1], faces_normals
        )
        vertex_normals = vertex_normals.index_add(
            0, self.faces[:, 2], faces_normals
        )

        self.vertex_normals = torch.nn.functional.normalize(
            vertex_normals, eps=1e-6, dim=1
        )

    @torch.no_grad()
    def _compute_laplacian(self):
        """Compute the Laplacian matrix (D - A) of the mesh using the adjacency matrix in a sparse form."""
        device = self.vertices.device
        num_vertices = self.vertices.shape[0]
        edge_list = self.edges

        # Create a tensor with ones to fill the adjacency matrix at the edge positions
        ones = torch.ones(edge_list.shape[0], dtype=torch.float32).to(device)

        # Create a sparse adjacency matrix using the edge list and ones
        adj_matrix_sparse = torch.sparse_coo_tensor(edge_list.t(), ones, size=(num_vertices, num_vertices),
                                                    dtype=torch.float32, device=device)

        # Make the adjacency matrix symmetric for undirected graphs
        adj_matrix_sparse = adj_matrix_sparse + adj_matrix_sparse.t()

        # Compute the degree (valence) for each vertex
        degree_vals = torch.sparse.sum(adj_matrix_sparse, dim=1).values()
        self.min_valence = torch.min(degree_vals).item()
        self.max_valence = torch.max(degree_vals).item()
        self.average_valence = torch.mean(degree_vals).item()

        degree_matrix_sparse = torch.sparse_coo_tensor(
            torch.stack([torch.arange(num_vertices), torch.arange(num_vertices)]).to(device), degree_vals,
            size=(num_vertices, num_vertices), dtype=torch.float32, device=device
        )
        self.laplacian_matrix = degree_matrix_sparse - adj_matrix_sparse

    def vertex2face_features(self, vertex_features: torch.Tensor) -> torch.Tensor:
        """
        :param vertex_features: A torch tensor with shape [batch_size, num_vertices, num_features]
        :return: A torch tensor with shape [batch_size, num_faces, 3, num_features]
        """
        return kaolin.ops.mesh.index_vertices_by_faces(vertex_features, self.faces)

    def __repr__(self):
        return f"Mesh(" \
               f"\n\tName = {self.mesh_name}" \
               f"\n\tVertices = {self.vertices.shape[0]}," \
               f"\n\tFaces = {self.faces.shape[0]}," \
               f"\n\tEdges = {self.edges.shape[0]}," \
               f"\n\tValence (min, max, average) = {(self.min_valence, self.max_valence, self.average_valence)}," \
               f"\n\tDevice = {self.vertices.device}," \
               f"\n)"


if __name__ == '__main__':
    mesh1 = Mesh.load_from_obj('../data/meshes/cat/cat.obj', subdivision_iter=1, device='cuda:0')
    mesh2 = Mesh.load_icosphere(2 ** 6, device='cuda:0')

    print(mesh1)

    print(mesh2)

