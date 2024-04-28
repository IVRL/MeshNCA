import torch

# Real valued spherical harmonics of different degrees
from utils.rsh import rsh_functions

from torch_geometric.nn import MessagePassing
from utils.mesh import Mesh


class MeshNCA(MessagePassing):
    def __init__(self, channels=16, fc_dim=128,
                 sh_order=1, aggregation='sum',
                 stochastic_update=True, seed_mode='zeros',
                 condition=None, device='cuda:0'):
        super(MeshNCA, self).__init__(aggr=aggregation)
        self.channels = channels
        self.fc_dim = fc_dim
        self.sh_order = sh_order
        self.stochastic_update = stochastic_update

        assert seed_mode in ['zeros', 'random']
        self.seed_mode = seed_mode

        assert condition in [None, 'MPE', 'PE']
        # MPE: Motion Positional Encoding, PE: Positional Encoding
        self.condition = condition

        self.device = device

        num_sh = (sh_order + 1) ** 2  # Number of spherical harmonics coefficients
        self.fc1 = torch.nn.Linear((num_sh + 1) * channels, fc_dim, bias=True)
        self.fc2 = torch.nn.Linear(fc_dim, channels, bias=True)
        torch.nn.init.xavier_normal_(self.fc1.weight, gain=0.2)
        torch.nn.init.xavier_normal_(self.fc2.weight, gain=1.2)

        self.adaptation_mlp = torch.nn.Sequential(
            self.fc1,
            torch.nn.ReLU(),
            self.fc2,
        )

        self.sh_func = rsh_functions[sh_order]

    def message(self, x_j: torch.Tensor, x_i: torch.Tensor) -> torch.Tensor:
        """
        :param x_j: Neighbor vertex features [num_edges, channels + 3]
        :param x_i: Center vertex features [num_edges, channels + 3]

        :return: The message passed from
        the neighbors to the center [num_edges, channels * 4]
        """
        center_pos = x_i[:, -3:]  # [num_edges, 3]
        center_features = x_i[:, :-3]  # [num_edges, channels]

        neighbor_pos = x_j[:, -3:]  # [num_edges, 3]
        neighbor_features = x_j[:, :-3]  # [num_edges, channels]

        direction = neighbor_pos - center_pos  # [num_edges, 3]
        direction = direction / (torch.norm(direction, dim=1, keepdim=True) + 1e-8)  # [num_edges, 3]

        sh_coefficients = self.sh_func(direction).unsqueeze(2)  # [num_edges, num_sh, 1]
        feature_diff = (neighbor_features - center_features).unsqueeze(1)  # [num_edges, 1, channels]

        message = (sh_coefficients * feature_diff).view(x_i.shape[0], -1)  # [num_edges, channels * num_sh]

        return message

    def perception(self, x: torch.Tensor, mesh: Mesh) -> torch.Tensor:
        """
        :param x: per-vertex features [batch_size, num_vertices, channels]
        :param mesh: Mesh object

        :return: per-vertex perception vector [batch_size, num_vertices, channels * (num_sh + 1)]
        """
        # torch_geometric does not support batched message passing. We need to handle the batch dimension manually.
        batch_size, num_vertices = x.shape[0], x.shape[1]

        x = x.view(-1, x.shape[-1])  # [batch_size * num_vertices, channels]

        edge_index = mesh.edge_index  # [2, num_edges * 2]
        vertex_positions = mesh.vertices  # [num_vertices, 3]
        if batch_size > 1:
            # We create a batched edge index by offsetting the edge indices for each element in the batch.
            edge_index = torch.cat(
                [
                    edge_index + i * mesh.Nv for i in range(batch_size)
                ],
                dim=1)  # [2, num_edges * 2 * batch_size]

            vertex_positions = vertex_positions.repeat(batch_size, 1)  # [num_vertices * batch_size, 3]

        # Concatenate the vertex positions to the features for the perception stage of the update rule
        # Per-vertex perception vector z [batch_size * num_vertices, channels * num_sh]
        z = self.propagate(edge_index, x=torch.cat([x, vertex_positions], dim=1))

        # Concatenate the per-vertex features to the perception vector for the adaptation stage of the update rule
        z = torch.cat([z, x], dim=1)  # [batch_size * num_vertices, channels * (num_sh + 1)]

        z = z.view(batch_size, num_vertices, -1)  # [batch_size, num_vertices, channels * (num_sh + 1)]

        return z

    def forward(self, x: torch.Tensor, mesh: Mesh, h: torch.Tensor = None) -> torch.Tensor:
        """
        :param x: per-vertex features [batch_size, num_vertices, channels]
        :param mesh: Mesh object
        :param h: Optional per-vertex condition vector [batch_size, num_vertices, condition_dim]

        :return: the updated per-vertex features [batch_size, num_vertices, num_features]
        """
        batch_size, num_vertices = x.shape[0], x.shape[1]
        z = self.perception(x, mesh)  # [batch_size, num_vertices, channels * (num_sh + 1)]

        # Adaptation stage of the update rule
        if h is not None:
            # Concatenate the condition vector to the perception vector
            z = torch.cat([z, h], dim=2)  # [batch_size, num_vertices, channels * (num_sh + 1) + condition_dim]

        delta_x = self.adaptation_mlp(z)  # [batch_size, num_vertices, channels]
        if self.stochastic_update:
            update_rate = 0.5
            # Per-vertex random binary mask
            update_mask = (torch.rand(batch_size, num_vertices, 1, device=delta_x.device) + update_rate).floor()
            delta_x = delta_x * update_mask

        return x + delta_x

    def graft_initialization(self):
        # @TODO: Implement the graft initialization method
        pass

    def seed(self, pool_size: int, num_vertices: int):
        if self.seed_mode == 'zeros':
            return torch.zeros(pool_size, num_vertices, self.channels, device=self.device)
        elif self.seed_mode == 'random':
            return torch.rand(pool_size, num_vertices, self.channels, device=self.device) * 0.1


if __name__ == '__main__':
    from utils.mesh import Mesh
    from utils.camera import PerspectiveCamera
    from utils.render import Renderer
    from utils.video import VideoWriter
    from tqdm import tqdm
    import numpy as np

    device = torch.device("cuda:0")

    meshnca = MeshNCA(device=device).to(device)
    state_dict = torch.load("../Waffle_001.pth")
    print(state_dict.keys())

    with torch.no_grad():
        meshnca.fc1.weight.data = state_dict['update_mlp.0.weight']
        meshnca.fc1.bias.data = state_dict['update_mlp.0.bias']
        meshnca.fc2.weight.data = state_dict['update_mlp.2.weight']
        meshnca.fc2.bias.data = state_dict['update_mlp.2.bias']

    with torch.no_grad():
        # Load a mesh from an .obj file
        # mesh = Mesh('../data/meshes/mug/mug_remesh_lvl1.obj', device=device)
        mesh = Mesh('../data/meshes/sphere/sphere_remesh_lvl1.obj', device=device)
        mesh = Mesh('../data/meshes/icosphere_train.obj', device=device)

        # Define a perspective camera
        np.random.seed(42)
        # camera = PerspectiveCamera.generate_random_view_cameras(1, distance=2.0, device=device)
        camera = PerspectiveCamera()
        renderer = Renderer(height=256, width=256, device=device)

        # meshnca = MeshNCA(device=device).to(device)
        x = meshnca.seed(1, mesh.Nv)

        with VideoWriter('../tmp.mp4', fps=30.0) as video:
            for i in tqdm(range(360)):
                for _ in range(4):
                    x = meshnca(x, mesh, None)
                color = x[:, :, :3] + 0.5
                color = torch.clamp(color, 0.0, 1.0)
                image = renderer.render(mesh, camera, color, True).cpu().numpy()
                camera.rotateY(1.0)
                image = np.hstack(image)
                video.add(image)
