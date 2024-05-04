import os
import numpy as np
import torch
import yaml
import argparse
import shutil
import wandb
from tqdm import tqdm

from losses.loss import Loss
from models.meshnca import MeshNCA
from utils.camera import PerspectiveCamera
from utils.mesh import Mesh
from utils.render import Renderer

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='configs/test.yaml', help="configuration")


def main(config):
    device = torch.device(config['device'])

    if 'wandb' in config and True:
        wandb.login(key=config['wandb']['key'], relogin=True)
        wandb.init(project=config['wandb']['project'], name=config['experiment_name'],
                   dir=config['experiment_path'], config=config)

    def get_device_config(cfg):
        cfg['device'] = device
        return cfg

    meshnca_config = get_device_config(config['meshnca'])
    model = MeshNCA(**meshnca_config).to(device)

    # state_dict = torch.load("Waffle_001.pth")
    #
    # with torch.no_grad():
    #     model.fc1.weight.data = state_dict['update_mlp.0.weight']
    #     model.fc1.bias.data = state_dict['update_mlp.0.bias']
    #     model.fc2.weight.data = state_dict['update_mlp.2.weight']
    #     model.fc2.bias.data = state_dict['update_mlp.2.bias']

    with torch.no_grad():
        icosphere_config = get_device_config(config['train']['icosphere'])
        icosphere = Mesh.load_icosphere(**icosphere_config)

        pool = model.seed(config['train']['pool_size'], icosphere.Nv)

        loss_fn = Loss(**config['loss'])

        renderer = Renderer(**config['renderer'])

        camera_config = get_device_config(config['train']['camera'])

        test_mesh_config = get_device_config(config['train']['test_mesh'])
        test_mesh = Mesh.load_from_obj(**test_mesh_config)

    optimizer = torch.optim.Adam(model.parameters(), lr=config['train']['lr'])
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=config['train']['lr_decay_steps'],
                                                        gamma=config['train']['lr_decay_gamma'])

    render_channels = model.get_render_channels()
    step_range = config['train']['step_range']
    inject_seed_interval = config['train']['inject_seed_interval']
    batch_size = config['train']['batch_size']
    epochs = config['train']['epochs']
    for epoch in tqdm(range(epochs)):
        with torch.no_grad():
            batch_idx = np.random.choice(len(pool), batch_size, replace=False)

            x = pool[batch_idx]
            if epoch % inject_seed_interval == 0:
                x[:1] = model.seed(1, icosphere.Nv)

        step_n = np.random.randint(step_range[0], step_range[1])
        # with torch.no_grad():
        for _ in range(step_n):
            x = model(x, icosphere, None)

        # render_channels = [0, 1, 2, 6, 7, 8, 9, 10, 11]
        x_render = x[..., render_channels] + 0.5
        camera = PerspectiveCamera.generate_random_view_cameras(**camera_config)

        rendered_image = renderer.render(icosphere, camera, x_render)
        # rendered_image: [batch_size, num_views, height, width, num_features]
        rendered_image = torch.flatten(rendered_image, start_dim=0, end_dim=1).permute(0, 3, 1, 2)
        # rendered_image: [batch_size * num_views, num_features, height, width]

        input_dict = {
            'rendered_images': rendered_image,
            'nca_state': x,
        }
        return_summary = (epoch + 0) % config['train']['summary_interval'] == 0
        loss, loss_log, summary = loss_fn(input_dict, return_summary=return_summary)

        with torch.no_grad():
            loss.backward()
            for p in model.parameters():
                p.grad /= (p.grad.norm() + 1e-8)  # normalize gradients
            optimizer.step()
            optimizer.zero_grad()
            lr_scheduler.step()
            pool[batch_idx] = x  # update pool

        wandb.log(loss_log, step=epoch)
        if return_summary:
            wandb.log({'rendered images': wandb.Image(summary['appearance-images'], caption='Rendered Images')},
                      step=epoch)

    torch.save(model.state_dict(), f'{config["experiment_path"]}/model.pth')


if __name__ == "__main__":
    args = parser.parse_args()
    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    exp_name = config['experiment_name']
    exp_path = f'results/{exp_name}/'
    config['experiment_path'] = exp_path
    if not os.path.exists(exp_path):
        os.makedirs(exp_path)

    shutil.copy(f'{args.config}', f'{exp_path}/config.yaml')
    main(config)
