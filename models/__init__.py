import os
from functools import partial

import torch


def _load_optic_flow_model(model_name, models_path, download=False):
    assert model_name == 'two_stream_dynamic_model'
    if not os.path.exists(f'{models_path}/two_stream/{model_name}.pth'):
        download = True

    if download:

        if os.path.exists(f'{models_path}/two_stream/{model_name}.pth'):
            os.system(f"rm -rf {models_path}/two_stream")
        import gdown
        url = 'https://drive.google.com/uc?id=10qoSx0P3TJzf17bUN42x1ZAFNjr-J69f'
        output = f'{models_path}/two_stream/{model_name}.pth'
        os.system(f"mkdir -p {models_path}/two_stream/")
        gdown.download(url, output, quiet=False)

    from models.optic_flow import MSOEmultiscale

    model = MSOEmultiscale()
    states_dict = torch.load(f'{models_path}/two_stream/{model_name}.pth')
    model.load_state_dict(states_dict)
    model = model.eval()

    return model


_model_factories = {}
_model_factories['optic_flow'] = partial(_load_optic_flow_model, model_name='two_stream_dynamic_model')


def get_available_models():
    return _model_factories.keys()


def get_model(name, *args, **kwargs):
    return _model_factories[name](*args, **kwargs)
