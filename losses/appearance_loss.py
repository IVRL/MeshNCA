import torch
import torchvision.models as torch_models
import torchvision.transforms.functional as TF
from PIL import Image

from utils.misc import load_texture_image


class AppearanceLoss(torch.nn.Module):
    def __init__(self,
                 image_size=(224, 224),
                 target_images_path="data/textures/bubbly_0101.jpg",
                 num_channels=3,
                 vgg_layers=(1, 3, 6, 8, 11, 13, 15, 22, 29),
                 device='cuda:0'):
        """
        :param image_size: Image resolution used for calculating the appearance  loss
        :param target_images_path: str or a dictionary of strings, path to the target images
        :param num_channels: tuple or a dictionary of tuples, corresponding MeshNCA channels for the target images
        :param vgg_layers: VGG layers used for calculating the style loss
        :param device: PyTorch device
        """
        super(AppearanceLoss, self).__init__()

        self.image_size = image_size
        if not isinstance(target_images_path, dict):
            self.target_images_path = {"RGB": target_images_path}
        else:
            self.target_images_path = target_images_path

        if not isinstance(num_channels, dict):
            self.num_channels = {"RGB": num_channels}
        else:
            self.num_channels = num_channels

        self.vgg_layers = vgg_layers

        self.device = device

        vgg = torch_models.vgg16(weights=torch_models.VGG16_Weights.IMAGENET1K_V1).features.to(device)
        vgg_layers = []
        for l in vgg:
            if isinstance(l, torch.nn.MaxPool2d):
                l = torch.nn.AvgPool2d(l.kernel_size, l.stride, l.padding, l.ceil_mode)
            vgg_layers.append(l)
        self.vgg = torch.nn.Sequential(*vgg_layers)

        self._load_target_images(low_pass_filter=True)

        self.style_loss_fn = OptimalTransportLoss(n_samples=1024, device=self.device)

    def get_vgg_features(self, x, flatten=False, include_image_as_feature=True):
        """

        :param x: input images [b, c, h, w]
        :param flatten: Whether to flatten the returned features to remove the spatial dimensions
        :param include_image_as_feature: Whether to inclued the input image in the returned features

        :return: A list of pytorch tensors containing the features extracted from different layers of the VGG network.
        """
        vgg_layers = self.vgg_layers
        mean = torch.tensor([0.485, 0.456, 0.406], device=x.device)[:, None, None]
        std = torch.tensor([0.229, 0.224, 0.225], device=x.device)[:, None, None]
        x = (x - mean) / std
        b, c, h, w = x.shape

        features = []
        if include_image_as_feature:
            features = [x.reshape(b, c, h * w)] if flatten else [x]

        for i, layer in enumerate(self.vgg[:max(vgg_layers) + 1]):
            x = layer(x)
            if i in vgg_layers:
                b, c, h, w = x.shape
                if flatten:
                    features.append(x.reshape(b, c, h * w))
                else:
                    features.append(x)
        return features

    @torch.no_grad()
    def _load_target_images(self, low_pass_filter=True):
        target_images = []
        total_channels = 0
        for i, key in enumerate(sorted(self.target_images_path)):
            target_image_path = self.target_images_path[key]
            if low_pass_filter:
                target_image, _ = load_texture_image(target_image_path, (128, 128))
                target_image = TF.resize(target_image, self.image_size, antialias=True)
            else:
                target_image, _ = load_texture_image(target_image_path, self.image_size)

            target_image = target_image.to(self.device)
            target_images.append(target_image)

            chn = self.num_channels[key]
            assert chn == 3 or chn == 1, \
                "Either RGB or mono-color images are supported. " \
                "The mono-color images will be repeated to 3 channels for calculating the loss."
            total_channels += chn

        self.target_images = torch.cat(target_images, dim=0)

        self.target_features = self.get_vgg_features(self.target_images)
        self.total_channels = total_channels

    def forward(self, input_dict, return_summary=True):
        """

        :param input_dict:  A dictionary containing the necessary tensors for calculating the appearance loss.
                            required keys: ['rendered_images']: a tensor of shape [b, c, h, w]
                            The rendered images should be in range [0, 1].
                            The number of channels c should match the total number of target channels.
        :param return_summary: Whether to return a summary dictionary
        :return: A tuple (loss, loss_log dictionary, summary dictionary)
        """

        channels = input_dict['rendered_images'].shape[1]
        assert channels == self.total_channels, \
            f"Target images have {self.total_channels} channels in total," \
            f"but the rendered images have {channels} channels."

        xs = []
        c_start = 0
        for key in sorted(self.num_channels):
            chn = self.num_channels[key]
            c_end = c_start + chn
            x = input_dict['rendered_images'][:, c_start:c_end]
            c_start = c_end

            # # Resize if the image size is different from the target image size
            # if x.shape[2] != self.image_size[0] or x.shape[3] != self.image_size[1]:
            #     x = TF.resize(x, self.image_size, antialias=True)

            # Repeat the mono-color images to 3 channels
            if chn == 1:
                x = x.repeat(1, 3, 1, 1)

            xs.append(x)

        generated_images = torch.stack(xs, dim=1)  # [b, n_targets, 3, h, w]

        b, n_targets, _, h, w = generated_images.shape
        summary = None
        if return_summary:
            with torch.no_grad():
                images = generated_images.permute(0, 1, 3, 4, 2) # [b, n_targets, h, w, 3]
                target_images = TF.resize(self.target_images, [h, w], antialias=True)
                images = torch.cat([target_images.permute(0, 2, 3, 1).unsqueeze(0), images], dim=0)
                images = torch.hstack([
                    torch.vstack([images[i, j] for j in range(images.shape[1])])
                    for i in range(images.shape[0])
                ])
                images = torch.clamp(images, 0.0, 1.0).cpu().numpy()
                images = (images * 255).astype('uint8')
                summary = {
                    "images": Image.fromarray(images)
                }
        generated_images = generated_images.view(-1, 3, h, w)
        generated_features = self.get_vgg_features(generated_images)

        loss = self.style_loss_fn(self.target_features, generated_features)  # [n_targets]

        loss_log = {
            k: loss[i] for i, k in enumerate(sorted(self.num_channels))
        }

        return loss.mean(), loss_log, summary


class OptimalTransportLoss(torch.nn.Module):
    def __init__(self, n_samples=1024, device='cuda:0'):
        super().__init__()

        self.n_samples = n_samples

        self.device = device
        self.color_transform = torch.tensor(
            [[0.577350, 0.577350, 0.577350],
             [-0.577350, 0.788675, -0.211325],
             [-0.577350, -0.211325, 0.788675]], device=device, requires_grad=False)  # [3, 3] matrix

    def rgb_to_yuv(self, rgb):
        """
        :param rgb: Torch tensor of shape [b, n, 3]
        :return: YUV colors of shape [b, n, 3]
        """
        return torch.einsum('bnc,ck->bnk', rgb, self.color_transform)

    def color_matching_loss(self, x, y):
        """
        Color matching distance between two RGB images.
        :param x: (b, n, 3)
        :param y: (b, m, 3)

        :return: (b, n, m)
        """
        x_yuv = self.rgb_to_yuv(x)
        y_yuv = self.rgb_to_yuv(y)

        pairwise_distance = self.pairwise_distances_l2(x_yuv, y_yuv) + self.pairwise_distances_cos(x_yuv, y_yuv)

        m1, m1_inds = pairwise_distance.min(1)
        m2, m2_inds = pairwise_distance.min(2)

        remd = torch.max(m1.mean(dim=1), m2.mean(dim=1))

        return remd

    @staticmethod
    def pairwise_distances_l2(x, y):
        """
        Pairwise L2 distance between two flattened feature sets.
        :param x: (b, n, c)
        :param y: (b, m, c)

        :return: (b, n, m)
        """
        # x, y: (b, n or m, c)
        x_norm = torch.norm(x, dim=2, keepdim=True) ** 2  # (b, n, 1)
        y_t = y.transpose(1, 2)  # (b, c, m) (m may be different from n)
        y_norm = torch.norm(y_t, dim=1, keepdim=True) ** 2  # (b, 1, m)
        cross = torch.matmul(x, y_t)
        dist = x_norm + y_norm - 2.0 * cross  # x + y is of shape b, n, m because of point-wise adding (broadcasting)
        return torch.clamp(dist, 1e-5, 1e5) / x.size(2)

    @staticmethod
    def pairwise_distances_cos(x, y):
        """
        Pairwise Cosine distance between two flattened feature sets.
        :param x: (b, n, c)
        :param y: (b, m, c)

        :return: (b, n, m)
        """
        x_norm = torch.norm(x, dim=2, keepdim=True)  # (b, n, 1)
        y_t = y.transpose(1, 2)  # (b, c, m) (m may be different from n)
        y_norm = torch.norm(y_t, dim=1, keepdim=True)  # (b, 1, m)
        dist = 1. - torch.matmul(x, y_t) / (x_norm * y_norm + 1e-10)  # (b, n, m)
        return dist

    @staticmethod
    def style_loss(x, y, metric="cos"):
        """
        Relaxed Earth Mover's Distance (EMD) between two sets of features.
        :param x: (b, n, c)
        :param y: (b, m, c)
        :param metric: Either 'cos' or 'L2'

        :return: (b, n, m)
        """
        if metric == "cos":
            pairwise_distance = OptimalTransportLoss.pairwise_distances_cos(x, y)
        else:
            pairwise_distance = OptimalTransportLoss.pairwise_distances_l2(x, y)

        m1, m1_inds = pairwise_distance.min(1)
        m2, m2_inds = pairwise_distance.min(2)

        remd = torch.max(m1.mean(dim=1), m2.mean(dim=1))

        return remd

    @staticmethod
    def moment_loss(x, y):
        """
        Calculates the distance between the first and second moments of two sets of features.
        :param x: (b, n, c)
        :param y: (b, m, c)

        :return: (b, n, m)
        """
        mu_x = torch.mean(x, 1, keepdim=True)
        mu_y = torch.mean(y, 1, keepdim=True)
        mu_diff = torch.abs(mu_x - mu_y).mean(dim=(1, 2))

        x_c = x - mu_x
        y_c = y - mu_y
        x_cov = torch.matmul(x_c.transpose(1, 2), x_c) / (x.shape[1] - 1)
        y_cov = torch.matmul(y_c.transpose(1, 2), y_c) / (y.shape[1] - 1)

        cov_diff = torch.abs(x_cov - y_cov).mean(dim=(1, 2))
        return mu_diff + cov_diff

    def forward(self, target_features, generated_features):
        """
        Calculate the optimal transport style loss between two sets of features.

        :param target_features:     List of features for the target images.
                                    Each feature is of shape (n_targets, c, h, w) with varying c, h, w
        :param generated_features:  List of features for the generated images.
                                    Each feature is of shape (b * n_targets, c, h, w) with varying c, h, w

        :return: The OT style loss between the target and generated features [n_targets]
        """
        loss = 0.0

        # Iterate over the VGG layers
        for i, (y, x) in enumerate(zip(target_features, generated_features)):
            layer_weight = 1.0

            n_targets, c_y, h_y, w_y = y.shape

            b, c_x, h_x, w_x = x.shape
            batch_size = b // n_targets
            assert batch_size * n_targets == b, "Batch size must be a multiple of the number of target images"

            # We repeat the target features to match the batch size of the generated features
            # y = y.repeat_interleave(repeats=batch_size, dim=0)
            y = y.repeat(batch_size, 1, 1, 1)

            n_x, n_y = h_x * w_x, h_y * w_y
            x = x.view(b, c_x, n_x)
            y = y.view(b, c_y, n_y)

            # We randomly select n_samples point from the features to calculate the OT loss
            n_samples = min(n_x, n_y, self.n_samples)

            indices_x = torch.argsort(torch.rand(b, 1, n_x, device=x.device), dim=-1)[..., :n_samples]
            x = x.gather(-1, indices_x.expand(b, c_x, n_samples))

            indices_y = torch.argsort(torch.rand(b, 1, n_y, device=y.device), dim=-1)[..., :n_samples]
            y = y.gather(-1, indices_y.expand(b, c_y, n_samples))

            x = x.transpose(1, 2)  # (b, n_samples, c)
            y = y.transpose(1, 2)  # (b, n_samples, c)

            if i == 0 and c_x == c_y == 3:
                layer_loss = self.color_matching_loss(x, y)
            else:
                layer_loss = OptimalTransportLoss.style_loss(x, y) + OptimalTransportLoss.moment_loss(x, y)
            loss += layer_loss * layer_weight

        loss = loss.view(batch_size, n_targets).mean(dim=0)
        return loss  # [n_targets]


if __name__ == '__main__':
    # loss_fn = AppearanceLoss(target_images_path="../data/textures/bubbly_0101.jpg")

    loss_fn = AppearanceLoss(target_images_path={
        "albedo": "../data/pbr_textures/Abstract_008/albedo.jpg",
        "height": "../data/pbr_textures/Abstract_008/height.jpg",
        "normal": "../data/pbr_textures/Abstract_008/normal.jpg",
    }, num_channels={
        "albedo": 3,
        "height": 1,
        "normal": 3
    })

    with torch.no_grad():
        x = loss_fn.target_images.reshape(-1, 224, 224).unsqueeze(0).repeat(4, 1, 1, 1)
        x = x[:, [0, 1, 2, 3, 6, 7, 8]]
        # x[0, :3] = torch.rand_like(x[0, :3])
        input_dict = {
            'rendered_images': x
        }

        loss, loss_log, summary = loss_fn(input_dict)
    print(loss_log)
    summary['images'].show()

    print(loss_fn.target_images.shape)
