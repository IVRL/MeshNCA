import torch
import numpy as np


def symmetric_padding(im, padding):
    h, w = im.shape[-2:]
    left, right, top, bottom = padding

    x_idx = np.arange(-left, w + right)
    y_idx = np.arange(-top, h + bottom)

    def reflect(x, minx, maxx):
        """ Reflects an array around two points making a triangular waveform that ramps up
        and down,  allowing for pad lengths greater than the input length """
        rng = maxx - minx
        double_rng = 2 * rng
        mod = np.fmod(x - minx, double_rng)
        normed_mod = np.where(mod < 0, mod + double_rng, mod)
        out = np.where(normed_mod >= rng, double_rng - normed_mod, normed_mod) + minx
        return np.array(out, dtype=x.dtype)

    x_pad = reflect(x_idx, -0.5, w - 0.5)
    y_pad = reflect(y_idx, -0.5, h - 0.5)
    xx, yy = np.meshgrid(x_pad, y_pad)
    return im[..., yy, xx]


def tf_consistent_bilinear_upsample(imgs, scale_factor=1.0):
    b, c, h, w = imgs.shape
    assert h == w

    N = int(h * scale_factor)
    delta = (1.0 / h)
    p = int(scale_factor) - 1

    xs = torch.linspace(-1.0 + delta, 1.0 - delta, N - p).to(imgs)
    ys = torch.linspace(-1.0 + delta, 1.0 - delta, N - p).to(imgs)
    grid = torch.meshgrid(xs, ys)
    gridy = grid[1]
    gridx = grid[0]
    gridx = torch.nn.functional.pad(gridx.unsqueeze(0), (0, p, 0, p), mode='replicate')[0]
    gridy = torch.nn.functional.pad(gridy.unsqueeze(0), (0, p, 0, p), mode='replicate')[0]
    grid = torch.stack([gridy, gridx], dim=-1).unsqueeze(0).repeat(b, 1, 1, 1)
    output = torch.nn.functional.grid_sample(imgs, grid, mode='bilinear', padding_mode='zeros')

    return output


class GaussianBlur(torch.nn.Module):
    def __init__(self, kernel_size, stride, sigma):
        super().__init__()
        self.kernel_size = kernel_size
        kernel_weights = self.gauss2d_kernel((kernel_size, kernel_size), sigma).astype(np.float32)
        self.kernel = torch.from_numpy(kernel_weights.reshape(1, 1, kernel_size, kernel_size))

    def forward(self, x):
        # x has the shape [B, 1, H, W, 2]
        x1 = x[..., 0]
        x2 = x[..., 1]

        p = self.kernel_size // 2
        x1 = symmetric_padding(x1, (p, p, p, p))
        x2 = symmetric_padding(x2, (p, p, p, p))

        self.kernel = self.kernel.to(x1)
        x1 = torch.nn.functional.conv2d(x1, weight=self.kernel, stride=2)
        x2 = torch.nn.functional.conv2d(x2, weight=self.kernel, stride=2)

        x = torch.stack([x1, x2], dim=-1)
        return x

    def gauss2d_kernel(self, shape=(3, 3), sigma=0.5):
        """
        2D gaussian mask - should give the same result as MATLAB's
        fspecial('gaussian',[shape],[sigma])
        """
        m, n = [(ss - 1.) / 2. for ss in shape]
        y, x = np.ogrid[-m:m + 1, -n:n + 1]
        h = np.exp(-(x * x + y * y) / (2. * sigma * sigma))
        h[h < np.finfo(h.dtype).eps * h.max()] = 0
        sumh = h.sum()
        if sumh != 0:
            h /= sumh
        return h


class MSOEnet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv3d(1, 32, (11, 11, 2))
        self.maxpool = torch.nn.MaxPool2d(5, stride=1, padding=2)
        self.conv2 = torch.nn.Conv2d(32, 64, 1)

    def l1_normalize(self, x):
        eps = torch.tensor(1e-12)
        norm = torch.sum(torch.abs(x), dim=1, keepdim=True)
        return x / (torch.maximum(norm, eps))

    def forward(self, x):
        # x has the shape [B, 1, H, W, 2]
        x0 = x[..., 0]
        x1 = x[..., 1]
        x0 = symmetric_padding(x0, (5, 5, 5, 5))
        x1 = symmetric_padding(x1, (5, 5, 5, 5))
        x = torch.stack([x0, x1], dim=-1)
        x = self.conv1(x)
        x = torch.square(x)
        x = x.squeeze(-1)
        x = self.maxpool(x)
        x = self.conv2(x)
        x = self.l1_normalize(x)
        # x has the shape [B, 64, H, W]
        return x


class MSOEmultiscale(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.n_scales = 5
        self.msoenet = MSOEnet()

        self.gaussian_blur = GaussianBlur(kernel_size=5, stride=2, sigma=2.0)

        self.decode_conv1 = torch.nn.Conv2d(64 * self.n_scales, 64, 3)
        self.decode_conv2 = torch.nn.Conv2d(64, 2, 1)

    def contrast_norm(self, x):
        # x has the shape [B, 1, H, W, 2]
        eps = 1e-12
        x_mean = torch.mean(x, dim=(1, 2, 3, 4), keepdim=True)

        x_var = torch.var(x, dim=(1, 2, 3, 4), keepdim=True)
        x_std = torch.sqrt(x_var + eps)
        x = (x - x_mean) / x_std

        return x

    def forward(self, x, return_features=False):
        features = []
        # x has the shape [B, 1, H, W, 2]
        x0 = self.contrast_norm(x)
        h0 = self.msoenet(x0)

        x1 = self.gaussian_blur(x0)
        h1 = self.msoenet(x1)

        x2 = self.gaussian_blur(x1)
        h2 = self.msoenet(x2)

        x3 = self.gaussian_blur(x2)
        h3 = self.msoenet(x3)

        x4 = self.gaussian_blur(x3)
        h4 = self.msoenet(x4)

        z0 = h0
        z1 = tf_consistent_bilinear_upsample(h1, scale_factor=2.0)
        z2 = tf_consistent_bilinear_upsample(h2, scale_factor=4.0)
        z3 = tf_consistent_bilinear_upsample(h3, scale_factor=8.0)
        z4 = tf_consistent_bilinear_upsample(h4, scale_factor=16.0)

        z = torch.cat([z0, z1, z2, z3, z4], dim=1)
        features.append(z)
        z = symmetric_padding(z, (1, 1, 1, 1))
        x = self.decode_conv1(z)

        x = torch.nn.functional.relu(x)
        flow = self.decode_conv2(x)
        flow[:, 1] *= -1

        if return_features:
            return flow, features
        else:
            return flow