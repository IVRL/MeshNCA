import torch
from PIL import Image
import numpy as np
from torchvision import transforms


def load_texture_image(img_path, img_size=(256, 256)):
    """
    Load a texture image and resize it to the desired size
    :return: [1, 3, H, W] tensor and the PIL image
    """
    style_img = Image.open(img_path).convert("RGB")
    w, h = style_img.size
    if w == h:
        style_img = style_img.resize(img_size)
        style_img = style_img.convert('RGB')
    else:
        style_img = style_img.convert('RGB')
        style_img = np.array(style_img)
        h, w, _ = style_img.shape

        ## Center crop the image
        cut_pixel = abs(w - h) // 2
        if w > h:
            style_img = style_img[:, cut_pixel:w - cut_pixel, :]
        else:
            style_img = style_img[cut_pixel:h - cut_pixel, :, :]
        style_img = Image.fromarray(style_img.astype(np.uint8))
        style_img = style_img.resize(img_size)

    with torch.no_grad():
        img_tensor = transforms.ToTensor()(style_img).unsqueeze(0)

    return img_tensor, style_img


if __name__ == '__main__':
    x, pil_img = load_texture_image("../data/textures/bubbly_0101.jpg")
    print(x.shape)

    pil_img.show()
