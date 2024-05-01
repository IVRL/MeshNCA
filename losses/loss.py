import torch
import numpy as np

from losses.appearance_loss import AppearanceLoss


class Loss(torch.nn.Module):
    def __init__(self, appearance_loss_weight=1.0, overflow_loss_weight=10000.0,
                 clip_loss_weight=0.0, motion_loss_weight=0.0,
                 appearance_loss_kwargs=None):
        super(Loss, self).__init__()

        self.appearance_loss_weight = appearance_loss_weight
        self.clip_loss_weight = clip_loss_weight
        self.motion_loss_weight = motion_loss_weight
        self.overflow_loss_weight = overflow_loss_weight

        self.appearance_loss_kwargs = appearance_loss_kwargs

        self._create_losses()

    def _create_losses(self):
        self.loss_mapper = {}
        self.loss_weights = {}

        if self.appearance_loss_weight != 0:
            self.loss_mapper['appearance'] = AppearanceLoss(**self.appearance_loss_kwargs)
            self.loss_weights['appearance'] = self.appearance_loss_weight

        if self.overflow_loss_weight != 0:
            self.loss_mapper['overflow'] = self.get_overflow_loss
            self.loss_weights['overflow'] = self.overflow_loss_weight

    def get_overflow_loss(self, input_dict, return_summary=True):
        """
        Evaluate the NCA overflow loss.
        :param input_dict:  A dictionary containing the necessary tensors for calculating the overflow loss.
                            Required keys: ['nca_state']
        :param return_summary: Not used.
        :return:
        """
        nca_state = input_dict['nca_state']
        overflow_loss = (nca_state - nca_state.clamp(-1.0, 1.0)).abs().mean()
        return overflow_loss, None, None

    def forward(self, input_dict, return_log=True, return_summary=True):
        """

        :param input_dict: A dictionary containing the necessary tensors for calculating all the losses.
        :param return_log: Whether to return the loss log dictionary.
        :param return_summary: Whether to return the summary dictionary.
        :return:
        """

        loss = 0
        loss_log_dict = {}
        summary_dict = {}
        for loss_name in self.loss_mapper:
            l, loss_log, sub_summary = self.loss_mapper[loss_name](input_dict, return_summary=return_summary)

            if loss_log is not None:
                for sub_loss_name in loss_log:
                    loss_log_dict[f'{loss_name}-{sub_loss_name}'] = loss_log[sub_loss_name].item()

            if sub_summary is not None:
                for summary_name in sub_summary:
                    summary_dict[f'{loss_name}-{summary_name}'] = sub_summary[summary_name]

            loss_log_dict[loss_name] = l.item()
            l *= self.loss_weights[loss_name]
            loss += l

        output = [loss]
        if return_log:
            output.append(loss_log_dict)
        if return_summary:
            output.append(summary_dict)
        else:
            output.append(None)
        if len(output) == 1:
            return output[0]
        else:
            return output


if __name__ == "__main__":
    appearance_loss_kwargs = {
        "target_images_path": [
            "../data/pbr_textures/Abstract_008/albedo.jpg",
            "../data/pbr_textures/Abstract_008/height.jpg",
            "../data/pbr_textures/Abstract_008/normal.jpg",
        ],
        "target_channels": [(0, 3), (3, 4), (5, 8)],
    }
    loss_fn = Loss(appearance_loss_weight=1.0, overflow_loss_weight=1.0, appearance_loss_kwargs=appearance_loss_kwargs)

    input_dict = {
        'nca_state': torch.tensor(np.random.rand(10, 10)),
        'generated_images': torch.rand(4, 9, 320, 320).to("cuda:0")
    }
    print(loss_fn(input_dict))
    print("Loss module works!")
