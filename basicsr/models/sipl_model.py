import torch
from torch.nn import functional as F
from collections import OrderedDict
from basicsr.utils.dist_util import get_dist_info
from tqdm import tqdm
from os import path as osp
import numpy as np
from basicsr.utils.dist_util import master_only
from basicsr.utils.registry import MODEL_REGISTRY
from .sr_model import SRModel
from basicsr.utils import get_root_logger, imwrite, tensor2img
from copy import deepcopy
from basicsr.metrics import calculate_metric


@MODEL_REGISTRY.register()
class SIPLModel(SRModel):

    @master_only
    def calculate_flops(self, input_dim=(3, 504, 504)):
        super().calculate_flops(input_dim=input_dim)

    def test(self):
        # pad to multiplication of window_size
        window_size = self.opt["network_g"].get("window_size", 16)
        temp_size = self.opt["val"].get("window_size", 16)
        window_size = max(window_size) if isinstance(
            window_size, list) else window_size
        window_size = max(temp_size, window_size)

        # re-padding image size with multi-scale window size
        if not isinstance(window_size, int):
            max_value = max(window_size)
            if not isinstance(max_value, int):
                max_value = max(max_value)
            if (max_value == 8) and (6 in window_size):
                window_size = 24
            else:
                window_size = max_value
        # window_size=16 #evaluation for Shift layer window size
        scale = self.opt.get("scale", 1)
        mod_pad_h, mod_pad_w = 0, 0
        _, _, h, w = self.lq.size()
        if h % window_size != 0:
            mod_pad_h = window_size - h % window_size
        if w % window_size != 0:
            mod_pad_w = window_size - w % window_size
        img = F.pad(self.lq, (0, mod_pad_w, 0, mod_pad_h), "reflect")

        if hasattr(self, "net_g_ema"):
            self.net_g_ema.eval()
            with torch.no_grad():
                for i in range(self.opt.get("more_iter_num", 2)):
                    output = self.net_g_ema(img)
                    privileged_features = self.net_g_ema(
                        output, return_feature=True)
                    privileged_weights = [0.1, 0.1, 0.1]
                    self.output = self.net_g_ema(
                        img, privileged_features=privileged_features, privileged_weights=privileged_weights)
                    img = self.output
        else:
            self.net_g.eval()
            with torch.no_grad():
                for i in range(self.opt.get("more_iter_num", 2)):
                    output = self.net_g(img)
                    privileged_features = self.net_g(
                        output, return_feature=True)
                    privileged_weights = [0.1, 0.1, 0.1]
                    self.output = self.net_g(
                        img, privileged_features=privileged_features, privileged_weights=privileged_weights)
                    img = self.output

            self.net_g.train()

        _, _, h, w = self.output.size()

        self.output = self.output[
            :, :, 0: h - mod_pad_h * scale, 0: w - mod_pad_w * scale
        ]

    def optimize_parameters(self, current_iter=None, tb_logger=None):
        self.optimizer_g.zero_grad()

        with torch.no_grad():
            output = self.net_g(self.gt)
            privileged_features = self.net_g(output, return_feature=True)
            privileged_weights = [0.1, 0.1, 0.1]

        self.output = self.net_g(
            self.lq, privileged_features=privileged_features, privileged_weights=privileged_weights)

        l_total = 0
        loss_dict = OrderedDict()
        # pixel loss
        if self.cri_pix:
            l_pix = self.cri_pix(self.output, self.gt)
            l_total += l_pix
            loss_dict['l_pix'] = l_pix
        # perceptual loss
        if self.cri_perceptual:
            l_percep, l_style = self.cri_perceptual(self.output, self.gt)
            if l_percep is not None:
                l_total += l_percep
                loss_dict['l_percep'] = l_percep
            if l_style is not None:
                l_total += l_style
                loss_dict['l_style'] = l_style

        l_total.backward()

        if self.DEBUG:
            for name, param in self.net_g.named_parameters():
                if param.grad is None:
                    print(name)

        if self.opt['rank'] == 0:
            if tb_logger:
                with torch.no_grad():
                    std_v, mean_v = torch.std_mean(
                        2.0*(self.output-self.gt), dim=[-3, -2, -1])
                    std_v = std_v.mean()
                    mean_v = mean_v.mean()
                tb_logger.add_scalar(f'train/grad/mean', mean_v, current_iter)
                tb_logger.add_scalar(f'train/grad/std', std_v, current_iter)

        self.optimizer_g.step()

        self.log_dict = self.reduce_loss_dict(
            loss_dict, current_iter=current_iter, tb_logger=tb_logger)

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)

    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img):
        self.dist_validation(dataloader, current_iter, tb_logger, save_img)
