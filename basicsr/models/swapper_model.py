import random
import torch
from collections import OrderedDict
from os import path as osp
from tqdm import tqdm
import numpy as np
from basicsr.archs import build_network
from basicsr.losses import build_loss
from basicsr.metrics import calculate_metric
from basicsr.utils import get_root_logger, imwrite, tensor2img
from basicsr.utils.registry import MODEL_REGISTRY
import torch.nn.functional as F
from .base_model import BaseModel
from basicsr.utils.plot import plot_batch


@MODEL_REGISTRY.register()
class SwapperModel(BaseModel):
    """Base Swapping model for single image."""

    def __init__(self, opt):
        super(SwapperModel, self).__init__(opt)

        # define network
        self.net_g = build_network(opt['network_g'])
        self.net_g = self.model_to_device(self.net_g)
        self.print_network(self.net_g)

        # load pretrained models
        load_path = self.opt['path'].get('pretrain_network_g', None)
        if load_path is not None:
            param_key = self.opt['path'].get('param_key_g', 'params')
            self.load_network(self.net_g, load_path, self.opt['path'].get('strict_load_g', True), param_key)

        if self.is_train:
            self.init_training_settings()

    def feed_data(self, data):
        self.gt = data['gt_image'].to(self.device)
        if 'id_image' in data:
            self.id_image = data['id_image'].to(self.device)
        if 'id_feature' in data:
            self.idfeature = data['id_feature'].to(self.device)

        self.b = self.gt.shape[0]
        self.randindex = [i for i in range(self.b)]

    def init_training_settings(self):
        logger = get_root_logger()
        train_opt = self.opt['train']
        self.autoswapping = train_opt.get('autoswapping', True)

        # ==================== network_g with Exponential Moving Average (EMA) =============== #
        self.ema_decay = train_opt.get('ema_decay', 0)
        if self.ema_decay > 0:
            logger.info(f'Use Exponential Moving Average with decay: {self.ema_decay}')
            # define network net_g with Exponential Moving Average (EMA)
            # net_g_ema is used only for testing on one GPU and saving
            self.net_g_ema = build_network(self.opt['network_g']).to(self.device)
            # load pretrained model
            load_path = self.opt['path'].get('pretrain_network_g', None)
            if load_path is not None:
                self.load_network(self.net_g_ema, load_path, self.opt['path'].get('strict_load_g', True), 'params_ema')
            else:
                self.model_ema(0)  # copy net_g weight
            self.net_g_ema.eval()

        else:
            raise NotImplementedError(f'Shoule have network_vqgan config or pre-calculated latent code.')

        self.net_g.train()

        # ====================== define net_d ====================== #
        if self.autoswapping:
            self.net_d = build_network(self.opt['network_d'])
            self.net_d = self.model_to_device(self.net_d)
            self.print_network(self.net_d)

            # load pretrained models
            load_path = self.opt['path'].get('pretrain_network_d', None)
            if load_path is not None:
                self.load_network(self.net_d, load_path, self.opt['path'].get('strict_load_d', True))

            self.net_d.train()

        # ====================== define losses ====================== #
        self.feat_loss_weight = train_opt.get('feat_loss_weight', 1.0)
        self.scale_adaptive_gan_weight = train_opt.get('scale_adaptive_gan_weight', 0.8)

        if train_opt.get('featmatch_opt'):
            self.cri_featmatch = build_loss(train_opt['featmatch_opt']).to(self.device)
        else:
            self.cri_featmatch = None

        if train_opt.get('perceptual_opt'):
            self.cri_perceptual = build_loss(train_opt['perceptual_opt']).to(self.device)
        else:
            self.cri_perceptual = None

        if train_opt.get('gan_opt'):
            self.cri_gan = build_loss(train_opt['gan_opt']).to(self.device)

        self.fix_generator = train_opt.get('fix_generator', True)
        logger.info(f'fix_generator: {self.fix_generator}')

        # ----------- define identity loss ----------- #
        if train_opt.get('identity_opt'):
            self.cri_identity = build_loss(train_opt['identity_opt']).to(self.device)
            # define identity network
            self.network_identity = build_network(self.opt['network_identity'])
            self.network_identity = self.model_to_device(self.network_identity)
            load_path = self.opt['path'].get('pretrain_network_identity')
            if load_path is not None:
                print('================')
                load_path = 'weights/arcface.pth'
                self.load_network(self.network_identity, load_path, True, None)

            self.network_identity.eval()
            for param in self.network_identity.parameters():
                param.requires_grad = False

        self.net_g_start_iter = train_opt.get('net_g_start_iter', 0)
        self.net_d_iters = train_opt.get('net_d_iters', 1)
        self.net_d_start_iter = train_opt.get('net_d_start_iter', 0)
        self.sample_iter = train_opt.get('sample_iter', 500)

        # set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()

    def calculate_adaptive_weight(self, recon_loss, g_loss, last_layer, disc_weight_max):
        recon_grads = torch.autograd.grad(recon_loss, last_layer, retain_graph=True)[0]
        g_grads = torch.autograd.grad(g_loss, last_layer, retain_graph=True)[0]

        d_weight = torch.norm(recon_grads) / (torch.norm(g_grads) + 1e-4)
        d_weight = torch.clamp(d_weight, 0.0, disc_weight_max).detach()
        return d_weight

    def setup_optimizers(self):
        train_opt = self.opt['train']
        # optimizer g
        optim_params_g = []
        logger = get_root_logger()

        for k, v in self.net_g.named_parameters():
            if v.requires_grad:
                optim_params_g.append(v)
                logger.warning(f'Params {k} will be optimized.')

        optim_type = train_opt['optim_g'].pop('type')
        self.optimizer_g = self.get_optimizer(optim_type, optim_params_g, **train_opt['optim_g'])
        self.optimizers.append(self.optimizer_g)

        if self.autoswapping:
            # optimizer d
            optim_params_d = []
            for k, v in self.net_d.named_parameters():
                if v.requires_grad:
                    optim_params_d.append(v)
                    logger.warning(f'Params {k} will be optimized.')

            optim_type = train_opt['optim_d'].pop('type')
            self.optimizer_d = self.get_optimizer(optim_type, optim_params_d, **train_opt['optim_d'])
            self.optimizers.append(self.optimizer_d)

    def resize_for_identity(self, out, size=112):
        out = F.interpolate(out[:, :, :448, 32:-32], (size, size), mode='bicubic', align_corners=False)
        return out

    def optimize_parameters(self, current_iter):

        l_g_total = 0
        loss_dict = OrderedDict()
        self.net_g.train()

        random.shuffle(self.randindex)
        rank = 2 + int(current_iter / 30000)

        if current_iter % rank == 0:
            img_id = self.id_image
            pair_flag = True
        else:
            img_id = self.id_image[self.randindex]
            pair_flag = False

        for interval in range(2):
            id112 = self.resize_for_identity(img_id)
            latent_id = self.network_identity(id112)
            latent_id = F.normalize(latent_id, p=2, dim=1)

            if interval:
                # ============== optimize net_d ===================
                if current_iter > self.net_d_start_iter:
                    for p in self.net_d.parameters():
                        p.requires_grad = True

                    try:
                        self.net_d.feature_network.requires_grad_(False)
                    except:
                        self.net_d.module.feature_network.requires_grad_(False)

                    self.optimizer_d.zero_grad()
                    # image fake
                    self.output, _, _ = self.net_g(self.gt, latent_id)

                    # real
                    real_d_pred, _ = self.net_d(self.gt, None)
                    l_d_real = self.cri_gan(real_d_pred, True, is_disc=True)
                    loss_dict['l_d_real'] = l_d_real
                    loss_dict['out_d_real'] = torch.mean(real_d_pred.detach())
                    l_d_real.backward()

                    # fake
                    fake_d_pred, _ = self.net_d(self.output.detach(), None)
                    l_d_fake = self.cri_gan(fake_d_pred, False, is_disc=True)
                    loss_dict['l_d_fake'] = l_d_fake
                    loss_dict['out_d_fake'] = torch.mean(fake_d_pred.detach())
                    l_d_fake.backward()

                    self.optimizer_d.step()
            else:
                # ============== optimize net_g ===================
                if current_iter > self.net_g_start_iter:

                    self.optimizer_g.zero_grad()
                    for p in self.net_d.parameters():
                        p.requires_grad = False

                    # image_fake
                    self.output, _, _ = self.net_g(self.gt, latent_id)

                    # G loss
                    gen_logits, _ = self.net_d(self.output, None)
                    l_g_gan = (-gen_logits).mean()
                    loss_dict['l_g_gan'] = l_g_gan

                    # id loss
                    out112 = self.resize_for_identity(self.output)
                    latent_fake = self.network_identity(out112)
                    latent_fake = F.normalize(latent_fake, p=2, dim=1)
                    l_identity = self.cri_identity(latent_fake, latent_id)
                    loss_dict['l_identity'] = l_identity

                    # feature loss
                    # real_feat = self.net_d.get_feature(self.gt)
                    # _ = self.net_d.module.get_feature(self.gt)
                    l_perceptual = self.cri_perceptual(self.gt, self.output)
                    loss_dict['l_perceptual'] = l_perceptual

                    l_g_total = l_g_gan + l_identity + l_perceptual

                    if pair_flag:
                        l_img_loss = self.cri_featmatch(self.output, self.gt)
                        l_g_total += l_img_loss
                        loss_dict['l_img_loss'] = l_img_loss

                    l_g_total.backward()
                    self.optimizer_g.step()

                if self.ema_decay > 0:
                    self.model_ema(decay=self.ema_decay)

            self.log_dict = self.reduce_loss_dict(loss_dict)

        if self.opt['rank'] == 0:
            ### display output images
            if (current_iter + 1) % self.sample_iter == 0:
                self.net_g_ema.eval()
                with torch.no_grad():
                    imgs = list()
                    zero_img = (torch.zeros_like(self.gt[0, ...]))
                    imgs.append(zero_img.cpu().numpy())
                    save_img = ((self.gt.cpu()) * 0.5 + 0.5).numpy()
                    for r in range(self.b):
                        imgs.append(save_img[r, ...])

                    out112 = F.interpolate(self.id_image, size=(112, 112), mode='bicubic')
                    latent_fake = self.network_identity(out112)
                    id_vector_src1 = F.normalize(latent_fake, p=2, dim=1)

                    for i in range(self.b):
                        imgs.append(save_img[i, ...])
                        image_infer = self.gt[i, ...].repeat(self.b, 1, 1, 1)
                        img_fake, _, _ = self.net_g_ema(image_infer, id_vector_src1)
                        img_fake = img_fake.cpu().numpy()
                        img_fake = img_fake * 0.5 + 0.5
                        for j in range(self.b):
                            imgs.append(img_fake[j, ...])

                    imgs = np.stack(imgs, axis=0).transpose(0, 2, 3, 1)
                    plot_batch(imgs,
                               osp.join(self.opt['path']['visualization'], 'EMA_iter_' + str(current_iter + 1) + '.jpg'))


    def test(self):
        with torch.no_grad():
            if hasattr(self, 'net_g_ema'):
                self.net_g_ema.eval()
                if self.autoswapping:
                    self.output, _, _ = self.net_g_ema(self.gt, self.idfeature, autoswapping=True, w=1.0)
                else:
                    self.output, _, _ = self.net_g_ema(self.gt, self.idfeature, autoswapping=True, w=0.0)
            else:
                logger = get_root_logger()
                logger.warning('Do not have self.net_g_ema, use self.net_g.')
                self.net_g.eval()
                if self.autoswapping:
                    self.output, _, _ = self.net_g_ema(self.gt, self.idfeature, autoswapping=True, w=1.0)
                else:
                    self.output, _, _ = self.net_g_ema(self.gt, self.idfeature, autoswapping=True, w=0.0)
                self.net_g.train()

    def dist_validation(self, dataloader, current_iter, tb_logger, save_img):
        if self.opt['rank'] == 0:
            self.nondist_validation(dataloader, current_iter, tb_logger, save_img)

    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img):
        dataset_name = dataloader.dataset.opt['name']
        with_metrics = self.opt['val'].get('metrics') is not None
        if with_metrics:
            self.metric_results = {metric: 0 for metric in self.opt['val']['metrics'].keys()}
        pbar = tqdm(total=len(dataloader), unit='image')

        for idx, val_data in enumerate(dataloader):
            img_name = osp.splitext(osp.basename(val_data['gt_path'][0]))[0]
            self.feed_data(val_data)
            self.test()

            visuals = self.get_current_visuals()
            swapping_img = tensor2img([visuals['result']], min_max=(-1, 1))
            if 'gt_image' in visuals:
                gt_image = tensor2img([visuals['gt_image']], min_max=(-1, 1))
                del self.gt

            del self.idfeature
            # tentative for out of GPU memory
            del self.output
            torch.cuda.empty_cache()

            if with_metrics:
                # calculate metrics
                for name, opt_ in self.opt['val']['metrics'].items():
                    metric_data = dict(img1=swapping_img, img2=gt_image)
                    self.metric_results[name] += calculate_metric(metric_data, opt_)

            if save_img:
                if self.opt['is_train']:
                    save_img_path = osp.join(self.opt['path']['visualization'], img_name,
                                             f'{img_name}_{current_iter}.png')
                else:
                    if self.opt['val']['suffix']:
                        save_img_path = osp.join(self.opt['path']['visualization'], dataset_name,
                                                 f'{img_name}_{self.opt["val"]["suffix"]}.png')
                    else:
                        save_img_path = osp.join(self.opt['path']['visualization'], dataset_name,
                                                 f'{img_name}_{self.opt["name"]}.png')
                swapping_img = np.concatenate([gt_image, swapping_img], axis=1)
                imwrite(swapping_img, save_img_path)

            pbar.update(1)
            pbar.set_description(f'Test {img_name}')
        pbar.close()

        if with_metrics:
            for metric in self.metric_results.keys():
                self.metric_results[metric] /= (idx + 1)

            self._log_validation_metric_values(current_iter, dataset_name, tb_logger)

    def _log_validation_metric_values(self, current_iter, dataset_name, tb_logger):
        log_str = f'Validation {dataset_name}\n'
        for metric, value in self.metric_results.items():
            log_str += f'\t # {metric}: {value:.4f}\n'
        logger = get_root_logger()
        logger.info(log_str)
        if tb_logger:
            for metric, value in self.metric_results.items():
                tb_logger.add_scalar(f'metrics/{metric}', value, current_iter)

    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict['gt_image'] = self.gt.detach().cpu()
        out_dict['result'] = self.output.detach().cpu()
        return out_dict

    def save(self, epoch, current_iter):
        if self.ema_decay > 0:
            self.save_network([self.net_g, self.net_g_ema], 'net_g', current_iter, param_key=['params', 'params_ema'])
        else:
            self.save_network(self.net_g, 'net_g', current_iter)
        self.save_network(self.net_d, 'net_d', current_iter)
        self.save_training_state(epoch, current_iter)