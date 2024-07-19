import os
import cv2
import argparse
import glob
import torch
import numpy as np
import torch.nn.functional as F
from torchvision.transforms.functional import normalize
from basicsr.utils import imwrite, img2tensor, tensor2img
from basicsr.utils.misc import get_device
from basicsr.utils.registry import ARCH_REGISTRY
from insightface.app import FaceAnalysis
from facelib.utils.face_restoration_helper import FaceRestoreHelper

ckpt_path = '============='
from copy import deepcopy
def load_network(net, load_path, strict=True, param_key='params'):
    """Load network.

    Args:
        load_path (str): The path of networks to be loaded.
        net (nn.Module): Network.
        strict (bool): Whether strictly loaded.
        param_key (str): The parameter key of loaded network. If set to
            None, use the root 'path'.
            Default: 'params'.
    """
    load_net = torch.load(load_path, map_location=lambda storage, loc: storage)
    if param_key is not None:
        if param_key not in load_net and 'params' in load_net:
            param_key = 'params'
        load_net = load_net[param_key]
    # remove unnecessary 'module.'
    for k, v in deepcopy(load_net).items():
        if k.startswith('module.'):
            load_net[k[7:]] = v
            load_net.pop(k)
    net.load_state_dict(load_net, strict=strict)
    return net

if __name__ == '__main__':

    device = get_device()
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_path', type=str, default='./inputs/suppswap',  help='Input image or folder. Default: inputs/gray_faces')
    parser.add_argument('-r', '--reference_path', type=str, default='./inputs/suppid',  help='Input ids or folder. Default: inputs/gray_faces')
    parser.add_argument('-o', '--output_path', type=str, default=None,  help='Output folder. Default: results/<input_name>')
    parser.add_argument('--suffix', type=str, default=None,  help='Suffix of the restored faces. Default: None')
    args = parser.parse_args()

    # ------------------------ input & output ------------------------
    print('[NOTE] The input face images should be aligned and cropped tyuh+ o a resolution of 512x512.')
    if args.input_path.endswith(('jpg', 'jpeg', 'png', 'JPG', 'JPEG', 'PNG')): # input single img path
        input_img_list = [args.input_path]
        result_root = f'results/test_swapper_img'
    else: # input img folder
        if args.input_path.endswith('/'):  # solve when path ends with /
            args.input_path = args.input_path[:-1]
        # scan all the jpg and png images
        input_img_list = sorted(glob.glob(os.path.join(args.input_path, '*.[jpJP][pnPN]*[gG]')))
        result_root = f'results/{os.path.basename(args.input_path)}'

    if args.reference_path.endswith('[jpJP][pnPN]*[gG]'): # input single img path
        ref_img_list = [args.reference_path]
    else: # input img folder
        if args.reference_path.endswith('/'):  # solve when path ends with /
            args.reference_path = args.reference_path[:-1]
        # scan all the jpg and png images
        ref_img_list = sorted(glob.glob(os.path.join(args.reference_path, '*.[jpJP][pnPN]*[gG]')))

    if not args.output_path is None: # set output path
        result_root = args.output_path

    test_img_num = len(input_img_list)
    net = ARCH_REGISTRY.get('SwapperR1')(dim_embd=256, codebook_size=1024, n_layers=9, connect_list=['16', '32', '64', '128']).to(device)
    # net = ARCH_REGISTRY.get('SwapperR1')(dim_embd=256, codebook_size=1024, n_layers=9, connect_list=['16', '32', '64']).to(device)
    checkpoint = torch.load(ckpt_path)['params_ema']
    net.load_state_dict(checkpoint, strict=False)
    net.eval()

    # define identity network
    net_identity = ARCH_REGISTRY.get('IR100ArcFace')(block='IBasicBlock', layers=[3, 13, 30, 3]).to(device)
    load_network(net_identity, 'weights/arcface.pth', True, None)
    net_identity.eval()

    face_helper = FaceRestoreHelper(
        1,
        face_size=512,
        crop_ratio=(1, 1),
        det_model='retinaface_resnet50',
        save_ext='png',
        use_parse=True,
        device=device)

    has_aligned = False
    for j, ref_path in enumerate(ref_img_list):
        if has_aligned:
            Ref_img = cv2.imread(ref_path)
            # the input faces are already cropped and aligned
            Ref_img = cv2.resize(Ref_img, (512, 512))
            face_helper.cropped_faces = [Ref_img]
        else:
            face_helper.read_image(ref_path)
            # get face landmarks for each face
            num_det_faces = face_helper.get_face_landmarks_5(
                only_center_face=False, resize=640, eye_dist_threshold=5)
            print(f'\tdetect {num_det_faces} faces')
            # align and warp each face
            face_helper.align_warp_face()

        Ref_img = face_helper.cropped_faces[-1]
        id_img_name = os.path.basename(ref_path)
        id_basename, ext = os.path.splitext(id_img_name)
        # Ref_faces = app.get(Ref_img)

        Ref_face = img2tensor(Ref_img / 255., bgr2rgb=True, float32=True)
        normalize(Ref_face, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
        Ref_face = Ref_face.unsqueeze(0).to(device)
        id112 = F.interpolate(Ref_face[:, :, :448, 32:-32], (112, 112), mode='bicubic', align_corners=False)
        latent_id = net_identity(id112)
        latent_id = F.normalize(latent_id, p=2, dim=1)

        # -------------------- start to processing ---------------------
        for i, img_path in enumerate(input_img_list):
            img_name = os.path.basename(img_path)
            basename, ext = os.path.splitext(img_name)
            print(f'[{i+1}/{test_img_num}] Processing: {img_name}')
            
            input_face = cv2.imread(img_path)
            if not has_aligned:
                face_helper.read_image(img_path)
                # get face landmarks for each face
                num_det_faces = face_helper.get_face_landmarks_5(
                    only_center_face=False, resize=640, eye_dist_threshold=5)
                print(f'\tdetect {num_det_faces} faces')
                # align and warp each face
                face_helper.align_warp_face()
                input_face = face_helper.cropped_faces[-1]

            # input_face = cv2.resize(input_face, (512, 512), interpolation=cv2.INTER_LINEAR)
            assert input_face.shape[:2] == (512, 512), 'Input resolution must be 512x512 for colorization.'
            
            input_face = img2tensor(input_face / 255., bgr2rgb=True, float32=True)
            normalize(input_face, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
            input_face = input_face.unsqueeze(0).to(device)
            try:
                with torch.no_grad():
                    # w is fixed to 0 since we didn't train the Stage III for colorization
                    output_face = net(input_face, latent_id)[0] 
                    save_face = tensor2img(output_face, rgb2bgr=True, min_max=(-1, 1))
                del output_face
                torch.cuda.empty_cache()
            except Exception as error:
                print(f'\tFailed inference for VQSwapper: {error}')
                save_face = tensor2img(input_face, rgb2bgr=True, min_max=(-1, 1))

            save_face = save_face.astype('uint8')

            # save face
            if args.suffix is not None:
                basename = f'{basename}_{args.suffix}'
            save_restore_path = os.path.join(result_root, f'{id_basename}_{basename}.png')
            imwrite(save_face, save_restore_path)

    print(f'\nAll results are saved in {result_root}')

