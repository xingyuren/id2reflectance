import glob
import random
import os.path as osp
import torch.utils.data as data
from torchvision.transforms.functional import normalize

from basicsr.utils import FileClient, get_root_logger, imfrombytes, img2tensor
from basicsr.utils.registry import DATASET_REGISTRY


@DATASET_REGISTRY.register()
class VGGHQFSDataset(data.Dataset):

    def __init__(self, opt):
        super(VGGHQFSDataset, self).__init__()
        self.logger = get_root_logger()
        self.opt = opt
        self.io_backend_opt = opt['io_backend']

        self.gt_size = opt.get('gt_size', 512)
        self.in_size = opt.get('in_size', 512)
        assert self.gt_size >= self.in_size, 'Wrong setting.'

        self.mean = opt.get('mean', [0.5, 0.5, 0.5])
        self.std = opt.get('std', [0.5, 0.5, 0.5])

        self.image_dir = opt['dataroot_gt']
        self.dataset = []
        self.random_seed = 1234
        self.preprocess()
        self.num_images = len(self.dataset)

        self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)


    def preprocess(self):
        """Preprocess the Swapping dataset."""
        self.logger.info("processing Swapping dataset images...")

        temp_path = osp.join(self.image_dir, '*/')
        pathes = glob.glob(temp_path)
        self.dataset = []
        for dir_item in pathes:
            join_path = glob.glob(osp.join(dir_item, '*.jpg'))
            # self.logger.info("processing %s" % dir_item)
            temp_list = []
            for item in join_path:
                temp_list.append(item)
            self.dataset.append(temp_list)
        random.seed(self.random_seed)
        random.shuffle(self.dataset)
        self.logger.info('Finished preprocessing the VGGFaceHQ Swapping dataset, total dirs number: %d...' % len(self.dataset))

    def __getitem__(self, index):
        dir_tmp1 = self.dataset[index]
        dir_tmp1_len = len(dir_tmp1)

        filename1 = dir_tmp1[random.randint(0, dir_tmp1_len - 1)]
        filename2 = dir_tmp1[random.randint(0, dir_tmp1_len - 1)]

        # load gt image
        img_bytes1 = self.file_client.get(filename1)
        image1 = imfrombytes(img_bytes1, float32=True)

        img_bytes2 = self.file_client.get(filename2)
        image2 = imfrombytes(img_bytes2, float32=True)

        # BGR to RGB, HWC to CHW, numpy to tensor
        image1, image2 = img2tensor([image1, image2], bgr2rgb=True, float32=True)
        normalize(image1, self.mean, self.std, inplace=True)
        normalize(image2, self.mean, self.std, inplace=True)
        return_dict = {'gt_image': image1, 'id_image': image2}

        return return_dict

    def __len__(self):
        return self.num_images