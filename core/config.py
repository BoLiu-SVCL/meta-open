import os
import numpy as np
import torch

from core.general_utils import AttrDict, Logger, merge_cfg_from_file


class Config(object):

    # DEFAULT VALUES
    data = AttrDict()
    data.name = 'miniimagenet'  # 'miniimagenet', 'cifar10' ('cifar10' not implemented)
    data.im_size = 224
    # 0, 1, 2
    data.augment = 0
    data.augment_val = 0
    data.augment_test = 0

    # ==============
    model = AttrDict()
    model.structure = 'resnet10'  # 'resnet10', 'resnet18', 'conv'
    model.num_classes = 64
    model.aug_scale_test = 1
    model.fold_test = 1

    # ==============
    io = AttrDict()
    io.root = 'output'
    io.exp_name = 'default'
    io.logger = None        # class object

    io.output_folder = ''
    io.model_file = ''
    io.best_model_file = ''
    io.log_file = ''
    io.log_file_test = ''

    # ==============
    ctrl = AttrDict()
    ctrl.cfg = ''
    ctrl.gpu_id = 0
    ctrl.device = 'cuda'

    ctrl.ep_save = 10
    ctrl.ep_vis_loss = 1
    ctrl.ep_val = 1
    ctrl.best_accuracy = -1.0
    ctrl.best_episode = -1

    # ==============
    train = AttrDict()
    train.batch_size = 1  # valid only in 'regular' mode
    train.batch_size_val = 1  # valid only in 'regular' mode
    train.batch_size_test = 1  # valid only in 'regular' mode
    train.nep = 1

    train.optim = 'adam'  # only 'adam' is implemented
    train.lr = 0.001
    train.weight_decay = .0005
    train.lr_policy = 'multi_step'  # only 'multi_step' is implemented
    train.lr_scheduler = [1, 2]
    train.lr_gamma = 0.5
    train.clip_grad = False

    train.mode = 'openfew'  # 'openfew', 'openmany', 'regular' (openmany not full implemented)
    train.open_detect = 'center'  # 'center', 'gauss'
    train.entropy = False
    train.aux = False

    train.loss_scale_entropy = 0.0
    train.loss_scale_aux = 0.0
    train.loss_scale_entropy_lut = [[30000], [0.5]]
    train.loss_scale_aux_lut = [[5000, 10000, 20000], [0.5, 0.3, 0.1]]

    # ==============
    fsl = AttrDict()
    fsl.n_way = 5
    fsl.n_way_val = 5
    fsl.n_way_test = 5
    fsl.k_shot = 1
    fsl.k_shot_val = 1
    fsl.k_shot_test = 1
    fsl.m_query = 15
    fsl.m_query_val = 15
    fsl.m_query_test = 15
    fsl.p_base = 75
    fsl.iterations = 100
    fsl.iterations_val = 100
    fsl.iterations_test = 100

    # ==============
    open = AttrDict()
    open.n_cls = 5
    open.n_cls_val = 5
    open.n_cls_test = 5
    open.m_sample = 1
    open.m_sample_val = 1
    open.m_sample_test = 1

    def __init__(self, cfg):
        merge_cfg_from_file(cfg, self)

    def setup(self, test=False):
        # set seed
        seed = np.random.randint(0, 10000)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

        # set up io.log_file
        # output_folder: output/mode/exp_name
        self.io.output_folder = os.path.join(
            self.io.root, self.train.mode, self.io.exp_name)
        if not os.path.exists(self.io.output_folder):
            os.makedirs(self.io.output_folder)
        self.io.log_file = os.path.join(self.io.output_folder, 'training_dynamic.txt')
        self.io.log_file_test = os.path.join(self.io.output_folder, 'testing_dynamic.txt')

        # set up logger
        if test:
            self.io.logger = Logger(log_file=self.io.log_file_test)
            self.io.logger('Writing logs into file: {}\n'.format(self.io.log_file_test), init=True)
        else:
            self.io.logger = Logger(log_file=self.io.log_file)
            self.io.logger('Writing logs into file: {}\n'.format(self.io.log_file), init=True)
        self.logger = self.io.logger

        self.io.model_file = os.path.join(self.io.output_folder, '{:s}_ep{{}}.pt'.format(self.data.name))
        self.io.best_model_file = os.path.join(self.io.output_folder, '{:s}_best.pt'.format(self.data.name))

        self.logger('gpu_id: {}\n'.format(self.ctrl.gpu_id))

        # adjust augment scale
        augment_lut = [1, 1, 10]
        fold_lut = [1, 1, 3]
        self.model.aug_scale_test = augment_lut[self.data.augment_test]
        self.model.fold_test = fold_lut[self.data.augment_test]

    def _print_attr_dict(self, k, v, indent):
        self.logger('{:s}{:s}:'.format(indent, k), quiet_ter=True)
        for _k, _v in sorted(v.items()):
            if isinstance(_v, AttrDict):
                self._print_attr_dict(_k, _v, indent=indent+'\t')
            elif isinstance(_v, Logger):
                self.logger('{:s}\t{:s}:\t\t\t{}'.format(indent, _k, 'Class object not shown here'))
            else:
                self.logger('{:s}\t{:s}:\t\t\t{}'.format(indent, _k, _v))

    def print_args(self):
        # ALL in ALL
        self.logger('CONFIGURATION BELOW')
        temp = self.__dict__
        for k in sorted(temp):
            if isinstance(temp[k], AttrDict):
                self._print_attr_dict(k, temp[k], indent='\t')
            elif isinstance(temp[k], bool):
                self.logger('\t{:s}:\t\t{}'.format(k, temp[k]), quiet_ter=True)
            else:
                self.logger('\t{:s}:\t\t{}'.format(k, 'Class object not shown here'), quiet_ter=True)
        self.logger('\n')
