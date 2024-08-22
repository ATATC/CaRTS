import torch
import torchvision.transforms as T
from torch.nn import BCELoss
from torch.optim import SGD
from torch.optim.lr_scheduler import StepLR

from augmentation import wrap, Smoke
from datasets.transformation.autoaugment import AutoAugment

transform = T.Compose([
    T.ToTensor(),
    T.Resize((270, 480), interpolation=T.InterpolationMode.NEAREST)
])


class UNetProjNura(object):
    train_dataset = dict(
        name="SegSTRONGC",
        args=dict(
            root_folder='/workspace/data/SegSTRONGC_release',
            split='train',
            set_indices=[3, 4, 5, 7, 8],
            subset_indices=[[0, 2], [0, 1, 2], [0, 2], [0, 1], [1, 2]],
            domains=['regular'],
            image_transforms=[transform, lambda x : x.to(torch.uint8), wrap(Smoke(p=0.5)), AutoAugment, lambda x: x.to(torch.float)],
            gt_transforms=[True, False, False, False, False]
        )
    )
    validation_dataset = dict(
        name="SegSTRONGC",
        args=dict(
            root_folder='/workspace/data/SegSTRONGC_release',
            split='val',
            set_indices=[1],
            subset_indices=[[0, 1, 2]],
            domains=['regular'],
            image_transforms=[transform],
            gt_transforms=[True], ))
    test_dataset = dict(
        name="SegSTRONGC",
        args=dict(
            root_folder='/workspace/data/SegSTRONGC_release',
            split='test',
            set_indices=[9],
            subset_indices=[[0, 1, 2]],
            domains=['regular'],
            image_transforms=[transform],
            gt_transforms=[True], ))
    model = dict(
        name="Unet",
        params=dict(
            input_dim=3,
            hidden_dims=[512, 256, 128, 64, 32],
            size=(15, 20),
            target_size=(270, 480),
            criterion=BCELoss(),
            train_params=dict(
                perturbation=None,
                lr_scheduler=dict(
                    lr_scheduler_class=StepLR,
                    args=dict(
                        step_size=5,
                        gamma=0.1)),
                optimizer=dict(
                    optim_class=SGD,
                    args=dict(
                        lr=0.01,
                        momentum=0.9,
                        weight_decay=10e-5)),
                max_epoch_number=40,
                save_interval=5,
                save_path='/checkpoints/unet_projnura/',
                log_interval=50)))
