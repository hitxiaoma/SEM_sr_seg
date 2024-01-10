import os.path as osp

from .builder import DATASETS
from .custom import CustomDataset


@DATASETS.register_module()
class PascalVOCDataset(CustomDataset):
    """Pascal VOC dataset.

    Args:
        split (str): Split txt file for Pascal VOC.
    """
    # CLASSES = ('background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
    #            'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog',
    #            'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa',
    #            'train', 'tvmonitor')
    #
    # PALETTE = [[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0], [0, 0, 128],
    #            [128, 0, 128], [0, 128, 128], [128, 128, 128], [64, 0, 0],
    #            [192, 0, 0], [64, 128, 0], [192, 128, 0], [64, 0, 128],
    #            [192, 0, 128], [64, 128, 128], [192, 128, 128], [0, 64, 0],
    #            [128, 64, 0], [0, 192, 0], [128, 192, 0], [0, 64, 128]]

    # CLASSES = ('others', 'forest', 'road', 'building', 'water')
    CLASSES = ('Pores', 'Hydrates', 'Unhydrated')
    PALETTE = [(85, 83, 252), (101, 225, 250), (162, 78, 0)]
    # PALETTE = [[242, 234, 218], [69, 185, 124], [143, 75, 46], [116, 120, 124], [51, 163, 220]]

    def __init__(self, split, **kwargs):
        super(PascalVOCDataset, self).__init__(
            img_suffix='.jpg', seg_map_suffix='.png', split=split, **kwargs)
        assert osp.exists(self.img_dir) and self.split is not None
