from . import transforms as T


def build_transforms(cfg, is_train=True):
    if is_train:
        min_size = cfg.INPUT.MIN_SIZE_TRAIN
        max_size = cfg.INPUT.MAX_SIZE_TRAIN
        flip_horizontal_prob = cfg.INPUT.HORIZONTAL_FLIP_PROB_TRAIN
        flip_vertical_prob = cfg.INPUT.VERTICAL_FLIP_PROB_TRAIN
        brightness = cfg.INPUT.BRIGHTNESS
        contrast = cfg.INPUT.CONTRAST
        saturation = cfg.INPUT.SATURATION
        hue = cfg.INPUT.HUE
    else:

        if cfg.INPUT.MIN_SIZE_TEST_LOW > 0:
            min_size = [cfg.INPUT.MIN_SIZE_TEST_LOW,cfg.INPUT.MIN_SIZE_TEST_MEDIUM,cfg.INPUT.MIN_SIZE_TEST_HIGH]
            max_size = [cfg.INPUT.MAX_SIZE_TEST_LOW,cfg.INPUT.MAX_SIZE_TEST_MEDIUM,cfg.INPUT.MAX_SIZE_TEST_HIGH]
        else:
            min_size = cfg.INPUT.MIN_SIZE_TEST
            max_size = cfg.INPUT.MAX_SIZE_TEST


        flip_horizontal_prob = 0.0
        flip_vertical_prob = 0.0
        brightness = 0.0
        contrast = 0.0
        saturation = 0.0
        hue = 0.0

    to_bgr255 = cfg.INPUT.TO_BGR255 and cfg.INPUT.COLOR_SPACE == 'RGB'


    normalize_transform = T.Normalize(
        mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD, to_bgr255=to_bgr255
    )
    color_jitter = T.ColorJitter(
        brightness=brightness,
        contrast=contrast,
        saturation=saturation,
        hue=hue,
    )
    if is_train and cfg.INPUT.CROPPING and not cfg.INPUT.ADAPTIVE_RESIZER:
        transform = T.Compose(
            [
                T.RandomCrop(cfg.INPUT.CROPPING_MIN,cfg.INPUT.CROPPING_MAX),
                color_jitter,
                T.Resize(min_size, max_size),
                T.RandomHorizontalFlip(flip_horizontal_prob),
                T.RandomVerticalFlip(flip_vertical_prob),
                T.ToTensor(),
                normalize_transform,
            ]
        )
    elif is_train and cfg.INPUT.CROPPING and cfg.INPUT.ADAPTIVE_RESIZER:
        transform = T.Compose(
            [
                color_jitter,
                T.Resize(min_size, max_size),
                T.RandomCrop(min_size, min_size),
                T.RandomHorizontalFlip(flip_horizontal_prob),
                T.RandomVerticalFlip(flip_vertical_prob),
                T.ToTensor(),
                normalize_transform,
            ]
        )
    elif is_train:
        transform = T.Compose(
            [
                color_jitter,
                T.Resize(min_size, max_size),
                T.RandomHorizontalFlip(flip_horizontal_prob),
                T.RandomVerticalFlip(flip_vertical_prob),
                T.ToTensor(),
                normalize_transform,
            ]
        )
    else:
        transform = T.Compose(
            [
                T.Resize(min_size, max_size),
                T.ToTensor(),
                normalize_transform
            ]
        )
    return transform
