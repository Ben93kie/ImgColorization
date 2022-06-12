import random

import torch
import torchvision
from torchvision.transforms import functional as F


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target, meta=None):

        for t in self.transforms:
            if meta != None and (isinstance(t, Resize) or isinstance(t, RandomCrop)):
                image, target = t(image, target,meta)
            else:
                image, target = t(image, target)
        return image, target

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string


class Resize(object):
    def __init__(self, min_size, max_size):
        if not isinstance(min_size, (list, tuple)):
            min_size = (min_size,)
        self.min_size = min_size
        self.max_size = max_size


    # modified from torchvision to add support for max size
    def get_size(self, image_size, meta):
        w, h = image_size
        # print("meta: ", meta)
        # print("min_size ", self.min_size)
        if meta != None and type(meta) == str :
            if 'high' in meta:
                min_size = self.min_size[2]
                max_size = self.max_size[2]
            elif 'medium' in meta:
                min_size = self.min_size[1]
                max_size = self.max_size[1]
            else:
                min_size = self.min_size[0]
                max_size = self.max_size[0]
        elif type(meta) == float:
            min_size=int(meta)
            max_size=None
            # print(min_size)
        else:
            min_size=self.min_size
            max_size = self.max_size


        if not isinstance(min_size, (list, tuple)):
            min_size = (min_size,)
        # print(self.)
        size = random.choice(min_size)
        # max_size = self.max_size
        if max_size is not None:
            min_original_size = float(min((w, h)))
            max_original_size = float(max((w, h)))
            if max_original_size / min_original_size * size > max_size:
                size = int(round(max_size * min_original_size / max_original_size))

        if (w <= h and w == size) or (h <= w and h == size):
            return (h, w)

        if w < h:
            ow = size
            oh = int(size * h / w)
        else:
            oh = size
            ow = int(size * w / h)

        return (oh, ow)

    def __call__(self, image, target=None, meta=None):
        size = self.get_size(image.size, meta)
        # image.show()
        # print("vorher",image.size)



        image = F.resize(image, size)
        # image.show()
        # print("nachher nach image resizing",image.size)
        # sys.exit()
        if target is None:
            return image
        target = target.resize(image.size)

        return image, target

class RandomCrop(object):
    def __init__(self, min_size, max_size):
        if not isinstance(min_size, (list, tuple)):
            min_size = (min_size,)
        #min_size = (min(256, min_size[0]), )
        self.min_size = min_size
        self.max_size = max_size

    # modified from torchvision to add support for max size
    def get_size(self, image_size):
        w, h = image_size
        # size = random.choice(self.min_size)
        size = random.randint(self.min_size[0], min(self.max_size,w))
        max_size = self.max_size
        if max_size is not None:
            min_original_size = float(min((w, h)))
            max_original_size = float(max((w, h)))
            if max_original_size / min_original_size * size > max_size:
                size = int(round(max_size * min_original_size / max_original_size))

        if (w <= h and w == size) or (h <= w and h == size):
            return (h, w)

        if w < h:
            ow = size
            oh = int(size * h / w)
        else:
            oh = size
            ow = int(size * w / h)

        return (oh, ow)

    def __call__(self, image, target, meta=None):
        """-
        :param image:
        :param target:
        :return:
        """

        width, height = image.size

        if meta != None and type(meta) == float:
            random_cropped_width = min(1000, width)
            resize_factor = random_cropped_width / width
            random_cropped_height = min(1000, height)

        else:

            random_cropped_width = random.randint(self.min_size[0],max(self.min_size[0],width))
            resize_factor = random_cropped_width / width
            random_cropped_height = int(height * resize_factor)

        rand_h = random.randint(0,max(0, height-random_cropped_height))
        rand_w = random.randint(0,max(0, width-random_cropped_width))

        if meta != None and type(meta) == float:
            image = F.crop(image, rand_h, rand_w, random_cropped_height, random_cropped_width)
        else:
            image = F.crop(image, rand_h, rand_w, random_cropped_height, random_cropped_width)

        target = target.crop([rand_w, rand_h, image.size[0]+rand_w, image.size[1]+rand_h])

        return image, target


class RandomHorizontalFlip(object):
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            image = F.hflip(image)
            target = target.transpose(0)
        return image, target

class RandomVerticalFlip(object):
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            image = F.vflip(image)
            target = target.transpose(1)
        return image, target

class ColorJitter(object):
    def __init__(self,
                 brightness=None,
                 contrast=None,
                 saturation=None,
                 hue=None,
                 ):
        self.color_jitter = torchvision.transforms.ColorJitter(
            brightness=brightness,
            contrast=contrast,
            saturation=saturation,
            hue=hue,)

    def __call__(self, image, target, meta=None):
        image = self.color_jitter(image)
        return image, target


class ToTensor(object):
    def __call__(self, image, target):
        return F.to_tensor(image), target


class Normalize(object):
    def __init__(self, mean, std, to_bgr255=True):
        self.mean = mean
        self.std = std
        self.to_bgr255 = to_bgr255

    def __call__(self, image, target=None):
        if self.to_bgr255:
            image = image[[2, 1, 0]] * 255
        image = F.normalize(image, mean=self.mean, std=self.std)
        if target is None:
            return image
        return image, target
