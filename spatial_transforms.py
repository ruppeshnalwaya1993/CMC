import random
import math
import numbers
import collections
import numpy as np
import torch
from torchvision.transforms import functional as F
from PIL import Image, ImageOps
import cv2

cv2.setNumThreads(0)

class Compose(object):
    """Composes several transforms together.
    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.
    Example:
        >>> transforms.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        offset = img[1]
        img = img[0]
        if isinstance(img, Image.Image):    # if PIL Image, convert to Numpy array
            img = np.array(img)
        for t in self.transforms:
            img = t(img)
        return (img, offset)

    def randomize_parameters(self):
        for t in self.transforms:
            if getattr(t, "randomize_parameters", None):
                t.randomize_parameters()


class GroupCompose(object):
    """Composes several transforms together.
    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.
    Example:
        >>> transforms.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img_group):
        for i,img in enumerate(img_group):
            if isinstance(img, Image.Image):    # if PIL Image, convert to Numpy array
                img_group[i] = np.array(img)
        for t in self.transforms:
            img_group = t(img_group)
        return img_group

    def randomize_parameters(self):
        for t in self.transforms:
            if getattr(t, "randomize_parameters", None):
                t.randomize_parameters()


class ToRGB2BGR(object):
    def __init__(self):
        pass
    def __call__(self, img):
        return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

class GroupToRGB2BGR(object):
    def __init__(self):
        pass
    def __call__(self, img_group):
        return [cv2.cvtColor(img, cv2.COLOR_RGB2BGR) for img in img_group] 

class ToBGR2RGB(object):
    def __init__(self):
        pass
    def __call__(self, img):
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

class GroupToBGR2RGB(object):
    def __init__(self):
        pass
    def __call__(self, img_group):
        return [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in img_group] 

class ToNormalizedTensor(object):
    def __init__(self, mean, std, norm_scale=255.0):
        self.mean = [norm_scale * m for m in mean]
        self.std = [norm_scale * s for s in std]

    def __call__(self, img):
        img = np.asarray(img, np.float32)
        img[:,:,0] = (img[:,:,0] - self.mean[0]) / self.std[0]
        img[:,:,1] = (img[:,:,1] - self.mean[1]) / self.std[1]
        img[:,:,2] = (img[:,:,2] - self.mean[2]) / self.std[2]
        return img

class GroupToNormalizedTensor(object):
    def __init__(self, mean, std, norm_scale=255.0):
        self.mean = [norm_scale * m for m in mean]
        self.std = [norm_scale * s for s in std]

    def __call__(self, img_group):
        if isinstance(img_group, list):
            out_group = []
            for img in img_group:
                img = np.asarray(img, np.float32)
                img[:,:,0] = (img[:,:,0] - self.mean[0]) / self.std[0]
                img[:,:,1] = (img[:,:,1] - self.mean[1]) / self.std[1]
                img[:,:,2] = (img[:,:,2] - self.mean[2]) / self.std[2]
                out_group.append(img)
            return out_group
        elif isinstance(img_group, torch.Tensor):
            img_group[0] = (img_group[0] - self.mean[0]) / self.std[0]
            img_group[1] = (img_group[1] - self.mean[1]) / self.std[1]
            img_group[2] = (img_group[2] - self.mean[2]) / self.std[2]
            return img_group
        else:
            print('unimplemented')
            return img_group

class RandomResizedCrop(object):
    """Crop the given PIL Image to random size and aspect ratio.
    A crop of random size (default: of 0.08 to 1.0) of the original size and a random
    aspect ratio (default: of 3/4 to 4/3) of the original aspect ratio is made. This crop
    is finally resized to given size.
    This is popularly used to train the Inception networks.
    Args:
        size: expected output size of each edge
        scale: range of size of the origin size cropped
        ratio: range of aspect ratio of the origin aspect ratio cropped
        interpolation: Default: PIL.Image.BILINEAR
    """

    def __init__(self, size, scale=(0.08, 1.0), ratio=(3. / 4., 4. / 3.), interpolation=cv2.INTER_LINEAR):
        if isinstance(size, int):
            self.size = (size, size)
        else:
            self.size = size
        self.interpolation = interpolation
        self.scale = scale
        self.ratio = ratio
        self.params = None

    @staticmethod
    def get_params(size, scale, ratio):
        """Get parameters for ``crop`` for a random sized crop.
        Args:
            img (PIL Image): Image to be cropped.
            scale (tuple): range of size of the origin size cropped
            ratio (tuple): range of aspect ratio of the origin aspect ratio cropped
        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for a random
                sized crop.
        """
        for attempt in range(10):
            area = size[0] * size[1]
            target_area = random.uniform(*scale) * area
            aspect_ratio = random.uniform(*ratio)

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if random.random() < 0.5:
                w, h = h, w

            if w <= size[0] and h <= size[1]:
                i = random.randint(0, size[1] - h)
                j = random.randint(0, size[0] - w)
                return i, j, h, w

        # Fallback
        w = min(size[0], size[1])
        i = (size[1] - w) // 2
        j = (size[0] - w) // 2
        return i, j, w, w

    def __call__(self, img):
        if isinstance(img, np.ndarray):
            if self.params == None:
                ih, iw = img.shape[:2]
                self.params = self.get_params((iw,ih), self.scale, self.ratio)
            i,j,h,w = self.params
            img = img[i:i+h,j:j+w]
            return cv2.resize(img, self.size, self.interpolation)
        else:
            if self.params == None:
                self.params = self.get_params(img.size, self.scale, self.ratio)
            return F.resized_crop(img, *self.params, self.size, self.interpolation)

    def randomize_parameters(self):
        self.params = None


class RandomCrop(object):
    def __init__(self, size):
        if isinstance(size, int):
            self.size = (size, size)
        else:
            self.size = size
        self.params = None

    @staticmethod
    def get_params(input_size, output_size):
        w, h = input_size
        th, tw = output_size
        if w == tw and h == th:
            return 0, 0, h, w

        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)
        return i, j, th, tw

    def __call__(self, img):
        if isinstance(img, np.ndarray):
            if self.params == None:
                ih, iw = img.shape[:2]
                self.params = self.get_params((iw,ih), self.size)
            i,j,h,w = self.params
            return img[i:i+h,j:j+w]
        else:
            if self.params == None:
                self.params = self.get_params(img.size, self.size)
            return F.crop(img, *self.params)

    def randomize_parameters(self):
        self.params = None


class GroupRandomCrop(object):
    def __init__(self, size, always_random=True):
        if isinstance(size, int):
            self.size = (size, size)
        else:
            self.size = size
        self.params = None
        self.always_random = always_random

    @staticmethod
    def get_params(input_size, output_size):
        w, h = input_size
        th, tw = output_size
        if w == tw and h == th:
            return 0, 0, h, w

        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)
        return i, j, th, tw

    def __call__(self, img_group):
        if self.always_random:
            self.randomize_parameters()
        out_group = []
        for img in img_group:
            if isinstance(img, np.ndarray):
                if self.params == None:
                    ih, iw = img.shape[:2]
                    self.params = self.get_params((iw,ih), self.size)
                i,j,h,w = self.params
                out_group.append(img[i:i+h,j:j+w])
            else:
                if self.params == None:
                    self.params = self.get_params(img.size, self.size)
                out_group.append(F.crop(img, *self.params))
        return out_group

    def randomize_parameters(self):
        self.params = None


class RandomResizedCrop2(object):
    """Crop the given PIL Image to random size and aspect ratio.
    A crop of random size (default: of 0.08 to 1.0) of the original size and a random
    aspect ratio (default: of 3/4 to 4/3) of the original aspect ratio is made. This crop
    is finally resized to given size.
    This is popularly used to train the Inception networks.
    Args:
        size: expected output size of each edge
        scale: range of size of the origin size cropped
        ratio: range of aspect ratio of the origin aspect ratio cropped
        interpolation: Default: PIL.Image.BILINEAR
    """

    def __init__(self, size, scale=(0.08, 1.0), ratio=(3. / 4., 4. / 3.), interpolation=cv2.INTER_LINEAR):
        if isinstance(size, int):
            self.size = (size, size)
        else:
            self.size = size
        self.interpolation = interpolation
        self.scale = scale
        self.ratio = ratio
        self.params = None

    @staticmethod
    def get_params(size, scale, ratio):
        minlen = min(size[0], size[1])
        target_scale = random.uniform(*scale)
        aspect_ratio = random.uniform(*ratio)
        w = int(round(minlen * target_scale * aspect_ratio))
        h = int(round(minlen * target_scale / aspect_ratio))

        pad_t = 0
        pad_b = 0
        pad_l = 0
        pad_r = 0
        pad_type = -1
        if size[0] >= w:
            j = random.randint(0, size[0] - w)
        else:
            j = 0
            pad_l = random.randint(0, w - size[0])
            pad_r = w - size[0] - pad_l
            pad_type = random.randint(0,2)
        if size[1] >= h:
            i = random.randint(0, size[1] - h)
        else:
            i = 0
            pad_t = random.randint(0, h - size[1])
            pad_b = h - size[1] - pad_t
            pad_type = random.randint(0,2)
        return i, j, h, w, pad_t, pad_b, pad_l, pad_r, pad_type

    def __call__(self, img):
        if isinstance(img, np.ndarray):
            if self.params == None:
                ih, iw = img.shape[:2]
                self.params = self.get_params((iw,ih), self.scale, self.ratio)
            i, j, h, w, pad_t, pad_b, pad_l, pad_r, pad_type = self.params
            if pad_type == 0:
                img = cv2.copyMakeBorder(img,pad_t,pad_b,pad_l,pad_r,cv2.BORDER_CONSTANT,value=0)
            elif pad_type == 1:
                img = cv2.copyMakeBorder(img,pad_t,pad_b,pad_l,pad_r,cv2.BORDER_REPLICATE)
            elif pad_type == 2:
                img = cv2.copyMakeBorder(img,pad_t,pad_b,pad_l,pad_r,cv2.BORDER_REFLECT_101)
            img = cv2.resize(img[i:i+h,j:j+w], self.size, self.interpolation)
            return img
        else:
            if self.params == None:
                self.params = self.get_params(img.size, self.scale, self.ratio)
            i, j, h, w, pad_t, pad_b, pad_l, pad_r, pad_type = self.params
            if pad_type == 0:
                img = F.pad(img, (pad_l, pad_t, pad_r, pad_b), padding_mode='constant')
            elif pad_type == 1:
                img = F.pad(img, (pad_l, pad_t, pad_r, pad_b), padding_mode='edge')
            elif pad_type == 2:
                img = F.pad(img, (pad_l, pad_t, pad_r, pad_b), padding_mode='reflect')
            return F.resized_crop(img, i, j, h, w, self.size, self.interpolation)

    def randomize_parameters(self):
        self.params = None


class RandomResizedCrop3(object):
    """Crop the given Image to random size.
    A crop of random size of the original size and a random is made. This crop
    is finally resized to given size.
    Args:
        size: expected output size of each edge
        scale: range of size of the origin size cropped
        interpolation: Default: PIL.Image.BILINEAR
    """

    def __init__(self, size, scale=(256, 320), interpolation=cv2.INTER_LINEAR):
        if isinstance(size, int):
            self.size = (size, size)
        else:
            self.size = size
        self.interpolation = interpolation
        self.scale = scale
        self.params = None

    @staticmethod
    def get_params(size, scale):
        minlen = min(size[0], size[1])
        target_scale = random.randint(*scale) / minlen
        w = int(round(size[0] * target_scale))
        h = int(round(size[1] * target_scale))

        pad_t = 0
        pad_b = 0
        pad_l = 0
        pad_r = 0
        pad_type = -1
        if size[0] >= w:
            j = random.randint(0, size[0] - w)
        else:
            j = 0
            pad_l = random.randint(0, w - size[0])
            pad_r = w - size[0] - pad_l
            pad_type = random.randint(0,2)
        if size[1] >= h:
            i = random.randint(0, size[1] - h)
        else:
            i = 0
            pad_t = random.randint(0, h - size[1])
            pad_b = h - size[1] - pad_t
            pad_type = random.randint(0,2)
        return i, j, h, w, pad_t, pad_b, pad_l, pad_r, pad_type

    def __call__(self, img):
        if isinstance(img, np.ndarray):
            if self.params == None:
                ih, iw = img.shape[:2]
                self.params = self.get_params((iw,ih), self.scale)
            i, j, h, w, pad_t, pad_b, pad_l, pad_r, pad_type = self.params
            if pad_type == 0:
                img = cv2.copyMakeBorder(img,pad_t,pad_b,pad_l,pad_r,cv2.BORDER_CONSTANT,value=0)
            elif pad_type == 1:
                img = cv2.copyMakeBorder(img,pad_t,pad_b,pad_l,pad_r,cv2.BORDER_REPLICATE)
            elif pad_type == 2:
                img = cv2.copyMakeBorder(img,pad_t,pad_b,pad_l,pad_r,cv2.BORDER_REFLECT_101)
            img = cv2.resize(img[i:i+h,j:j+w], self.size, self.interpolation)
            return img
        else:
            if self.params == None:
                self.params = self.get_params(img.size, self.scale)
            i, j, h, w, pad_t, pad_b, pad_l, pad_r, pad_type = self.params
            if pad_type == 0:
                img = F.pad(img, (pad_l, pad_t, pad_r, pad_b), padding_mode='constant')
            elif pad_type == 1:
                img = F.pad(img, (pad_l, pad_t, pad_r, pad_b), padding_mode='edge')
            elif pad_type == 2:
                img = F.pad(img, (pad_l, pad_t, pad_r, pad_b), padding_mode='reflect')
            return F.resized_crop(img, i, j, h, w, self.size, self.interpolation)

    def randomize_parameters(self):
        self.params = None


class GroupRandomResizedCrop3(object):
    """Crop the given Image to random size.
    A crop of random size of the original size and a random is made. This crop
    is finally resized to given size.
    Args:
        size: expected output size of each edge
        scale: range of size of the origin size cropped
        interpolation: Default: PIL.Image.BILINEAR
    """

    def __init__(self, size, scale=(256, 320), interpolation=cv2.INTER_LINEAR, always_random=True):
        if isinstance(size, int):
            self.size = (size, size)
        else:
            self.size = size
        self.interpolation = interpolation
        self.scale = scale
        self.params = None
        self.always_random = always_random

    @staticmethod
    def get_params(size, scale):
        minlen = min(size[0], size[1])
        target_scale = random.randint(*scale) / minlen
        w = int(round(size[0] * target_scale))
        h = int(round(size[1] * target_scale))

        pad_t = 0
        pad_b = 0
        pad_l = 0
        pad_r = 0
        pad_type = -1
        if size[0] >= w:
            j = random.randint(0, size[0] - w)
        else:
            j = 0
            pad_l = random.randint(0, w - size[0])
            pad_r = w - size[0] - pad_l
            pad_type = random.randint(0,2)
        if size[1] >= h:
            i = random.randint(0, size[1] - h)
        else:
            i = 0
            pad_t = random.randint(0, h - size[1])
            pad_b = h - size[1] - pad_t
            pad_type = random.randint(0,2)
        return i, j, h, w, pad_t, pad_b, pad_l, pad_r, pad_type

    def __call__(self, img_group):
        if self.always_random:
            self.randomize_parameters()
        out_group = []
        for img in img_group:
            if isinstance(img, np.ndarray):
                if self.params == None:
                    ih, iw = img.shape[:2]
                    self.params = self.get_params((iw,ih), self.scale)
                i, j, h, w, pad_t, pad_b, pad_l, pad_r, pad_type = self.params
                if pad_type == 0:
                    img = cv2.copyMakeBorder(img,pad_t,pad_b,pad_l,pad_r,cv2.BORDER_CONSTANT,value=0)
                elif pad_type == 1:
                    img = cv2.copyMakeBorder(img,pad_t,pad_b,pad_l,pad_r,cv2.BORDER_REPLICATE)
                elif pad_type == 2:
                    img = cv2.copyMakeBorder(img,pad_t,pad_b,pad_l,pad_r,cv2.BORDER_REFLECT_101)
                img = cv2.resize(img[i:i+h,j:j+w], self.size, self.interpolation)
                out_group.append(img)
            else:
                if self.params == None:
                    self.params = self.get_params(img.size, self.scale)
                i, j, h, w, pad_t, pad_b, pad_l, pad_r, pad_type = self.params
                if pad_type == 0:
                    img = F.pad(img, (pad_l, pad_t, pad_r, pad_b), padding_mode='constant')
                elif pad_type == 1:
                    img = F.pad(img, (pad_l, pad_t, pad_r, pad_b), padding_mode='edge')
                elif pad_type == 2:
                    img = F.pad(img, (pad_l, pad_t, pad_r, pad_b), padding_mode='reflect')
                out_group.append(F.resized_crop(img, i, j, h, w, self.size, self.interpolation))
        return out_group

    def randomize_parameters(self):
        self.params = None


class RandomHorizontalFlip(object):
    """Horizontally flip the given PIL Image randomly with a probability of 0.5."""

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be flipped.
        Returns:
            PIL Image: Randomly flipped image.
        """
        if self.p < 0.5:
            if isinstance(img, np.ndarray):
                return cv2.flip(img, 0)
            else:
                return F.hflip(img)
        return img

    def randomize_parameters(self):
        self.p = random.random()


class GroupRandomHorizontalFlip(object):
    """Horizontally flip the given PIL Image randomly with a probability of 0.5."""

    def __call__(self, img_group):
        """
        Args:
            img_group (PIL Image): Image group to be flipped.
        Returns:
            Image group: Randomly flipped image group.
        """
        self.randomize_parameters()
        out_group = []
        if self.p < 0.5:
            for img in img_group:
                if isinstance(img, np.ndarray):
                    out_group.append(cv2.flip(img, 0))
                else:
                    out_group.append(F.hflip(img))
        else:
            out_group = img_group
        return out_group

    def randomize_parameters(self):
        self.p = random.random()


class Resize(object):
    def __init__(self, size, interpolation=cv2.INTER_LINEAR):
        if isinstance(size, int):
            # self.size = (size, size)
            self.size = size
            print("[CHANGED Jul 18, 2018] Now RESIZE(INT) keeps the aspect ratio of input!!!")
        else:
            self.size = size
        self.interpolation = interpolation

    def __call__(self, img):
        size = self.size
        if isinstance(img, np.ndarray):
            if isinstance(size, int):
                h, w = img.shape[:2]
                if (w <= h and w == size) or (h <= w and h == size):
                    return img
                if w < h:
                    ow = size
                    oh = int(size * h / w)
                    return cv2.resize(img, (ow, oh), self.interpolation)
                else:
                    oh = size
                    ow = int(size * w / h)
                    return cv2.resize(img, (ow, oh), self.interpolation)
            return cv2.resize(img, size, self.interpolation)
        else:
            return F.resize(img, size, self.interpolation)


class GroupResize(object):
    def __init__(self, size, interpolation=cv2.INTER_LINEAR):
        if isinstance(size, int):
            # self.size = (size, size)
            self.size = size
            print("[CHANGED Jul 18, 2018] Now RESIZE(INT) keeps the aspect ratio of input!!!")
        else:
            self.size = size
        self.interpolation = interpolation

    def __call__(self, img_group):
        size = self.size
        out_group = []
        for img in img_group:
            if isinstance(img, np.ndarray):
                if isinstance(size, int):
                    h, w = img.shape[:2]
                    if (w <= h and w == size) or (h <= w and h == size):
                        out_group.append(img)
                    elif w < h:
                        ow = size
                        oh = int(size * h / w)
                        out_group.append(cv2.resize(img, (ow, oh), self.interpolation))
                    else:
                        oh = size
                        ow = int(size * w / h)
                        out_group.append(cv2.resize(img, (ow, oh), self.interpolation))
                else:
                    out_group.append(cv2.resize(img, size, self.interpolation))
            else:
                out_group.append(F.resize(img, size, self.interpolation))
        return out_group


class GroupRandomResize(object):
    def __init__(self, scale=(256,320), interpolation=cv2.INTER_LINEAR, always_random=True):
        self.scale = scale
        self.interpolation = interpolation
        self.params = None
        self.always_random = always_random

    @staticmethod
    def get_params(scale):
        return random.randint(*scale)

    def __call__(self, img_group):
        if self.always_random:
            self.randomize_parameters()
        out_group = []
        if self.params == None:
            self.params = self.get_params(self.scale)
        size = self.params
        for img in img_group:
            if isinstance(img, np.ndarray):
                if isinstance(size, int):
                    h, w = img.shape[:2]
                    if (w <= h and w == size) or (h <= w and h == size):
                        out_group.append(img)
                    elif w < h:
                        ow = size
                        oh = int(size * h / w)
                        out_group.append(cv2.resize(img, (ow, oh), self.interpolation))
                    else:
                        oh = size
                        ow = int(size * w / h)
                        out_group.append(cv2.resize(img, (ow, oh), self.interpolation))
                else:
                    out_group.append(cv2.resize(img, size, self.interpolation))
            else:
                out_group.append(F.resize(img, size, self.interpolation))
        return out_group

    def randomize_parameters(self):
        self.params = None


class CenterCrop(object):
    def __init__(self, size):
        if isinstance(size, int):
            self.size = (size, size)
        else:
            self.size = size

    def __call__(self, img):

        if isinstance(img, np.ndarray):
            h, w = img.shape[:2]
            th, tw = self.size
            i = int(round((h - th) / 2.))
            j = int(round((w - tw) / 2.))
            return img[i:i+th,j:j+tw]
        else:
            return F.center_crop(img, self.size)


class GroupCenterCrop(object):
    def __init__(self, size):
        if isinstance(size, int):
            self.size = (size, size)
        else:
            self.size = size

    def __call__(self, img_group):

        out_group = []
        for img in img_group:
            if isinstance(img, np.ndarray):
                h, w = img.shape[:2]
                th, tw = self.size
                i = int(round((h - th) / 2.))
                j = int(round((w - tw) / 2.))
                out_group.append(img[i:i+th,j:j+tw])
            else:
                out_group.append(F.center_crop(img, self.size))
        return out_group


class ColorJitter(object):
    """Randomly change the brightness, contrast and saturation of an image.
    Args:
        brightness (float): How much to jitter brightness. brightness_factor
            is chosen uniformly from [max(0, 1 - brightness), 1 + brightness].
        contrast (float): How much to jitter contrast. contrast_factor
            is chosen uniformly from [max(0, 1 - contrast), 1 + contrast].
        saturation (float): How much to jitter saturation. saturation_factor
            is chosen uniformly from [max(0, 1 - saturation), 1 + saturation].
        hue(float): How much to jitter hue. hue_factor is chosen uniformly from
            [-hue, hue]. Should be >=0 and <= 0.5.
    """
    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue

    def __call__(self, img):
        # apply transforms in HSV colorspace
        imghsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        if self.brightness > 0 or self.contrast > 0:
            imghsv[:,:,2] = cv2.convertScaleAbs(imghsv[:,:,2], alpha=self.contrast_factor, beta=self.brightness_factor*255)
        if self.saturation > 0:
            imghsv[:,:,1] = cv2.multiply(imghsv[:,:,1], self.saturation_factor)
        if self.hue > 0:
            imghsv[:,:,0] = np.uint8((np.int16(imghsv[:,:,0]) + self.hue_factor*180) % 180)
        img_rgb = cv2.cvtColor(imghsv, cv2.COLOR_HSV2RGB)
        # blend the input image with the transformed image
        return cv2.addWeighted(img, 0.5, img_rgb, 0.5, 0)

    def __repr__(self):
        return self.__class__.__name__ + '()'

    def randomize_parameters(self):
        self.brightness_factor = np.random.uniform(-self.brightness, self.brightness)
        self.contrast_factor = np.random.uniform(max(0, 1 - self.contrast), 1 + self.contrast)
        self.saturation_factor = np.random.uniform(max(0, 1 - self.saturation), 1 + self.saturation)
        self.hue_factor = np.random.uniform(-self.hue, self.hue)


class GroupColorJitter(object):
    """Randomly change the brightness, contrast and saturation of an image.
    Args:
        brightness (float): How much to jitter brightness. brightness_factor
            is chosen uniformly from [max(0, 1 - brightness), 1 + brightness].
        contrast (float): How much to jitter contrast. contrast_factor
            is chosen uniformly from [max(0, 1 - contrast), 1 + contrast].
        saturation (float): How much to jitter saturation. saturation_factor
            is chosen uniformly from [max(0, 1 - saturation), 1 + saturation].
        hue(float): How much to jitter hue. hue_factor is chosen uniformly from
            [-hue, hue]. Should be >=0 and <= 0.5.
    """
    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0, always_random=True):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue
        self.always_random = always_random

    def __call__(self, img_group):
        if self.always_random:
            self.randomize_parameters()
        out_group = []
        for img in img_group:
            # apply transforms in HSV colorspace
            imghsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
            if self.brightness > 0 or self.contrast > 0:
                imghsv[:,:,2] = cv2.convertScaleAbs(imghsv[:,:,2], alpha=self.contrast_factor, beta=self.brightness_factor*255)
            if self.saturation > 0:
                imghsv[:,:,1] = cv2.multiply(imghsv[:,:,1], self.saturation_factor)
            if self.hue > 0:
                imghsv[:,:,0] = np.uint8((np.int16(imghsv[:,:,0]) + self.hue_factor*180) % 180)
            img_rgb = cv2.cvtColor(imghsv, cv2.COLOR_HSV2RGB)
            # blend the input image with the transformed image
            out_group.append(cv2.addWeighted(img, 0.5, img_rgb, 0.5, 0))
        return out_group

    def __repr__(self):
        return self.__class__.__name__ + '()'

    def randomize_parameters(self):
        self.brightness_factor = np.random.uniform(-self.brightness, self.brightness)
        self.contrast_factor = np.random.uniform(max(0, 1 - self.contrast), 1 + self.contrast)
        self.saturation_factor = np.random.uniform(max(0, 1 - self.saturation), 1 + self.saturation)
        self.hue_factor = np.random.uniform(-self.hue, self.hue)


class GroupFiveCrop(object):
    """Crop the given PIL Image into four corners and the central crop
    .. Note::
         This transform returns a tuple of images and there may be a mismatch in the number of
         inputs and targets your Dataset returns. See below for an example of how to deal with
         this.
    Args:
         size (sequence or int): Desired output size of the crop. If size is an ``int``
            instead of sequence like (h, w), a square crop of size (size, size) is made.
    Example:
         >>> transform = Compose([
         >>>    FiveCrop(size), # this is a list of PIL Images
         >>>    Lambda(lambda crops: torch.stack([ToTensor()(crop) for crop in crops])) # returns a 4D tensor
         >>> ])
         >>> #In your test loop you can do the following:
         >>> input, target = batch # input is a 5d tensor, target is 2d
         >>> bs, ncrops, c, h, w = input.size()
         >>> result = model(input.view(-1, c, h, w)) # fuse batch size and ncrops
         >>> result_avg = result.view(bs, ncrops, -1).mean(1) # avg over crops
    """

    def __init__(self, size):
        self.size = size
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            assert len(size) == 2, "Please provide only two dimensions (h, w) for size."
            self.size = size

    def __call__(self, img_group):
        out_group_tl = []
        out_group_tr = []
        out_group_bl = []
        out_group_br = []
        out_group_center = []
        for img in img_group:
            if isinstance(img, np.ndarray):
                h, w = img.shape[:2]
                crop_h, crop_w = self.size
                if crop_w > w or crop_h > h:
                    raise ValueError("Requested crop size {} is bigger than input size {}".format(size,
                                                                                                  (h, w)))
                tl = img[0:crop_h, 0:crop_w] 
                tr = img[0:crop_h, w - crop_w:w]
                bl = img[h - crop_h:h, 0:crop_w]
                br = img[h - crop_h:h, w - crop_w:w]
                i = int(round((h - crop_h) / 2.))
                j = int(round((w - crop_w) / 2.))
                center = img[i:i+crop_h,j:j+crop_w]
            else:
                tl, tr, bl, br, center = F.five_crop(img, self.size)
            out_group_tl.append(tl)
            out_group_tr.append(tr)
            out_group_bl.append(bl)
            out_group_br.append(br)
            out_group_center.append(center)
        return (out_group_tl, out_group_tr, out_group_bl, out_group_br, out_group_center)

    def __repr__(self):
        return self.__class__.__name__ + '(size={0})'.format(self.size)


class GroupTenCrop(object):
    """Crop the given PIL Image into four corners and the central crop
    .. Note::
         This transform returns a tuple of images and there may be a mismatch in the number of
         inputs and targets your Dataset returns. See below for an example of how to deal with
         this.
    Args:
         size (sequence or int): Desired output size of the crop. If size is an ``int``
            instead of sequence like (h, w), a square crop of size (size, size) is made.
    Example:
         >>> transform = Compose([
         >>>    FiveCrop(size), # this is a list of PIL Images
         >>>    Lambda(lambda crops: torch.stack([ToTensor()(crop) for crop in crops])) # returns a 4D tensor
         >>> ])
         >>> #In your test loop you can do the following:
         >>> input, target = batch # input is a 5d tensor, target is 2d
         >>> bs, ncrops, c, h, w = input.size()
         >>> result = model(input.view(-1, c, h, w)) # fuse batch size and ncrops
         >>> result_avg = result.view(bs, ncrops, -1).mean(1) # avg over crops
    """

    def __init__(self, size):
        self.size = size
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            assert len(size) == 2, "Please provide only two dimensions (h, w) for size."
            self.size = size

    def __call__(self, img_group):
        out_group_tl = []
        out_group_tr = []
        out_group_bl = []
        out_group_br = []
        out_group_center = []
        out_group_tl_flip = []
        out_group_tr_flip = []
        out_group_bl_flip = []
        out_group_br_flip = []
        out_group_center_flip = []
        for img in img_group:
            if isinstance(img, np.ndarray):
                h, w = img.shape[:2]
                crop_h, crop_w = self.size
                if crop_w > w or crop_h > h:
                    raise ValueError("Requested crop size {} is bigger than input size {}".format(size,
                                                                                                  (h, w)))
                tl = img[0:crop_h, 0:crop_w]
                tr = img[0:crop_h, w - crop_w:w]
                bl = img[h - crop_h:h, 0:crop_w]
                br = img[h - crop_h:h, w - crop_w:w]
                i = int(round((h - crop_h) / 2.))
                j = int(round((w - crop_w) / 2.))
                center = img[i:i+crop_h,j:j+crop_w]
                tl_flip, tr_flip, bl_flip, br_flip, center_flip = tuple(map(lambda x: cv2.flip(x,0), (tl,tr,bl,br,center)))
            else:
                tl, tr, bl, br, center, tl_flip, tr_flip, bl_flip, br_flip, center_flip = F.ten_crop(img, self.size)
            out_group_tl.append(tl)
            out_group_tr.append(tr)
            out_group_bl.append(bl)
            out_group_br.append(br)
            out_group_center.append(center)
            out_group_tl_flip.append(tl_flip)
            out_group_tr_flip.append(tr_flip)
            out_group_bl_flip.append(bl_flip)
            out_group_br_flip.append(br_flip)
            out_group_center_flip.append(center_flip)
        return (out_group_tl, out_group_tr, out_group_bl, out_group_br, out_group_center, out_group_tl_flip, out_group_tr_flip, out_group_bl_flip, out_group_br_flip, out_group_center_flip)

    def __repr__(self):
        return self.__class__.__name__ + '(size={0})'.format(self.size)


class GroupThreeCrop(object):
    """Crop the given PIL Image into four corners and the central crop
    .. Note::
         This transform returns a tuple of images and there may be a mismatch in the number of
         inputs and targets your Dataset returns. See below for an example of how to deal with
         this.
    Args:
         size (sequence or int): Desired output size of the crop. If size is an ``int``
            instead of sequence like (h, w), a square crop of size (size, size) is made.
    Example:
         >>> transform = Compose([
         >>>    FiveCrop(size), # this is a list of PIL Images
         >>>    Lambda(lambda crops: torch.stack([ToTensor()(crop) for crop in crops])) # returns a 4D tensor
         >>> ])
         >>> #In your test loop you can do the following:
         >>> input, target = batch # input is a 5d tensor, target is 2d
         >>> bs, ncrops, c, h, w = input.size()
         >>> result = model(input.view(-1, c, h, w)) # fuse batch size and ncrops
         >>> result_avg = result.view(bs, ncrops, -1).mean(1) # avg over crops
    """

    def __init__(self, size):
        self.size = size
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            assert len(size) == 2, "Please provide only two dimensions (h, w) for size."
            self.size = size

    def __call__(self, img_group):
        out_group_l = []
        out_group_r = []
        out_group_center = []
        convert2PIL = False
        for img in img_group:
            if not isinstance(img, np.ndarray):
                img = np.array(img)
                convert2PIL = True
            h, w = img.shape[:2]
            crop_h, crop_w = self.size
            if crop_w > w or crop_h > h:
                raise ValueError("Requested crop size {} is bigger than input size {}".format(size,
                                                                                              (h, w)))

            i = int(round((h - crop_h) / 2.))
            j = int(round((w - crop_w) / 2.))
            left = img[i:i+crop_h, 0:crop_w] 
            right = img[i:i+crop_h, w - crop_w:w]
            center = img[i:i+crop_h,j:j+crop_w]
            if convert2PIL:
                out_group_l.append(Image. fromarray(left))
                out_group_r.append(Image. fromarray(right))
                out_group_center.append(Image. fromarray(center))
            else:
                out_group_l.append(left)
                out_group_r.append(right)
                out_group_center.append(center)
        return (out_group_l, out_group_r, out_group_center)

    def __repr__(self):
        return self.__class__.__name__ + '(size={0})'.format(self.size)


class GroupFormat(object):

    def __init__(self):
        return

    def __call__(self, img_group):
        clip = img_group
        if len(clip) == 0:
            return clip
        # format data to torch tensor
        if isinstance(clip[0], list):
            clip_ = []
            for c in clip:
                clip_.append(torch.from_numpy(np.stack(c, 0).transpose(3, 0, 1, 2)))
            clip = torch.stack(clip_)
        else:
            clip = torch.from_numpy(np.stack(clip, 0).transpose(3, 0, 1, 2))

        return clip
