import random
import torch
import torchvision.transforms.functional as F
import numpy as np
from PIL import Image
import math
import numbers

def _augmentation(image, aug, resize_shape=128):
    """
    Do train time augmentation mainly using torch.transforms.functional.

        Args:
            image: PIL Image.
            aug (bool): True or False, whether to do augmentation or do just resize.
            resize_shape (int): resized shape, default 128.
        Returns:
            image (PIL Image): resized image
    """
   
    image = Image.fromarray(np.uint8(image))
    # image = F.to_pil_image(image)

    # ## to gray
    # image = F.to_grayscale(image, num_output_channels=3)
    if aug:
        # hflip
        if random.random() < 0.5:
            image = F.hflip(image)
        
        if random.random() < 0.5:
            image = F.vflip(image)

        ## center crop
        image = F.center_crop(image, int(0.8*image.size[0]))

        ## to gray
        # if random.random() < 0.5:
        #     image = F.to_grayscale(image, num_output_channels=3)

        ## brightness, contrast, saturation, hue
        r_bright = random.uniform(0.7, 1.3)
        image = F.adjust_brightness(image, r_bright)
        # r_contrast = random.uniform(0.7, 1.3)
        # image = F.adjust_contrast(image, r_contrast)
        # r_sat = random.uniform(0.3, 1.7)
        # image = F.adjust_saturation(image, r_sat)
        # r_hue = random.uniform(-0.3, 0.3)
        # image = F.adjust_hue(image, r_hue)

        # rotate
        # angle = random.uniform(-90, 90)
        # image = F.rotate(image, angle)

        ## affine
        # # image = F.pad(image, padding=50, padding_mode='reflect')
        # angle = random.uniform(-10, 10)
        # translate = random.uniform(-0.1, 0.1)
        # translate = (0,0)#(int(translate*image.size[0]), int(translate*image.size[1]))
        # scale = random.uniform(0.8, 1.2)
        # shear = random.uniform(-10, 10)
        # image = F.affine(image, angle, translate, scale, shear, resample=0, fillcolor=None)
    
        # perspective
        # distortion_scale = random.uniform(0, 0.2)
        # height, width = image.size[0], image.size[1]
        # half_height = int(height / 2)
        # half_width = int(width / 2)
        # topleft = (random.randint(0, int(distortion_scale * half_width)),
        #            random.randint(0, int(distortion_scale * half_height)))
        # topright = (random.randint(width - int(distortion_scale * half_width) - 1, width - 1),
        #             random.randint(0, int(distortion_scale * half_height)))
        # botright = (random.randint(width - int(distortion_scale * half_width) - 1, width - 1),
        #             random.randint(height - int(distortion_scale * half_height) - 1, height - 1))
        # botleft = (random.randint(0, int(distortion_scale * half_width)),
        #            random.randint(height - int(distortion_scale * half_height) - 1, height - 1))
        # startpoints = [(0, 0), (width - 1, 0), (width - 1, height - 1), (0, height - 1)]
        # endpoints = [topleft, topright, botright, botleft]
        # image = F.perspective(image, startpoints, endpoints)

        ## random erasing
        # image = torch.from_numpy(np.array(image, np.float32, copy=False))
        # p = 0.5
        # scale = (0.02, 0.33)
        # ratio = (0.3, 3.3)
        # value = 0
        # inplace = False
        # img_h, img_w, _ = image.shape
        # img_c = 3
        # area = img_h * img_w
        
        # for attempt in range(10):
        #     erase_area = random.uniform(scale[0], scale[1]) * area
        #     aspect_ratio = random.uniform(ratio[0], ratio[1])
        
        #     h = int(round(math.sqrt(erase_area * aspect_ratio)))
        #     w = int(round(math.sqrt(erase_area / aspect_ratio)))
        
        #     if h < img_h and w < img_w:
        #         i = random.randint(0, img_h - h)
        #         j = random.randint(0, img_w - w)
        #         if isinstance(value, numbers.Number):
        #             v = value
        #         elif isinstance(value, torch._six.string_classes):
        #             v = torch.empty([img_c, h, w], dtype=torch.float32).normal_()
        #         elif isinstance(value, (list, tuple)):
        #             v = torch.tensor(value, dtype=torch.float32).view(-1, 1, 1).expand(-1, h, w)
        #         x, y, h, w, v = i, j, h, w, v
        #         break
        #     if attempt ==9:
        #         # Return original image
        #         x, y, h, w, v = 0, 0, img_h, img_w, image
        # image = F.erase(image, x, y, h, w, v, inplace)
        # image = Image.fromarray(np.uint8(image))

    ## resize
    image = F.resize(image, resize_shape)
    return image

def batch_augment(batch_x, aug=True, resize_shape=128):
    """
    do data augmentation on batch_x
    """
    if resize_shape is not None:
        new_batch_x = np.zeros((batch_x.shape[0],resize_shape,resize_shape,3))#np.zeros_like(batch_x)
    for i, x in enumerate(batch_x):
        new_batch_x[i] = _augmentation(x, aug, resize_shape)
    return new_batch_x