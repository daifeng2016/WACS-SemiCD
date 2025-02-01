import random
import math

import numpy as np
from PIL import Image, ImageOps, ImageFilter
import torch
from torchvision import transforms
import cv2
import torch.nn.functional as F

def crop(imgA, imgB, mask, size, ignore_value=255):
    w, h = imgA.size
    padw = size - w if w < size else 0
    padh = size - h if h < size else 0
    imgA = ImageOps.expand(imgA, border=(0, 0, padw, padh), fill=0)
    imgB = ImageOps.expand(imgB, border=(0, 0, padw, padh), fill=0)
    mask = ImageOps.expand(mask, border=(0, 0, padw, padh), fill=ignore_value)

    w, h = imgA.size
    x = random.randint(0, w - size)
    y = random.randint(0, h - size)
    imgA = imgA.crop((x, y, x + size, y + size))
    imgB = imgB.crop((x, y, x + size, y + size))
    mask = mask.crop((x, y, x + size, y + size))

    return imgA, imgB, mask


def hflip(imgA, imgB, mask, p=0.5):
    if random.random() < p:
        imgA = imgA.transpose(Image.FLIP_LEFT_RIGHT)
        imgB = imgB.transpose(Image.FLIP_LEFT_RIGHT)
        mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
    return imgA, imgB, mask

def vflip(imgA, imgB, mask, p=0.5):
    if random.random() < p:
        imgA = imgA.transpose(Image.FLIP_TOP_BOTTOM)
        imgB = imgB.transpose(Image.FLIP_TOP_BOTTOM)
        mask = mask.transpose(Image.FLIP_TOP_BOTTOM)
    return imgA, imgB, mask


def normalize(img, mask=None):
    img = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])(img)
    if mask is not None:
        mask = torch.from_numpy(np.array(mask)).long()
        return img, mask
    return img

def denormalize(img):#tensor
    #mean,std=[0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    mean=torch.from_numpy(np.array([0.485, 0.456, 0.406]))
    std=torch.from_numpy(np.array([0.229, 0.224, 0.225]))
    mean=mean.reshape(1,3,1,1).cuda()
    std=std.reshape(1,3,1,1).cuda()
    img_new=img*std+mean
    return img_new.float()





def resize(imgA, imgB, mask, ratio_range):
    w, h = imgA.size
    long_side = random.randint(int(max(h, w) * ratio_range[0]), int(max(h, w) * ratio_range[1]))

    if h > w:
        oh = long_side
        ow = int(1.0 * w * long_side / h + 0.5)
    else:
        ow = long_side
        oh = int(1.0 * h * long_side / w + 0.5)

    imgA = imgA.resize((ow, oh), Image.BILINEAR)
    imgB = imgB.resize((ow, oh), Image.BILINEAR)
    mask = mask.resize((ow, oh), Image.NEAREST)
    return imgA, imgB, mask


def blur(img, p=0.5):
    if random.random() < p:
        sigma = np.random.uniform(0.1, 2.0)
        img = img.filter(ImageFilter.GaussianBlur(radius=sigma))
    return img


def obtain_cutmix_box(img_size, p=0.5, size_min=0.02, size_max=0.4, ratio_1=0.3, ratio_2=1/0.3):
    mask = torch.zeros(img_size, img_size)
    if random.random() > p:
        return mask

    size = np.random.uniform(size_min, size_max) * img_size * img_size
    while True:
        ratio = np.random.uniform(ratio_1, ratio_2)
        cutmix_w = int(np.sqrt(size / ratio))
        cutmix_h = int(np.sqrt(size * ratio))
        x = np.random.randint(0, img_size)
        y = np.random.randint(0, img_size)

        if x + cutmix_w <= img_size and y + cutmix_h <= img_size:
            break

    mask[y:y + cutmix_h, x:x + cutmix_w] = 1

    return mask


#==============for teacher aug========================
def transforms_for_noise(inputs_u2, std):
    ch=inputs_u2.shape[1]
    gaussian = np.random.normal(0, std, (inputs_u2.shape[0], ch, inputs_u2.shape[-1], inputs_u2.shape[-1]))
    gaussian = torch.from_numpy(gaussian).float().cuda()
    inputs_u2_noise = inputs_u2 + gaussian

    return inputs_u2_noise

def transforms_for_rot(ema_inputs):

    rot_mask = np.random.randint(0, 4, ema_inputs.shape[0])#ema_inputs=[N,C,H,W]
    flip_mask = np.random.randint(0, 2, ema_inputs.shape[0])

    # flip_mask = [0,0,0,0,1,1,1,1]
    # rot_mask = [0,1,2,3,0,1,2,3]

    for idx in range(ema_inputs.shape[0]):
        if flip_mask[idx] == 1:
            ema_inputs[idx] = torch.flip(ema_inputs[idx], [1])##flipup

        ema_inputs[idx] = torch.rot90(ema_inputs[idx], int(rot_mask[idx]), dims=[1,2])

    return ema_inputs, rot_mask, flip_mask


def transforms_for_scale(ema_inputs, image_size):

    scale_mask = np.random.uniform(low=0.8, high=1.2, size=ema_inputs.shape[0])
    scale_mask = scale_mask * image_size
    scale_mask = [int(item) for item in scale_mask]
    scale_mask = [item-1 if item % 2 != 0 else item for item in scale_mask]
    half_size = int(image_size / 2)

    ema_outputs = torch.zeros_like(ema_inputs)

    for idx in range(ema_inputs.shape[0]):
        # to numpy
        img = np.transpose(ema_inputs[idx].cpu().numpy(), (1,2,0))
        # crop
        if scale_mask[idx] > image_size:
            # new_img = np.zeros((scale_mask[idx], scale_mask[idx], 3), dtype="uint8")
            # new_img[int(scale_mask[idx]/2)-half_size:int(scale_mask[idx]/2)+half_size,
            # int(scale_mask[idx] / 2) - half_size:int(scale_mask[idx]/2) + half_size, :] = img
            new_img1 = np.expand_dims(np.pad(img[:, :, 0],
                                             (int((scale_mask[idx]-image_size)/2),
                                             int((scale_mask[idx]-image_size)/2)), 'edge'), axis=-1)
            new_img2 = np.expand_dims(np.pad(img[:, :, 1],
                                             (int((scale_mask[idx]-image_size)/2),
                                             int((scale_mask[idx]-image_size)/2)), 'edge'), axis=-1)
            new_img3 = np.expand_dims(np.pad(img[:, :, 2],
                                             (int((scale_mask[idx] - image_size) / 2),
                                              int((scale_mask[idx] - image_size) / 2)), 'edge'), axis=-1)

            new_img4 = np.expand_dims(np.pad(img[:, :, 3],
                                             (int((scale_mask[idx] - image_size) / 2),
                                              int((scale_mask[idx] - image_size) / 2)), 'edge'), axis=-1)
            new_img5 = np.expand_dims(np.pad(img[:, :, 4],
                                             (int((scale_mask[idx] - image_size) / 2),
                                              int((scale_mask[idx] - image_size) / 2)), 'edge'), axis=-1)
            new_img6= np.expand_dims(np.pad(img[:, :, 5],
                                             (int((scale_mask[idx] - image_size) / 2),
                                              int((scale_mask[idx] - image_size) / 2)), 'edge'), axis=-1)


            new_img = np.concatenate([new_img1, new_img2, new_img3,new_img4, new_img5, new_img6], axis=-1)
            img = new_img
        else:
            img = img[half_size-int(scale_mask[idx]/2):half_size + int(scale_mask[idx]/2),
            half_size-int(scale_mask[idx]/2): half_size + int(scale_mask[idx]/2),:]

        # resize
        img_tensor=torch.from_numpy(img.transpose((2, 0, 1)))
        resized_img=F.interpolate(img_tensor.unsqueeze(0),(image_size,image_size),mode='bicubic',align_corners=True)
        #resized_img = cv2.resize(img, (image_size, image_size), interpolation=cv2.INTER_CUBIC)
        # to tensor
        ema_outputs[idx] =resized_img[0].cuda() # torch.from_numpy(resized_img.transpose((2, 0, 1))).cuda()

    return ema_outputs, scale_mask



def transforms_back_scale(ema_inputs, scale_mask, image_size):
    half_size = int(image_size/2)
    returned_img = np.zeros((ema_inputs.shape[0],  image_size, image_size, ema_inputs.shape[1]))#[16,128,128,1]

    ema_outputs = torch.zeros_like(ema_inputs)#[16,1,128,128]

    for idx in range(ema_inputs.shape[0]):
        # to numpy
        #img = np.transpose(ema_inputs[idx].cpu().numpy(), (1,2,0))#[128,128,1]
        # resize
        #resized_img = cv2.resize(img, (int(scale_mask[idx]), int(scale_mask[idx])), interpolation=cv2.INTER_CUBIC)#
        img=ema_inputs[idx].cpu()
        resized_img=F.interpolate(img.unsqueeze(0),(int(scale_mask[idx]), int(scale_mask[idx])),mode='bicubic',align_corners=True).squeeze(0)
        resized_img=resized_img.permute(1,2,0).numpy()



        if img.shape[-1]==1:
            resized_img=np.expand_dims(resized_img,axis=-1)

        if scale_mask[idx] > image_size:
            returned_img[idx] = resized_img[int(scale_mask[idx]/2)-half_size:int(scale_mask[idx]/2)+half_size,
            int(scale_mask[idx] / 2) - half_size:int(scale_mask[idx]/2) + half_size, :]

        else:
            returned_img[idx, half_size-int(scale_mask[idx]/2):half_size + int(scale_mask[idx]/2),
            half_size-int(scale_mask[idx]/2): half_size + int(scale_mask[idx]/2), :] = resized_img
        # to tensor
        ema_outputs[idx] = torch.from_numpy(returned_img[idx].transpose((2,0,1))).cuda()

    return ema_outputs, scale_mask

def postprocess_scale(input, scale_mask, image_size):
    half_size = int(input.shape[-1]/2)
    new_input = torch.zeros((input.shape[0], input.shape[1], input.shape[-1], input.shape[-1]))

    for idx in range(input.shape[0]):

        if scale_mask[idx] > image_size:
            new_input = input
        #     scale_num = int((image_size/(scale_mask[idx]/image_size))/2)
        #     new_input[idx, :, half_size - scale_num:half_size + scale_num,
        #     half_size - scale_num: half_size + scale_num] \
        #         = input[idx, :, half_size - scale_num:half_size + scale_num,
        #     half_size - scale_num: half_size + scale_num]
        else:
            new_input[idx, :, half_size-int(scale_mask[idx]/2):half_size + int(scale_mask[idx]/2),
            half_size-int(scale_mask[idx]/2): half_size + int(scale_mask[idx]/2)] \
            = input[idx, :, half_size-int(scale_mask[idx]/2):half_size + int(scale_mask[idx]/2),
            half_size-int(scale_mask[idx]/2): half_size + int(scale_mask[idx]/2)]

    return new_input.cuda()

def transforms_back_rot(ema_output,rot_mask, flip_mask):

    for idx in range(ema_output.shape[0]):

        ema_output[idx] = torch.rot90(ema_output[idx], int(rot_mask[idx]), dims=[2,1])

        if flip_mask[idx] == 1:
            ema_output[idx] = torch.flip(ema_output[idx], [1])

    return ema_output

def transforms_input_for_shift(ema_inputs, image_size):

    scale_mask = np.random.uniform(low=0.9, high=0.99, size=ema_inputs.shape[0])
    scale_mask = scale_mask * image_size
    scale_mask = [int(abs(item-image_size)) for item in scale_mask]
    scale_mask = [item-1 if item % 2 != 0 else item for item in scale_mask]
    scale_mask = [2 if item == 0 else item for item in scale_mask]
    # half_size = int(image_size / 2)

    shift_mask = np.random.randint(0, 4, ema_inputs.shape[0])


    for idx in range(ema_inputs.shape[0]):
        img = np.transpose(ema_inputs[idx].cpu().numpy(), (1, 2, 0))
        new_img1 = np.expand_dims(np.pad(img[:, :, 0], (scale_mask[idx], scale_mask[idx]), 'edge'), axis=-1)
        new_img2 = np.expand_dims(np.pad(img[:, :, 1], (scale_mask[idx], scale_mask[idx]), 'edge'), axis=-1)
        new_img3 = np.expand_dims(np.pad(img[:, :, 2], (scale_mask[idx], scale_mask[idx]), 'edge'), axis=-1)
        new_img = np.concatenate([new_img1, new_img2, new_img3], axis=-1)

        if shift_mask[idx] == 0:
            ema_inputs[idx] = torch.from_numpy(new_img[0:image_size, 0:image_size,:].transpose((2,0,1))).cuda()
        elif shift_mask[idx] == 1:
            ema_inputs[idx] = torch.from_numpy(new_img[-image_size:, 0:image_size,:].transpose((2,0,1))).cuda()
        elif shift_mask[idx] == 2:
            ema_inputs[idx] = torch.from_numpy(new_img[0:image_size, -image_size:,:].transpose((2,0,1))).cuda()
        elif shift_mask[idx] == 3:
            ema_inputs[idx] = torch.from_numpy(new_img[-image_size:, -image_size:,:].transpose((2,0,1))).cuda()

    return ema_inputs, shift_mask, scale_mask

def transforms_output_for_shift(ema_inputs, shift_mask, scale_mask, image_size):

    for idx in range(ema_inputs.shape[0]):
        # shift back
        img = np.transpose(ema_inputs[idx].cpu().numpy(), (1, 2, 0))
        new_img1 = np.expand_dims(np.pad(img[:, :, 0], (scale_mask[idx], scale_mask[idx]), 'edge'), axis=-1)
        new_img2 = np.expand_dims(np.pad(img[:, :, 1], (scale_mask[idx], scale_mask[idx]), 'edge'), axis=-1)
        # new_img3 = np.expand_dims(np.pad(img[:, :, 2], (scale_mask[idx], scale_mask[idx]), 'edge'), axis=-1)
        new_img = np.concatenate([new_img1, new_img2], axis=-1)

        if shift_mask[idx] == 0:
            ema_inputs[idx] = torch.from_numpy(new_img[0:image_size, 0:image_size, :].transpose((2, 0, 1))).cuda()
        elif shift_mask[idx] == 1:
            ema_inputs[idx] = torch.from_numpy(new_img[-image_size:, 0:image_size, :].transpose((2, 0, 1))).cuda()
        elif shift_mask[idx] == 2:
            ema_inputs[idx] = torch.from_numpy(new_img[0:image_size, -image_size:, :].transpose((2, 0, 1))).cuda()
        elif shift_mask[idx] == 3:
            ema_inputs[idx] = torch.from_numpy(new_img[-image_size:, -image_size:, :].transpose((2, 0, 1))).cuda()

    return ema_inputs


def crop_output_for_shift(ema_inputs, shift_mask, scale_mask):


    ema_outputs = torch.zeros((ema_inputs.shape[0], 2, ema_inputs.shape[-1], ema_inputs.shape[-1]))

    for idx in range(ema_inputs.shape[0]):
        # shift back
        new_img = np.transpose(ema_inputs[idx].cpu().numpy(), (1, 2, 0)).copy()
        if shift_mask[idx] == 0:
            ema_outputs[idx,:,0:224-scale_mask[idx],0:224-scale_mask[idx]] = torch.from_numpy(new_img[scale_mask[idx]:, scale_mask[idx]:, :].transpose((2, 0, 1)))
        elif shift_mask[idx] == 1:
            ema_outputs[idx,:,0:224-scale_mask[idx],0:224-scale_mask[idx]] = torch.from_numpy(new_img[:-scale_mask[idx], scale_mask[idx]:, :].transpose((2, 0, 1)))
        elif shift_mask[idx] == 2:
            ema_outputs[idx,:,0:224-scale_mask[idx],0:224-scale_mask[idx]] = torch.from_numpy(new_img[scale_mask[idx]:, :-scale_mask[idx], :].transpose((2, 0, 1)))
        elif shift_mask[idx] == 3:
            ema_outputs[idx,:,0:224-scale_mask[idx],0:224-scale_mask[idx]] = torch.from_numpy(new_img[:-scale_mask[idx], :-scale_mask[idx], :].transpose((2, 0, 1)))

    ema_outputs = ema_outputs.cuda()
    return ema_outputs

def crop_output_back_shift(ema_inputs, shift_mask, scale_mask,image_size):

    ema_outputs = torch.zeros((ema_inputs.shape[0], 2 , ema_inputs.shape[-1], ema_inputs.shape[-1]))

    for idx in range(ema_inputs.shape[0]):
        # shift back
        new_img = np.transpose(ema_inputs[idx].cpu().numpy(), (1, 2, 0)).copy()
        if shift_mask[idx] == 0:
            ema_outputs[idx,:,0:224-scale_mask[idx],0:224-scale_mask[idx]] = torch.from_numpy(new_img[:-scale_mask[idx], :-scale_mask[idx], :].transpose((2, 0, 1)))
        elif shift_mask[idx] == 1:
            ema_outputs[idx,:,0:224-scale_mask[idx],0:224-scale_mask[idx]] = torch.from_numpy(new_img[scale_mask[idx]:, :-scale_mask[idx], :].transpose((2, 0, 1)))
        elif shift_mask[idx] == 2:
            ema_outputs[idx,:,0:224-scale_mask[idx],0:224-scale_mask[idx]] = torch.from_numpy(new_img[:-scale_mask[idx], scale_mask[idx]:, :].transpose((2, 0, 1)))
        elif shift_mask[idx] == 3:
            ema_outputs[idx,:,0:224-scale_mask[idx],0:224-scale_mask[idx]] = torch.from_numpy(new_img[scale_mask[idx]:, scale_mask[idx]:, :].transpose((2, 0, 1)))


    return ema_outputs.cuda()
#===========================================