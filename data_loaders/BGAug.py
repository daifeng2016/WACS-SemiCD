import torch
import numpy as np
import torchvision.transforms as transforms
from . import augmentations
from PIL import Image
import os
from .transform import resize,crop,hflip,vflip,normalize

def BGAware(img1, img2, uimg1, uimg2, mask):#uimg1:[4,3,256,256],mask:[256,256,1]
    weight = np.float32(np.random.dirichlet([1] * 3))
    # weight2 = np.float32(np.random.dirichlet([1]))
    m = np.float32(np.random.beta(1, 1))
    # n = np.float32(np.random.beta(1, 1))

    # aug1 = []
    # aug2 = []
    img1 = np.array(img1).astype(np.float32)#[256,256,3]
    img2 = np.array(img2).astype(np.float32)
    # B,_,_,_ = img1.shape
    # for i in range(4):  # output dimension
    mix1 = img1
    mix2 = img2
    # print(img1)
    mixed1 = np.zeros_like(img1)
    mixed2 = np.zeros_like(img1)
    # mix2 = np.zeros_like(img2[0])
    width = 3
    for j in range(width):
        depth = np.random.randint(1, 4)
        for _ in range(depth):  # mixed depth
            op = np.random.randint(0, 4)
            im1 = np.array(uimg1[op]).astype(np.float32)#[256,256,3]
            im2 = np.array(uimg2[op]).astype(np.float32)
            # mix1 = mask * mix1 + im1 * (1-mask)
            # mix2 = mask * mix2 + im2 * (1-mask)
            mix1 = mask * mix1 * (1 - m) + im1 * (1 - mask) * m
            mix2 = mask * mix2 * (1 - m) + im2 * (1 - mask) * m
        mixed1 += weight[j] * mix1
        mixed2 += weight[j] * mix2
    mixed11 = mixed1.copy()
    mixed22 = mixed2.copy()
    mixed11[mixed1 < 0] = 0#to avoid uuint8(-1)=255
    mixed11[mixed1 > 255] = 255
    mixed22[mixed2 < 0] = 0
    mixed22[mixed2 > 255] = 255
    return Image.fromarray(np.uint8(mixed11)), Image.fromarray(np.uint8(mixed22))
    # return torch.stack(aug1), torch.stack(aug2)


def BGMix(image1, image2, uimage1, uimage2, mask):
    preprocess = transforms.Compose([transforms.Resize((256, 256)),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                     ])
    aug_list = augmentations.augmentations

    ws = np.float32(
        np.random.dirichlet([1] * 3))
    m = np.float32(np.random.beta(1, 1))
    aug1 = []
    aug2 = []

    B, _, _, _ = image1.shape
    for index in range(B):
        aug = np.random.choice([True, False])#
        #note Image.fromarray must accept 8bit unsigned data
        img1 = Image.fromarray(image1[index].numpy())
        img2 = Image.fromarray(image2[index].numpy())
        cdm = mask[index].cpu().numpy().transpose(1, 2, 0)#[256,256,1]
        #cdm = mask[index].detach().cpu().numpy().transpose(1, 2, 0)

        img1_tensor = preprocess(img1)#[3,256,256]
        img2_tensor = preprocess(img2)#[3,256,256]

        if aug:

            mix = torch.zeros_like(img1_tensor)#[3,256,256]
            mix2 = torch.zeros_like(img2_tensor)#[3,256,256]

            for i in range(3):  # three paths
                image_aug = img1.copy()
                image_aug2 = img2.copy()

                depth = np.random.randint(1, 4)
                for _ in range(depth):
                    idx = np.random.choice([0, 1])
                    if idx == 0:
                        op_idx = np.random.choice(np.arange(len(aug_list)))
                        image_aug = aug_list[op_idx](image_aug, 1)
                        image_aug2 = aug_list[op_idx](image_aug2, 1)

                    else:
                        # augamented by UIP
                        image_aug, image_aug2 = BGAware(image_aug, image_aug2, uimage1, uimage2, cdm)

                # Preprocessing commutes since all coefficients are convex

                mix += ws[i] * preprocess(image_aug)
                mix2 += ws[i] * preprocess(image_aug2)

            mixed = (1 - m) * img1_tensor + m * mix
            mixed2 = (1 - m) * img2_tensor + m * mix2

            aug1.append(mixed)
            aug2.append(mixed2)

        else:
            aug1.append(img1_tensor)
            aug2.append(img2_tensor)

    cg1, cg2 = torch.stack(aug1).cuda(), torch.stack(aug2).cuda()  # change imag

    return cg1, cg2


def BGMix2(img1, img2, uimg1, uimg2, mask_c):#uimg1:[256,256,3],mask:[256,256,1]

    # preprocess = transforms.Compose([transforms.Resize((256, 256)),
    #                                  # transforms.RandomHorizontalFlip(p=0.5),
    #                                  # transforms.RandomVerticalFlip(p=0.5),
    #                                  transforms.ToTensor(),
    #                                  transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    #                                  ])

    B = img1.shape[0]
    mixed1=[]
    mixed2=[]
    mask_new=[]
    mask_c=mask_c[...,np.newaxis]#np.expand_dims(arr, 0)
    for i in range(B):
        idx_u=np.random.randint(0, B)
        mix1= mask_c[i] * img1[i] + uimg1[idx_u] * (1 - mask_c[i])
        mix2= mask_c[i] * img2[i] + uimg2[idx_u] * (1 - mask_c[i])
        #mask_temp=np.squeeze(mask_c[i])
        mix1=Image.fromarray(np.uint8(mix1))
        mix2= Image.fromarray(np.uint8(mix2))
        mask=Image.fromarray(np.uint8(np.squeeze(mask_c[i])))#np.squeeze(arr, 1)
        imgA, imgB, mask = resize(mix1, mix2, mask, (0.8, 1.2))  # [277,277]
        imgA, imgB, mask = crop(imgA, imgB, mask,256)  # [256,256]
        imgA, imgB, mask = hflip(imgA, imgB, mask, p=0.5)
        #imgA, imgB, mask = vflip(imgA, imgB, mask, p=0.5)
        imgA, mask = normalize(imgA, mask)
        imgB = normalize(imgB)
        mixed1.append(imgA)
        mixed2.append(imgB)
        mask_new.append(mask)

    return torch.stack(mixed1).cuda(),torch.stack(mixed2).cuda(),torch.stack(mask_new).cuda()



    # weight = np.float32(np.random.dirichlet([1] * 3))
    # # weight2 = np.float32(np.random.dirichlet([1]))
    # m = np.float32(np.random.beta(1, 1))
    # # n = np.float32(np.random.beta(1, 1))
    #
    # # aug1 = []
    # # aug2 = []
    # img1 = np.array(img1).astype(np.float32)#[256,256,3]
    # img2 = np.array(img2).astype(np.float32)
    # # B,_,_,_ = img1.shape
    # # for i in range(4):  # output dimension
    # mix1 = img1
    # mix2 = img2
    # # print(img1)
    # mixed1 = np.zeros_like(img1)
    # mixed2 = np.zeros_like(img1)
    # # mix2 = np.zeros_like(img2[0])
    # width = 3
    # B=uimg1.shape[0]
    # for j in range(width):
    #     depth = np.random.randint(1, 4)
    #     for _ in range(depth):  # mixed depth
    #         op = np.random.randint(0, B)
    #         im1 = np.array(uimg1[op]).astype(np.float32)#[256,256,3]
    #         im2 = np.array(uimg2[op]).astype(np.float32)
    #         # mix1 = mask * mix1 + im1 * (1-mask)
    #         # mix2 = mask * mix2 + im2 * (1-mask)
    #         mix1 = mask * mix1 * (1 - m) + im1 * (1 - mask) * m
    #         mix2 = mask * mix2 * (1 - m) + im2 * (1 - mask) * m
    #     mixed1 += weight[j] * mix1
    #     mixed2 += weight[j] * mix2
    # # aug1.append(mixed1)
    # # aug2.append(mixed1)
    # # print(mixed1.shape)
    # #return mixed1,mixed2
    # # mixed1=0 if mixed1<0 else mixed1
    # # mixed1 = 255 if mixed1 > 255 else mixed1
    # #
    # # mixed2 = 0 if mixed2 < 0 else mixed2
    # # mixed2 = 255 if mixed2 > 255 else mixed2
    # mixed11=mixed1.copy()
    # mixed22=mixed2.copy()
    # mixed11[mixed1<0]=0
    # mixed11[mixed1>255]=255
    # mixed22[mixed2 < 0] = 0
    # mixed22[mixed2 > 255] = 255
    #return Image.fromarray(np.uint8(mixed11)), Image.fromarray(np.uint8(mixed22))
    # return torch.stack(aug1), torch.stack(aug2)




if __name__ == '__main__':

    #device = torch.device("cpu")
    # imgs1 = torch.rand(4, 3, 256, 256).to(device)
    # imgs2 = torch.rand(4, 3, 256, 256).to(device)
    # masks = torch.rand(4, 1, 256, 256)
    # masks = masks.to(device)
    # mixed1,mixed2=BGAware(imgs1,imgs2,imgs1,imgs2,masks.data.numpy())

    # device=torch.device("cpu")
    # imgs1 = torch.rand(4, 3, 256, 256).to(device)
    # imgs2 = torch.rand(4, 3, 256, 256).to(device)
    # masks = torch.rand(4, 1, 256, 256)
    # masks = masks.to(device)
    # mixed1, mixed2 = BGMix(imgs1, imgs2, imgs1, imgs2, masks.data.numpy())






    device=torch.device("cpu")
    imgs1=torch.randint(0,256,(4, 256, 256,3)).byte().to(device)
    imgs2 = torch.randint(0, 256, (4, 256, 256,3)).byte().to(device)

    masks = torch.rand(4, 1, 256, 256)
    masks = masks.to(device)
    mixed1, mixed2 = BGMix(imgs1, imgs2, imgs1, imgs2, masks)

    print(mixed1.shape)
