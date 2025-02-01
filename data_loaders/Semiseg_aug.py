from . import transformsgpu
from . import transformsmask

def augmentationTransform(parameters, data=None, target=None, probs=None, jitter_vale=0.4, min_sigma=0.2, max_sigma=2., ignore_label=255):
    """
    Args:
        parameters: dictionary with the augmentation configuration
        data: BxCxWxH input data to augment
        target: BxWxH labels to augment
        probs: BxWxH probability map to augment
        jitter_vale:  jitter augmentation value
        min_sigma: min sigma value for blur
        max_sigma: max sigma value for blur
        ignore_label: value for ignore class

    Returns:
            augmented data, target, probs
    """
    assert ((data is not None) or (target is not None))
    if "Mix" in parameters:
        data, target, probs = transformsgpu.mix(mask=parameters["Mix"], data=data, target=target, probs=probs)

    if "RandomScaleCrop" in parameters:
        data, target, probs = transformsgpu.random_scale_crop(scale=parameters["RandomScaleCrop"], data=data,
                                                              target=target, probs=probs, ignore_label=ignore_label)
    if "flip" in parameters:
        data, target, probs = transformsgpu.flip(flip=parameters["flip"], data=data, target=target, probs=probs)

    if "ColorJitter" in parameters:
        data, target, probs = transformsgpu.colorJitter(colorJitter=parameters["ColorJitter"], data=data, target=target,
                                                        probs=probs, s=jitter_vale)
    if "GaussianBlur" in parameters:
        data, target, probs = transformsgpu.gaussian_blur(blur=parameters["GaussianBlur"], data=data, target=target,
                                                          probs=probs, min_sigma=min_sigma, max_sigma=max_sigma)

    if "Grayscale" in parameters:
        data, target, probs = transformsgpu.grayscale(grayscale=parameters["Grayscale"], data=data, target=target,
                                                      probs=probs)
    if "Solarize" in parameters:
        data, target, probs = transformsgpu.solarize(solarize=parameters["Solarize"], data=data, target=target,
                                                     probs=probs)

    return data, target,


def augment_samples(images, labels, probs, do_classmix, batch_size, ignore_label, weak = False):
    """
    Perform data augmentation

    Args:
        images: BxCxWxH images to augment
        labels:  BxWxH labels to augment
        probs:  BxWxH probability maps to augment
        do_classmix: whether to apply classmix augmentation
        batch_size: batch size
        ignore_label: ignore class value
        weak: whether to perform weak or strong augmentation

    Returns:
        augmented data, augmented labels, augmented probs

    """

    if do_classmix:
        # ClassMix: Get mask for image A
        for image_i in range(batch_size):  # for each image
            classes = torch.unique(labels[image_i])  # get unique classes in pseudolabel A
            nclasses = classes.shape[0]

            # remove ignore class
            if ignore_label in classes and len(classes) > 1 and nclasses > 1:
                classes = classes[classes != ignore_label]
                nclasses = nclasses - 1

            if dataset == 'pascal_voc':  # if voc dataaset, remove class 0, background
                if 0 in classes and len(classes) > 1 and nclasses > 1:
                    classes = classes[classes != 0]
                    nclasses = nclasses - 1

            # pick half of the classes randomly
            classes = (classes[torch.Tensor(
                np.random.choice(nclasses, int(((nclasses - nclasses % 2) / 2) + 1), replace=False)).long()]).cuda()

            # acumulate masks
            if image_i == 0:
                MixMask = transformmasks.generate_class_mask(labels[image_i], classes).unsqueeze(0).cuda()
            else:
                MixMask = torch.cat(
                    (MixMask, transformmasks.generate_class_mask(labels[image_i], classes).unsqueeze(0).cuda()))


        params = {"Mix": MixMask}
    else:
        params = {}

    if weak:
        params["flip"] = random.random() < 0.5
        params["ColorJitter"] = random.random() < 0.2
        params["GaussianBlur"] = random.random() < 0.
        params["Grayscale"] = random.random() < 0.0
        params["Solarize"] = random.random() < 0.0
        if random.random() < 0.5:
            scale = random.uniform(0.75, 1.75)
        else:
            scale = 1
        params["RandomScaleCrop"] = scale

        # Apply strong augmentations to unlabeled images
        image_aug, labels_aug, probs_aug = augmentationTransform(params,
                                                                 data=images, target=labels,
                                                                 probs=probs, jitter_vale=0.125,
                                                                 min_sigma=0.1, max_sigma=1.5,
                                                                 ignore_label=ignore_label)
    else:
        params["flip"] = random.random() < 0.5
        params["ColorJitter"] = random.random() < 0.8
        params["GaussianBlur"] = random.random() < 0.2
        params["Grayscale"] = random.random() < 0.0
        params["Solarize"] = random.random() < 0.0
        if random.random() < 0.80:
            scale = random.uniform(0.75, 1.75)
        else:
            scale = 1
        params["RandomScaleCrop"] = scale

        # Apply strong augmentations to unlabeled images
        image_aug, labels_aug, probs_aug = augmentationTransform(params,
                                                                 data=images, target=labels,
                                                                 probs=probs, jitter_vale=0.25,
                                                                 min_sigma=0.1, max_sigma=1.5,
                                                                 ignore_label=ignore_label)

    return image_aug, labels_aug, probs_aug, params