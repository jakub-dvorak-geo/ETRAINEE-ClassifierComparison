"""Set of functions for visualisations."""
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from time import gmtime, strftime

COLOR_LIST = ['white', 'red', 'green', 'yellow', 'orange', 'pink',
              'blue', 'cyan', 'black', 'grey']
CMAP = ListedColormap(COLOR_LIST)
CLASS_NAMES = ['No Data', 'af', 'afs', 'bor', 'desch', 'klec', 'nard', 'sut',
               'vres', 'vyfuk']
CLASS_NAMES = ['No Data', 'vyfoukávané alpinské trávníky',
               'vyfoukávané alpinské trávníky',
               'subalpínská brusnicová vegetace', 'metlice trsnatá',
               'kosodřevina', 'smilka tuhá', 'kamenná moře',
               'alpínská vřesoviště', 'vyfoukávané alpinské trávníky']


def _image_show(raster, title='Natural color composite'):
    """Show a figure based on a hyperspectral raster."""
    plt.imshow(raster/3000)
    plt.title(title)
    plt.axis('off')


def _class_show(raster, title):
    """Show a figure based on a classification."""
    plt.imshow(raster, cmap=CMAP)
    # plt.colorbar(ticks=(np.linspace(0.5, 8.5, 10)))
    plt.title(title)
    plt.axis('off')


def show_img_ref(hs_img, gt_img):
    """Show the hyperspectral image and a training reference."""
    plt.figure(figsize=[16, 8])
    plt.subplot(1, 2, 1)
    _image_show(hs_img)
    plt.subplot(1, 2, 2)
    _class_show(gt_img, 'Reference data')

    for label, color in zip(CLASS_NAMES, COLOR_LIST):
        plt.plot(0, 0, 's', label=label,
                 color=color, markeredgecolor='black')
    plt.legend()


def show_spectral_curve(tile_dict, tile_num,
                        title='Spectral curve for pixel #'):
    """Show a figure of the spectal curve."""
    x = np.linspace(404, 997, num=54)
    if len(tile_dict["imagery"].shape) == 4:
        y = tile_dict["imagery"][tile_num, 0, 0, :]
        lbl = tile_dict["reference"][tile_num, 0, 0, :][0] + 1
    elif len(tile_dict["imagery"].shape) == 3:
        y = tile_dict["imagery"][tile_num, 0, :]
        lbl = tile_dict["reference"][tile_num] + 1
    else:
        print('The input data is in an incompatible shape.')

    plt.plot(x, y, label=f'{CLASS_NAMES[lbl]}')
    plt.title(f'{title} {tile_num}')
    plt.xlabel('Wavelength [nm]')
    plt.legend(bbox_to_anchor=(0.5, 0.89), loc='lower center')


def show_augment_spectral(tile_dict, tile_num, aug_funct):
    """Show a figure of the original spectal curve and the augmented curve."""
    plt.figure(figsize=[8, 4])
    plt.subplot(1, 2, 1)
    show_spectral_curve(tile_dict, tile_num,
                        title='Original spectral curve for pixel #')
    plt.subplot(1, 2, 2)
    tensor_obs = torch.from_numpy(tile_dict["imagery"])
    tensor_gt = torch.from_numpy(tile_dict["reference"])
    aug_obs, aug_gt = aug_funct(tensor_obs, tensor_gt)
    aug_dict = {'imagery': aug_obs, 'reference': aug_gt}
    show_spectral_curve(aug_dict, tile_num,
                        title='Augmented spectral curve for pixel #')


def show_augment_spatial(tile_dict, tile_num, aug_funct):
    """Show a figure of the original and the augmented RGB composite."""
    img_rgb = tile_dict['imagery'][tile_num, [25, 15, 5], :, :]
    img_rgb_transposed = img_rgb.transpose((1, 2, 0))
    tile_gt = tile_dict['reference'][tile_num, :, :]
    plt.figure(figsize=[15, 5])

    plt.subplot(1, 3, 1)
    _image_show(img_rgb_transposed*3000, title='Original RGB composite')

    plt.subplot(1, 3, 2)
    img_hs = tile_dict['imagery'][tile_num, :, :, :]
    img_augmented, _ = aug_funct(torch.from_numpy(img_hs),
                                 torch.from_numpy(tile_gt[None, :, :]))
    print(img_augmented.shape)
    img_augmented_np = np.array(img_augmented)
    img_aug_trans = img_augmented_np[0, [25, 15, 5], :, :].transpose(1, 2, 0)

    _image_show(np.array(img_aug_trans)*3000, title='Augmented RGB composite')

    plt.subplot(1, 3, 3)
    _class_show(tile_gt, 'Original reference data')

    for label, color in zip(CLASS_NAMES, COLOR_LIST):
        plt.plot(0, 0, 's', label=label,
                 color=color, markeredgecolor='black')
    plt.legend()


def show_classified(hs_img, gt_img, class_img):
    """Compare the classification result to the reference data."""
    plt.figure(figsize=[15, 5])
    plt.subplot(1, 3, 1)
    _image_show(hs_img)

    plt.subplot(1, 3, 2)
    _class_show(gt_img, 'Reference data')

    for label, color in zip(CLASS_NAMES, COLOR_LIST):
        plt.plot(0, 0, 's', label=label,
                 color=color, markeredgecolor='black')
    plt.legend()

    plt.subplot(1, 3, 3)
    _class_show(class_img, 'Classified data')


def sec_to_hms(sec):
    """Convert seconds to hours, minutes, seconds."""
    ty_res = gmtime(sec)
    res = strftime("%Hh, %Mm, %Ss", ty_res)
    return str(res)
