import numpy as np
import imageio
import torch
import os


def read_imagery(img_dir):
    """Reads the individual imagery patches and prepares them for """
    img_file_list = os.listdir(img_dir)
    img_list = []
        
    for file in img_file_list:
        img_patch = imageio.imread(os.path.join(img_dir, file)).astype(np.float32)
        img_patch = img_patch[:,:,:].transpose([2,0,1])
        img_patch = img_patch * 1/255
            
        img_list.append(img_patch)
        del img_patch

    img_features = np.stack(img_list, axis=0)
    return img_features

def read_patch(root_folder, cir, rgb, pan, mhs, gt=True):
    """Reads data from images as floats"""
    
    if cir:
        cir_features = read_imagery(os.path.join(root_folder, 'CIR'))
    if rgb:
        rgb_features = read_imagery(os.path.join(root_folder, 'RGB'))
    if mhs[0]:
        mhs_features = read_imagery(os.path.join(root_folder, 'MHS'))

    if pan:
        pan_file_list = os.listdir(os.path.join(root_folder, 'PAN'))
        pan_list = []
        for file in pan_file_list:
            pan_patch = imageio.imread(os.path.join(root_folder, 'PAN', file)).astype(np.float32)
            pan_patch = pan_patch * 1/255
            pan_patch = np.expand_dims(pan_patch, axis=0)
            pan_list.append(pan_patch)
            del pan_patch
        pan_features = np.stack(pan_list, axis=0)


    if cir and rgb:
        features = np.concatenate([cir_features, rgb_features], axis=1)
    elif cir:
        features = cir_features
    elif rgb:
        features = rgb_features
    elif pan:
        features = pan_features
    elif mhs:
        features = mhs_features
    else:
        print('No valid data input.')
    features = torch.from_numpy(features)
    
    
    if gt:
        gt_file_list = os.listdir(os.path.join(root_folder, 'GT'))
        gt_list = []

        for file in gt_file_list:
            gt_patch = imageio.imread(os.path.join(root_folder, 'GT', file)).astype(np.int64)
            # assigns 0 to classes 3 and above
            # gt_patch[gt_patch > 2] = 0
            
            gt_list.append(gt_patch[:,:])
            del gt_patch

        ground_truth = np.stack(gt_list, axis=0)
        ground_truth = torch.from_numpy(ground_truth)
    
    if gt:
        return features, ground_truth
    else:
        return features

def classify_and_export(model, in_features, results_path):
    for i, patch in enumerate(os.listdir(results_path)):
        in_patch = in_features[i,:,:,:]
        pred = model(in_patch[None,:,:,:].cuda()).cpu().detach().numpy()
        pred = pred[0,:,:,:].argmax(0).squeeze()

        imageio.imwrite(os.path.join(results_path, patch), pred.astype(np.uint8))
