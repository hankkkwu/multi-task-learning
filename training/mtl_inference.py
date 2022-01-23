import cv2
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import glob
import matplotlib.cm as cm
import matplotlib.colors as co
from mtl_training import MobileNetv2, MTLWRefineNet

def prepare_img(img):
    """
    Preprocess Images
    """
    IMG_SCALE  = 1./255
    IMG_MEAN = np.array([0.485, 0.456, 0.406]).reshape((1, 1, 3))
    IMG_STD = np.array([0.229, 0.224, 0.225]).reshape((1, 1, 3))
    return (img * IMG_SCALE - IMG_MEAN) / IMG_STD

def pipeline(hydranet, img, device, CMAP):
    """
    inference
    """
    # Pre-processing and post-processing constants #
    NUM_CLASSES = 40
    with torch.no_grad():
        img_var = torch.from_numpy(prepare_img(img)).permute(2, 0, 1).unsqueeze(0).float() #Put the Image in PYTorch Variable
        img_var = img_var.to(device)
        outputs = hydranet(img_var) # Call the HydraNet. segm = [B, 6, h, w], depth = [B, 1, h, w]
        segm = cv2.resize(outputs[0][0, :NUM_CLASSES].cpu().numpy().transpose(1,2,0),
                          img.shape[:2][::-1],
                          interpolation=cv2.INTER_CUBIC)#PostProcess / Resize
        depth = cv2.resize(outputs[1][0, 0].cpu().numpy(),
                           img.shape[:2][::-1],
                           interpolation=cv2.INTER_CUBIC) #PostProcess / Resize
        segm = CMAP[segm.argmax(axis=2)].astype(np.uint8) #Use the CMAP
        depth = np.abs(depth) #Take the Absolute Value]
        return depth, segm

def depth_to_rgb(depth):
    """
    convert depth map to rgb image
    """
    normalizer = co.Normalize(vmin=0, vmax=80)
    mapper = cm.ScalarMappable(norm=normalizer, cmap='plasma')
    colormapped_im = (mapper.to_rgba(depth)[:, :, :3] * 255).astype(np.uint8)
    return colormapped_im

def display_result(img, depth, segm, is_horizontal=False):
    """
    display result image
    """
    if is_horizontal:
        f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(30,20))
        ax1.imshow(img)
        ax1.set_title('Original', fontsize=30)
        ax2.imshow(segm)
        ax2.set_title('Predicted Segmentation', fontsize=30)
        ax3.imshow(depth, cmap="plasma", vmin=0, vmax=80)
        ax3.set_title("Predicted Depth", fontsize=30)
        plt.savefig('result_horizontal.png')
        plt.show()
    else:
        depth_rgb = depth_to_rgb(depth)
        new_img = np.vstack((img, segm, depth_rgb))
        plt.imshow(new_img)
        plt.savefig('result_vertical.png')
        plt.show()

if __name__ == '__main__':
    # Inference Challenge
    # load an image from the test dataset and run your model on it.
    # MEGA POINTS — Load a video, and implement a video pipeline

    # Load model
    encoder = MobileNetv2()
    num_classes = (40, 1)
    decoder = MTLWRefineNet(encoder._out_c, num_classes)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = nn.Sequential(encoder, decoder).to(device)
    ckpt = torch.load("checkpoints/checkpoint.pth.tar")
    model.load_state_dict(ckpt['state_dict'])

    # Load image
    images_files = glob.glob('nyud/rgb/*.png')
    idx = np.random.randint(0, len(images_files))
    img_path = images_files[idx]
    img = np.array(Image.open(img_path))

    # Load color Map for Semantic Segmentation
    CMAP = np.load('nyud/cmap_nyud.npy')
    # prediction
    depth, segm = pipeline(model, img, device, CMAP)
    # show and save image result
    display_result(img, depth, segm, is_horizontal=False)
