import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import cv2
import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from torch.autograd import Variable
import glob
import matplotlib.cm as cm
import matplotlib.colors as co


def batchnorm(num_features):
    """
    https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm2d.html
    """
    return nn.BatchNorm2d(num_features, affine=True, eps=1e-5, momentum=0.1)

def convbnrelu(in_channels, out_channels, kernel_size, stride=1, groups=1, act=True):
    "conv-batchnorm-relu"
    if act:
        return nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=int(kernel_size / 2.), groups=groups, bias=False),
                             batchnorm(out_channels),
                             nn.ReLU6(inplace=True))
    else:
        return nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=int(kernel_size / 2.), groups=groups, bias=False),
                             batchnorm(out_channels))

def conv1x1(in_channels, out_channels, stride=1, groups=1, bias=False,):
    "1x1 Convolution: Pointwise"
    return nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0, bias=bias, groups=groups)

def conv3x3(in_channels, out_channels, stride=1, dilation=1, groups=1, bias=False):
    """3x3 Convolution: Depthwise:
    https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
    """
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=dilation, dilation=dilation, bias=bias, groups=groups)

class InvertedResidualBlock(nn.Module):
    """Inverted Residual Block from https://arxiv.org/abs/1801.04381"""
    def __init__(self, in_planes, out_planes, expansion_factor, stride=1):
        super().__init__() # Python 3
        intermed_planes = in_planes * expansion_factor
        self.residual = (in_planes == out_planes) and (stride == 1) # Boolean/Condition
        # TODO: Create a Sequential Model that will implement the Inverted Residual Block
        self.output = nn.Sequential(convbnrelu(in_planes, intermed_planes, kernel_size=1),
                                    convbnrelu(intermed_planes, intermed_planes, kernel_size=3, stride=stride, groups=intermed_planes),
                                    convbnrelu(intermed_planes, out_planes, kernel_size=1, act=False))

    def forward(self, x):
        #residual = x
        out = self.output(x)
        if self.residual:
            return (out + x)#+residual
        else:
            return out

class CRPBlock(nn.Module):
    """CRP definition"""
    def __init__(self, in_planes, out_planes, n_stages, groups=False):
        super().__init__() #Python 3
        for i in range(n_stages):
            setattr(self, '{}_{}'.format(i + 1, 'outvar_dimred'),
                    conv1x1(in_planes if (i == 0) else out_planes,
                            out_planes, stride=1,
                            bias=False, groups=in_planes if groups else 1)) #setattr(object, name, value)

        self.stride = 1
        self.n_stages = n_stages
        self.maxpool = nn.MaxPool2d(kernel_size=5, stride=1, padding=2) # set padding to 2, so that the spatial size would stay intact

    def forward(self, x):
        top = x
        for i in range(self.n_stages):
            top = self.maxpool(top)
            top = getattr(self, '{}_{}'.format(i + 1, 'outvar_dimred'))(top)#getattr(object, name[, default])
            x = top + x
        return x

class HydraNet(nn.Module):
    def __init__(self):
        #super(HydraNet, self).__init__() # Python2
        super().__init__() # Python 3
        self.num_tasks = 2
        self.num_classes = 6  # the pre-trained only has 6 classes on KITTI dataset
        # Encoder: MOBILENETV2
        self.mobilenet_config = [[1, 16, 1, 1], # [expansion rate, output channels, number of repeats, stride]
                                 [6, 24, 2, 2],
                                 [6, 32, 3, 2],
                                 [6, 64, 4, 2],
                                 [6, 96, 3, 1],
                                 [6, 160, 3, 2],
                                 [6, 320, 1, 1],
                                 ]
        self.in_channels = 32 # number of input channels
        self.num_layers = len(self.mobilenet_config)
        self.layer1 = convbnrelu(3, self.in_channels, kernel_size=3, stride=2)
        self.c_layer = 2
        # define mobilenet
        for t,c,n,s in (self.mobilenet_config):
            layers = []
            for idx in range(n):
                layers.append(InvertedResidualBlock(self.in_channels, c, expansion_factor=t, stride=s if idx == 0 else 1))
                self.in_channels = c
            setattr(self, 'layer{}'.format(self.c_layer), nn.Sequential(*layers)) # setattr(object, name, value)
            self.c_layer += 1

        # Decoder: lightweight refinenet
        self.conv8 = conv1x1(320, 256, bias=False)
        self.conv7 = conv1x1(160, 256, bias=False)
        self.conv6 = conv1x1(96, 256, bias=False)
        self.conv5 = conv1x1(64, 256, bias=False)
        self.conv4 = conv1x1(32, 256, bias=False)
        self.conv3 = conv1x1(24, 256, bias=False)
        self.crp4 = self._make_crp(256, 256, 4, groups=False)
        self.crp3 = self._make_crp(256, 256, 4, groups=False)
        self.crp2 = self._make_crp(256, 256, 4, groups=False)
        self.crp1 = self._make_crp(256, 256, 4, groups=True)

        self.conv_adapt4 = conv1x1(256, 256, bias=False)
        self.conv_adapt3 = conv1x1(256, 256, bias=False)
        self.conv_adapt2 = conv1x1(256, 256, bias=False)

        self.pre_depth = conv1x1(256, 256, groups=256, bias=False)  #TODO: Define the Purple Pre-Head for Depth
        self.depth = conv3x3(256, 1, bias=True) #TODO: Define the Final layer of Depth. For depth, we only has 1 value that represent depth
        self.pre_segm = conv1x1(256, 256, groups=256, bias=False)   #TODO: Call the Purple Pre-Head for Segm
        self.segm = conv3x3(256, self.num_classes, bias=True)  #TODO: Define the Final layer of Segmentation
        self.relu = nn.ReLU6(inplace=True)  #TODO: Define a RELU 6 Operation

        if self.num_tasks == 3:
            #TODO: Create a Head for Surface Normals
            self.pre_normal = conv1x1(256, 256, groups=256, bias=False)  #TODO: Define the Purple Pre-Head for Surface Normals
            self.normal = conv3x3(256, 3, bias=True) #TODO: Define the Final layer of Surface Normals. For Surface Normals, we will predict the x, y and z components of the normal at each pixel.

    def _make_crp(self, in_planes, out_planes, stages, groups=False):
        # TODO: Call a CRP BLOCK in Layers
        layers = [CRPBlock(in_planes, out_planes, stages, groups=groups)]
        return nn.Sequential(*layers)

    def forward(self, x):
        # MOBILENET V2
        x = self.layer1(x)
        x = self.layer2(x)    # 16, x / 2
        l3 = self.layer3(x)   # 24, x / 4
        l4 = self.layer4(l3)  # 32, x / 8
        l5 = self.layer5(l4)  # 64, x / 16
        l6 = self.layer6(l5)  # 96, x / 16
        l7 = self.layer7(l6)  # 160, x / 32
        l8 = self.layer8(l7)  # 320, x / 32

        # LIGHT-WEIGHT REFINENET
        l8 = self.conv8(l8)
        l7 = self.conv7(l7)
        l7 = self.relu(l8 + l7)
        l7 = self.crp4(l7)
        l7 = self.conv_adapt4(l7)   # Fusion-LW - conv 1x1
        l7 = nn.Upsample(size=l6.size()[2:], mode='bilinear', align_corners=False)(l7)  # Fusion-LW - Upsample,  l6 = [B, C, H, W]

        l6 = self.conv6(l6)
        l5 = self.conv5(l5)
        l5 = self.relu(l5 + l6 + l7)  # Fusion-LW - Sum
        l5 = self.crp3(l5)
        l5 = self.conv_adapt3(l5)   # Fusion-LW - conv 1x1
        l5 = nn.Upsample(size=l4.size()[2:], mode='bilinear', align_corners=False)(l5)  # Fusion-LW - Upsample

        l4 = self.conv4(l4)
        l4 = self.relu(l5 + l4)   # Fusion-LW - Sum
        l4 = self.crp2(l4)
        l4 = self.conv_adapt2(l4) # Fusion-LW - conv 1x1
        l4 = nn.Upsample(size=l3.size()[2:], mode='bilinear', align_corners=False)(l4)  # Fusion-LW - Upsample

        l3 = self.conv3(l3)
        l3 = self.relu(l3 + l4)   # Fusion-LW - Sum
        l3 = self.crp1(l3)

        # HEADS
        #TODO: Design the 3 Heads
        out_segm = self.pre_segm(l3)
        out_segm = self.relu(out_segm)
        out_segm = self.segm(out_segm)

        out_d = self.pre_depth(l3)
        out_d = self.relu(out_d)
        out_d = self.depth(out_d)

        if self.num_tasks == 3:
            out_n = self.pre_normal(l3)
            out_n = self.relu(out_n)
            out_n = self.normal(out_n)
            return out_segm, out_d, out_n
        else:
            return out_segm, out_d

"""
Functions below are for inference
"""
def prepare_img(img):
    """
    Preprocess Images
    """
    IMG_SCALE  = 1./255
    IMG_MEAN = np.array([0.485, 0.456, 0.406]).reshape((1, 1, 3))
    IMG_STD = np.array([0.229, 0.224, 0.225]).reshape((1, 1, 3))
    return (img * IMG_SCALE - IMG_MEAN) / IMG_STD

def pipeline(hydranet, img, device):
    """
    inference
    """
    # Pre-processing and post-processing constants #
    CMAP = np.load('cmap_kitti.npy')
    NUM_CLASSES = 6
    with torch.no_grad():
        img_var = torch.from_numpy(prepare_img(img)).permute(2, 0, 1).unsqueeze(0).float() #Put the Image in PYTorch Variable
        img_var = img_var.to(device)
        segm, depth = hydranet(img_var) # Call the HydraNet. segm = [B, 6, 93, 306], depth = [B, 1, 93, 306]
        segm = cv2.resize(segm[0, :NUM_CLASSES].cpu().numpy().transpose(1,2,0),
                          img.shape[:2][::-1],
                          interpolation=cv2.INTER_CUBIC)#PostProcess / Resize
        depth = cv2.resize(depth[0, 0].cpu().numpy(),
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

def save_video(path, model, device):
    video_files = sorted(glob.glob(path))
    # Run the pipeline
    result_video = []
    for idx, img_path in enumerate(video_files):
        image = np.array(Image.open(img_path))
        h, w, _ = image.shape
        depth, seg = pipeline(model, image, device)
        filename = 'data/seg' + str(idx) + ".jpg"
        cv2.imwrite(filename, seg)
        # result_video.append(cv2.cvtColor(cv2.vconcat([image, seg, depth_to_rgb(depth)]), cv2.COLOR_BGR2RGB))

    # out = cv2.VideoWriter('campus.mp4',cv2.VideoWriter_fourcc(*'mp4v'), 15, (w,3*h))
    #
    # for i in range(len(result_video)):
    #     out.write(result_video[i])
    # out.release()

if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = HydraNet().to(device)

    ckpt = torch.load('ExpKITTI_joint.ckpt')
    model.load_state_dict(ckpt['state_dict'])

    # load images
    images_files = glob.glob('data/*.png')
    idx = np.random.randint(0, len(images_files))
    img_path = images_files[idx]
    img = np.array(Image.open(img_path))

    # prediction
    depth, segm = pipeline(model, img, device)

    # show and save image result
    # display_result(img, depth, segm, is_horizontal=True)

    # write images as video
    images_path = "data/*.png"
    save_video(images_path, model, device)


    """
    ## 3D Segmentation
    """
    import open3d as o3d

    """
    RGBD - Fuse the RGB Image and the Depth Map
    """
    # make a RGBD image from color and depth image
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(o3d.geometry.Image(img), o3d.geometry.Image(depth_rgb), convert_rgb_to_intensity=False)
    # print(rgbd)

    # create_from_color_and_depth create a RGBD image. The color image is converted into a grayscale image(if convert_rgb_to_intensity=True),
    # stored in float ranged in [0, 1]. The depth image is stored in float, representing the depth value in meters.

    """
    Build a Point Cloud based on this. For that, we'll need the camera's intrinsic parameters.
    """
    intrinsics = o3d.camera.PinholeCameraIntrinsic(width = 1242, height = 375, fx = 721., fy = 721., cx = 609., cy = 609.)

    #TODO: Convert RGBD image to a Point Cloud
    point_cloud = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsics)
    o3d.io.write_point_cloud("test.pcd", point_cloud)

    """
    3D Segmentation — Fuse the Segmentation Map with the Depth Map
    """

    segm_3d = o3d.geometry.RGBDImage.create_from_color_and_depth(o3d.geometry.Image(segm), o3d.geometry.Image(depth_rgb), convert_rgb_to_intensity=False)

    point_cloud2 = o3d.geometry.PointCloud.create_from_rgbd_image(segm_3d, intrinsics)

    o3d.io.write_point_cloud("test_segm.pcd", point_cloud2)

    # print(point_cloud2)
    # print(np.asarray(point_cloud2.points))
