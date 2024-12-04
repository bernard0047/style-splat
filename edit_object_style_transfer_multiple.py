# Copyright (C) 2024, Style-Splat
# All rights reserved.

# ------------------------------------------------------------------------
# Modified from codes in Gaussian-Splatting and Gaussian-Grouping
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# Gaussian-Grouping research group, https://github.com/lkeab/gaussian-grouping

from argparse import ArgumentParser
from random import randint
from os import makedirs
from tqdm import tqdm
from PIL import Image
import open3d as o3d
import numpy as np
import torchvision
import torch
import lpips
import json
import cv2
import os

from arguments import ModelParams, PipelineParams, OptimizationParams, get_combined_args
from utils.general_utils import safe_state, PILtoTorch, points_inside_convex_hull
from render import feature_to_rgb, visualize_obj
from utils.loss_utils import masked_l1_loss
from gaussian_renderer import GaussianModel
from utils.nnfm_loss import NNFMLoss
from gaussian_renderer import render
from scene import Scene

def cleanPointCloud(points, mask3d):
    mask3d = mask3d.bool().squeeze().cpu().numpy() # N,
    points = points.detach().cpu().numpy() # N x 3
    print("Before: ", np.sum(mask3d))
    object_points = points[mask3d]
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(object_points)
    cl, ind = point_cloud.remove_statistical_outlier(nb_neighbors=75, std_ratio=0.5)
    inlier_mask = np.zeros(object_points.shape[0], dtype=bool)
    inlier_mask[ind] = True
    updated_mask = mask3d.copy()
    updated_mask[mask3d] = inlier_mask
    return updated_mask

def finetune_style(opt, model_path, iteration, views, gaussians, pipeline, background, classifier, selected_obj_ids, cameras_extent, removal_thresh, finetune_iteration):
    iterations = finetune_iteration
    progress_bar = tqdm(range(iterations), desc="Finetuning progress")
    nnfm_loss_fn = NNFMLoss("cuda")
    
    # selected_obj_ids = selected_obj_ids[2:]
    print("OPTIMIZING: ", selected_obj_ids)
    mask3ds = []
    # mask3d_combine = None
    with torch.no_grad():
        for selected_obj_id in selected_obj_ids:
            logits3d = classifier(gaussians._objects_dc.permute(2,0,1))
            prob_obj3d = torch.softmax(logits3d,dim=0)
            mask = prob_obj3d[[selected_obj_id], :, :] > 0.95
            mask3d = mask.any(dim=0).squeeze()
            mask3d = torch.Tensor(cleanPointCloud(gaussians._xyz, mask3d)).to(gaussians._xyz.device).bool()
            mask3ds.append(mask3d[:,None,None])

    gaussians.style_transfer_setup(opt, None)


    viewpoint_stack = views.copy()
    viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))
    gt_image = viewpoint_cam.original_image.cuda()

    style_images = []
    paths = ["style-ims/starry.jpg", "style-ims/venitian-canal.jpg", "style-ims/water-lillies.jpg"]#[2:]
    for i in range(len(selected_obj_ids)):
        style_image = Image.open(paths[i]).resize(gt_image.shape[1:])
        style_image = PILtoTorch(style_image, style_image.size, normalize=True).to("cuda")
        style_image = style_image.permute(0,2,1)
        style_images.append(style_image)
    
    for iteration in range(iterations):
        for i in range(len(selected_obj_ids)):

            viewpoint_stack = views.copy()
            viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))
            gt_image = viewpoint_cam.original_image.cuda()
            render_pkg = render(viewpoint_cam, gaussians, pipeline, background)
            image, viewspace_point_tensor, visibility_filter, radii, objects = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"], render_pkg["render_object"]

            # loss = 0
            mask2d = viewpoint_cam.objects == selected_obj_ids[i]
            mask2d = mask2d.unsqueeze(0).to(image.device)

            image, gt_image = image * mask2d, gt_image * mask2d
            im = (torch.clamp(image, min=0, max=1.0))#.unsqueeze(0)
            gt = (torch.clamp(gt_image, min=0, max=1.0))#.unsqueeze(0)
            lossdict = nnfm_loss_fn(outputs=im, styles=style_images[i].unsqueeze(0), contents=gt, loss_names=["nnfm_loss"])
            loss = lossdict['nnfm_loss']

            loss.backward()
            gaussians._features_dc.grad = gaussians._features_dc.grad * mask3ds[i]
            gaussians._features_rest.grad = gaussians._features_rest.grad * mask3ds[i]
            gaussians.optimizer.step()
            gaussians.optimizer.zero_grad(set_to_none=True)

        if iteration % 10 == 0:
            progress_bar.set_postfix({"Loss": f"{loss:.{7}f}"})
            progress_bar.update(10)
    progress_bar.close()
    
    # save gaussians
    point_cloud_path = os.path.join(model_path, "point_cloud_multiple/iteration_{}".format(iteration+1))
    gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))

    return gaussians

def render_set(model_path, name, iteration, views, gaussians, pipeline, background, classifier):
    render_path = os.path.join(model_path, name, "ours{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours{}".format(iteration), "gt")
    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)


    for idx, view in enumerate(tqdm(views[:50], desc="Rendering progress")):
        results = render(view, gaussians, pipeline, background)
        rendering = results["render"]
        gt = view.original_image[0:3, :, :]
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))


def style(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool, opt : OptimizationParams, select_obj_id : int, removal_thresh : float,  finetune_iteration: int):
    # 1. load gaussian checkpoint
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
    num_classes = dataset.num_classes
    print("Num classes: ",num_classes)
    classifier = torch.nn.Conv2d(gaussians.num_objects, num_classes, kernel_size=1)
    classifier.cuda()
    classifier.load_state_dict(torch.load(os.path.join(dataset.model_path,"point_cloud","iteration_"+str(scene.loaded_iter),"classifier.pth")))
    bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    # 2. style selected object
    gaussians = finetune_style(opt, dataset.model_path, scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background, classifier, select_obj_id, scene.cameras_extent, removal_thresh, finetune_iteration)

    # 3. render new result
    dataset.object_path = 'object_mask'
    dataset.images = 'images'
    scene = Scene(dataset, gaussians, load_iteration='_multiple/iteration_'+str(finetune_iteration), shuffle=False)
    with torch.no_grad():
        if not skip_train:
             render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background, classifier)

        if not skip_test:
             render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background, classifier)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    opt = OptimizationParams(parser)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")

    parser.add_argument("--config_file", type=str, default="config/object_style_transfer/counter.json", help="Path to the configuration file")


    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Read and parse the configuration file
    try:
        with open(args.config_file, 'r') as file:
            config = json.load(file)
    except FileNotFoundError:
        print(f"Error: Configuration file '{args.config_file}' not found.")
        exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Failed to parse the JSON configuration file: {e}")
        exit(1)

    args.num_classes = config.get("num_classes", 200)
    args.removal_thresh = config.get("removal_thresh", 0.3)
    args.select_obj_id = config.get("select_obj_id", [34])
    args.images = config.get("images", "images")
    args.object_path = config.get("object_path", "object_mask")
    args.resolution = config.get("r", 1)
    args.lambda_dssim = config.get("lambda_dlpips", 0.5)
    args.finetune_iteration = config.get("finetune_iteration", 2000)

    safe_state(args.quiet)
    style(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test, opt.extract(args), args.select_obj_id, args.removal_thresh, args.finetune_iteration)
