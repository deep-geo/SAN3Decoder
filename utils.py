import os
import logging
import random

import numpy as np
import cv2
import torch
import albumentations as A

from typing import List
from albumentations.pytorch import ToTensorV2
from torch.nn import functional as F
from skimage.measure import label, regionprops
from segment_anything import SamAutomaticMaskGenerator

STEP = 15


def to_device(batch_input, device):
    device_input = {}
    for key, value in batch_input.items():
        if value is not None:
            if key == 'image' or key == 'label':
                device_input[key] = value.float().to(device)
            elif type(value) is list or type(value) is torch.Size:
                device_input[key] = value
            else:
                device_input[key] = value.to(device)
        else:
            device_input[key] = value
    return device_input


def get_boxes_from_mask(mask, box_num=1, std = 0.1, max_pixel = 5):
    """
    Args:
        mask: Mask, can be a torch.Tensor or a numpy array of binary mask.
        box_num: Number of bounding boxes, default is 1.
        std: Standard deviation of the noise, default is 0.1.
        max_pixel: Maximum noise pixel value, default is 5.
    Returns:
        noise_boxes: Bounding boxes after noise perturbation,
            returned as a torch.Tensor.
    """
    if isinstance(mask, torch.Tensor):
        mask = mask.numpy()
        
    label_img = label(mask)
    regions = regionprops(label_img)

    # Iterate through all regions and get the bounding box coordinates
    boxes = [tuple(region.bbox) for region in regions]

    # If the generated number of boxes is greater than the number of categories,
    # sort them by region area and select the top n regions
    if len(boxes) >= box_num:
        sorted_regions = sorted(regions, key=lambda x: x.area, reverse=True)[:box_num]
        boxes = [tuple(region.bbox) for region in sorted_regions]

    # If the generated number of boxes is less than the number of categories,
    # duplicate the existing boxes
    elif len(boxes) < box_num:
        num_duplicates = box_num - len(boxes)
        boxes += [boxes[i % len(boxes)] for i in range(num_duplicates)]

    # Perturb each bounding box with noise
    noise_boxes = []
    for box in boxes:
        y0, x0,  y1, x1 = box
        width, height = abs(x1 - x0), abs(y1 - y0)
        # Calculate the standard deviation and maximum noise value
        noise_std = min(width, height) * std
        max_noise = min(max_pixel, int(noise_std * 5))
         # Add random noise to each coordinate
        try:
            noise_x = np.random.randint(-max_noise, max_noise)
        except:
            noise_x = 0
        try:
            noise_y = np.random.randint(-max_noise, max_noise)
        except:
            noise_y = 0
        x0, y0 = x0 + noise_x, y0 + noise_y
        x1, y1 = x1 + noise_x, y1 + noise_y
        noise_boxes.append((x0, y0, x1, y1))

    return torch.as_tensor(noise_boxes, dtype=torch.float)


def get_random_edge_points(mask_val: int, mask: np.ndarray, point_num: int):
    mask_vals = np.unique(mask)
    if mask_val not in mask_vals:
        points = [(0, 0) for _ in range(point_num)]
    else:
        mask_copy = mask.copy()
        mask_copy[mask != mask_val] = 0
        mask_copy[mask == mask_val] = 255
        edges = cv2.Canny(mask_copy.astype(np.uint8), 100, 200)
        contours, _ = cv2.findContours(edges, cv2.RETR_TREE,
                                       cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            points = [(0, 0) for _ in range(point_num)]
        else:
            contour_points = contours[0].squeeze(1).tolist()
            points = random.choices(contour_points, k=point_num)

    mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    for x, y in points:
        cv2.circle(mask, (x, y), 2, (0, 0, 255), -1)

    # cv2.imshow("mask", mask)
    # cv2.imshow("msk_copy", mask_copy)
    # cv2.waitKey(0)

    return points


def get_edge_points_from_mask(mask_val: int, mask: np.ndarray, point_num=3):
    # count["total"] += 1
    # for test
    # img = mask / mask.max() * 255
    # img = img.astype(np.uint8)
    # img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    mask_vals = np.unique(mask)
    if len(mask_vals) == 1:
        points = [(0, 0) for _ in range(point_num)]
        return points

    other_mask_vals = [val for val in mask_vals if val != 0 and val != mask_val]
    if not other_mask_vals:
        points = get_random_edge_points(mask_val, mask, point_num)
    else:
        current_mask = (mask == mask_val).astype(np.uint8)
        kernel = np.ones((3, 3), np.uint8)
        current_mask = cv2.dilate(current_mask, kernel, iterations=1)
        edge_points_list = []
        for val in other_mask_vals:
            other_mask = (mask == val).astype(np.uint8)
            other_mask = cv2.dilate(other_mask, kernel, iterations=1)
            coords = np.argwhere(current_mask & other_mask)[:, ::-1]

            # for test
            # img[coords[:, 1], coords[:, 0], :] = [random.choice(range(255)),
            #                                       random.choice(range(255)),
            #                                       random.choice(range(255))]

            if len(coords) >= point_num:
                edge_points_list.append(coords)

        if not edge_points_list:
            points = get_random_edge_points(mask_val, mask, point_num)
        else:
            random_points = random.choice(edge_points_list)
            points = random.choices(random_points, k=point_num)

    # for test
    # if (sum(np.array(points)) > 0).any():
    #     print(points)
    #
    #     img[mask == mask_val] = [0, 255, 0]
    #     for point in points:
    #         # img[point[1], point[0], :] = [0, 0, 255]
    #         cv2.circle(img, (point[0], point[1]), radius=1, color=(0, 0, 255),
    #                    thickness=-1)
    #     cv2.imshow("img", img)
    #     cv2.waitKey(0)
    #return torch.tensor(points, dtype=torch.float)
    
    return torch.tensor(np.array(points), dtype=torch.float)


def select_random_points(pr, gt, point_num=9):
    """
    Selects random points from the predicted and ground truth masks and
    assigns labels to them.
    Args:
        pr (torch.Tensor): Predicted mask tensor.
        gt (torch.Tensor): Ground truth mask tensor.
        point_num (int): Number of random points to select. Default is 9.
    Returns:
        batch_points (np.array): Array of selected points coordinates (x, y)
            for each batch.
        batch_labels (np.array): Array of corresponding labels
            (0 for background, 1 for foreground) for each batch.
    """
    pred, gt = pr.data.cpu().numpy(), gt.data.cpu().numpy()
    error = np.zeros_like(pred)
    error[pred != gt] = 1

    # error = np.logical_xor(pred, gt)
    batch_points = []
    batch_labels = []
    for j in range(error.shape[0]):
        one_pred = pred[j].squeeze(0)
        one_gt = gt[j].squeeze(0)
        one_erroer = error[j].squeeze(0)

        indices = np.argwhere(one_erroer == 1)
        if indices.shape[0] > 0:
            selected_indices = indices[np.random.choice(indices.shape[0], point_num, replace=True)]
        else:
            indices = np.random.randint(0, 256, size=(point_num, 2))
            selected_indices = indices[np.random.choice(indices.shape[0], point_num, replace=True)]
        selected_indices = selected_indices.reshape(-1, 2)

        points, labels = [], []
        for i in selected_indices:
            x, y = i[0], i[1]
            if one_pred[x,y] == 0 and one_gt[x,y] == 1:
                label = 1
            elif one_pred[x,y] == 1 and one_gt[x,y] == 0:
                label = 0
            else:
                label = -1
            points.append((y, x))   #Negate the coordinates
            labels.append(label)

        batch_points.append(points)
        batch_labels.append(labels)
    return np.array(batch_points), np.array(batch_labels)


def init_point_sampling(mask, get_point=1):
    """
    Initialization samples points from the mask and assigns labels to them.
    Args:
        mask (torch.Tensor): Input mask tensor.
        num_points (int): Number of points to sample. Default is 1.
    Returns:
        coords (torch.Tensor): Tensor containing the sampled points' coordinates (x, y).
        labels (torch.Tensor): Tensor containing the corresponding labels
            (0 for background, 1 for foreground).
    """
    if isinstance(mask, torch.Tensor):
        mask = mask.numpy()
        
     # Get coordinates of black/white pixels
    fg_coords = np.argwhere(mask == 1)[:, ::-1]
    bg_coords = np.argwhere(mask == 0)[:, ::-1]

    fg_size = len(fg_coords)
    bg_size = len(bg_coords)

    if get_point == 1:
        if fg_size > 0:
            index = np.random.randint(fg_size)
            fg_coord = fg_coords[index]
            label = 1
        else:
            index = np.random.randint(bg_size)
            fg_coord = bg_coords[index]
            label = 0
        return (torch.as_tensor([fg_coord.tolist()], dtype=torch.float),
                torch.as_tensor([label], dtype=torch.int))
    else:
        num_fg = get_point // 2
        num_bg = get_point - num_fg
        fg_indices = np.random.choice(fg_size, size=num_fg, replace=True)
        bg_indices = np.random.choice(bg_size, size=num_bg, replace=True)
        fg_coords = fg_coords[fg_indices]
        bg_coords = bg_coords[bg_indices]
        coords = np.concatenate([fg_coords, bg_coords], axis=0)
        labels = np.concatenate([np.ones(num_fg), np.zeros(num_bg)]).astype(int)
        indices = np.random.permutation(get_point)
        coords, labels = (torch.as_tensor(coords[indices], dtype=torch.float),
                          torch.as_tensor(labels[indices], dtype=torch.int))
        return coords, labels
    

def train_transforms(img_size, ori_h, ori_w):
    transforms = []
    if ori_h < img_size and ori_w < img_size:
        transforms.append(
            A.PadIfNeeded(
                min_height=img_size,
                min_width=img_size,
                border_mode=cv2.BORDER_CONSTANT,
                value=(0, 0, 0)
            )
        )
    else:
        transforms.append(
            A.Resize(int(img_size), int(img_size),
                     interpolation=cv2.INTER_NEAREST)
        )
    transforms.append(ToTensorV2(p=1.0))

    return A.Compose(transforms, p=1.)


def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
       "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])

    os.makedirs(os.path.dirname(filename), exist_ok=True)

    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger


def generate_point(masks, labels, low_res_masks, batched_input, point_num):
    masks_clone = masks.clone()
    masks_sigmoid = torch.sigmoid(masks_clone)
    masks_binary = (masks_sigmoid > 0.5).float()

    low_res_masks_clone = low_res_masks.clone()
    low_res_masks_logist = torch.sigmoid(low_res_masks_clone)

    points, point_labels = select_random_points(masks_binary, labels, point_num=point_num)
    batched_input["mask_inputs"] = low_res_masks_logist
    batched_input["point_coords"] = torch.as_tensor(points)
    batched_input["point_labels"] = torch.as_tensor(point_labels)
    batched_input["boxes"] = None
    return batched_input


def setting_prompt_none(batched_input):
    batched_input["point_coords"] = None
    batched_input["point_labels"] = None
    batched_input["boxes"] = None
    return batched_input


def draw_boxes(img, boxes):
    img_copy = np.copy(img)
    for box in boxes:
        cv2.rectangle(img_copy, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
    return img_copy


def save_masks(preds, save_path, mask_name, image_size, original_size, pad=None,
               boxes=None, points=None, visual_prompt=False):

    ori_h, ori_w = original_size

    preds = torch.sigmoid(preds)
    preds[preds > 0.5] = int(1)
    preds[preds <= 0.5] = int(0)

    mask = preds.squeeze().cpu().numpy()
    mask = cv2.cvtColor(mask * 255, cv2.COLOR_GRAY2BGR)

    if visual_prompt: #visualize the prompt
        if boxes is not None:
            boxes = boxes.squeeze().cpu().numpy()

            x0, y0, x1, y1 = boxes
            if pad is not None:
                x0_ori = int((x0 - pad[1]) + 0.5)
                y0_ori = int((y0 - pad[0]) + 0.5)
                x1_ori = int((x1 - pad[1]) + 0.5)
                y1_ori = int((y1 - pad[0]) + 0.5)
            else:
                x0_ori = int(x0 * ori_w / image_size) 
                y0_ori = int(y0 * ori_h / image_size) 
                x1_ori = int(x1 * ori_w / image_size) 
                y1_ori = int(y1 * ori_h / image_size)

            boxes = [(x0_ori, y0_ori, x1_ori, y1_ori)]
            mask = draw_boxes(mask, boxes)

        if points is not None:
            point_coords, point_labels = (points[0].squeeze(0).cpu().numpy(),
                                          points[1].squeeze(0).cpu().numpy())
            point_coords = point_coords.tolist()
            if pad is not None:
                ori_points = [
                    [int((x * ori_w / image_size)) , int((y * ori_h / image_size))]
                    if l == 0 else [x - pad[1], y - pad[0]]
                    for (x, y), l in zip(point_coords, point_labels)
                ]
            else:
                ori_points = [
                    [int((x * ori_w / image_size)) , int((y * ori_h / image_size))]
                    for x, y in point_coords
                ]

            for point, label in zip(ori_points, point_labels):
                x, y = map(int, point)
                color = (0, 255, 0) if label == 1 else (0, 0, 255)
                mask[y, x] = color
                cv2.drawMarker(mask, (x, y), color, markerType=cv2.MARKER_CROSS,
                               markerSize=7, thickness=2)

    os.makedirs(save_path, exist_ok=True)
    mask_path = os.path.join(save_path, f"{mask_name}")
    cv2.imwrite(mask_path, np.uint8(mask))


def calc_step(num_colors: int):
    return max(1, min(STEP, int((255 - 25) / num_colors)))


def semantic2instances(gray: np.ndarray) -> np.ndarray:
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    _, label = cv2.connectedComponents(binary)
    label = label.astype(np.uint16)
    return label


def get_transform(dst_size: int, ori_h: int, ori_w: int):
    if ori_h < dst_size and ori_w < dst_size:
        t = A.PadIfNeeded(min_height=dst_size, min_width=dst_size,
                          position="center", border_mode=cv2.BORDER_CONSTANT,
                          value=0)
    else:
        t = A.Resize(dst_size, dst_size, interpolation=cv2.INTER_NEAREST)
    return t


def postprocess_masks(low_res_masks, image_size, original_size):
    ori_h, ori_w = original_size
    masks = F.interpolate(low_res_masks, (image_size, image_size),
                          mode="bilinear", align_corners=False)

    if ori_h < image_size and ori_w < image_size:
        top = torch.div((image_size - ori_h), 2, rounding_mode='trunc')  #(image_size - ori_h) // 2
        left = torch.div((image_size - ori_w), 2, rounding_mode='trunc') #(image_size - ori_w) // 2
        masks = masks[..., top: ori_h + top, left: ori_w + left]
        pad = (top, left)
    else:
        masks = F.interpolate(masks, original_size,
                              mode="bilinear", align_corners=False)
        pad = None

    return masks, pad


def prompt_and_decoder(args, batched_input, model, image_embeddings,
                       decoder_iter=False):
    if batched_input["point_coords"] is not None:
        points = (batched_input["point_coords"], batched_input["point_labels"])
    else:
        points = None

    if decoder_iter:
        with torch.no_grad():
            sparse_embeddings, dense_embeddings = model.prompt_encoder(
                points=points,
                boxes=batched_input.get("boxes", None),
                masks=batched_input.get("mask_inputs", None),
                cluster_edges=batched_input.get("cluster_edge_coords", None)
            )

    else:
        sparse_embeddings, dense_embeddings = model.prompt_encoder(
            points=points,
            boxes=batched_input.get("boxes", None),
            masks=batched_input.get("mask_inputs", None),
            cluster_edges=batched_input.get("cluster_edge_coords", None)
        )

    # low_res_masks, iou_predictions = model.mask_decoder(
    #     image_embeddings=image_embeddings,
    #     image_pe=model.prompt_encoder.get_dense_pe(),
    #     sparse_prompt_embeddings=sparse_embeddings,
    #     dense_prompt_embeddings=dense_embeddings,
    #     multimask_output=args.multimask,)

    # print(f"image_embeddings shape: {image_embeddings.shape}")
    # print(f"sparse_embeddings shape: {sparse_embeddings.shape}")
    # print(f"dense_embeddings shape: {dense_embeddings.shape}")

    dense_embeddings_resized = F.interpolate(dense_embeddings, size=image_embeddings.shape[-2:], mode='bilinear', align_corners=False)
    #print(f"dense_embeddings_resized shape: {dense_embeddings_resized.shape}")

    
    low_res_seg_masks, seg_iou_predictions = model.segmentation_decoder(
        image_embeddings=image_embeddings,
        image_pe=model.prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse_embeddings,
        dense_prompt_embeddings=dense_embeddings,
        multimask_output=args.multimask,
    )

    low_res_norm_edge_masks, norm_edge_iou_predictions = model.normal_edge_decoder(
        image_embeddings=image_embeddings,
        image_pe=model.prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse_embeddings,
        dense_prompt_embeddings=dense_embeddings,
        multimask_output=args.multimask,
    )

    low_res_cluster_edge_masks, cluster_edge_iou_predictions = model.cluster_edge_decoder(
        image_embeddings=image_embeddings,
        image_pe=model.prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse_embeddings,
        dense_prompt_embeddings=dense_embeddings,
        multimask_output=args.multimask,
    )

    # if args.multimask:
    #     max_values, max_indexs = torch.max(iou_predictions, dim=1)
    #     max_values = max_values.unsqueeze(1)
    #     iou_predictions = max_values
    #     low_res = []
    #     for i, idx in enumerate(max_indexs):
    #         low_res.append(low_res_masks[i:i + 1, idx])
    #     low_res_masks = torch.stack(low_res, 0)

    if args.multimask:
        max_values, max_indexs = torch.max(seg_iou_predictions, dim=1)
        max_values = max_values.unsqueeze(1)
        seg_iou_predictions = max_values
        seg_low_res = []
        for i, idx in enumerate(max_indexs):
            seg_low_res.append(low_res_seg_masks[i:i + 1, idx])
        low_seg_masks = torch.stack(seg_low_res, 0)


    #masks = F.interpolate(low_res_masks, (args.image_size, args.image_size), mode="bilinear", align_corners=False)
    seg_masks = F.interpolate(low_res_seg_masks, (args.image_size, args.image_size), mode="bilinear", align_corners=False)
    norm_edge_masks = F.interpolate(low_res_norm_edge_masks, (args.image_size, args.image_size), mode="bilinear", align_corners=False)
    cluster_edge_masks = F.interpolate(low_res_cluster_edge_masks, (args.image_size, args.image_size), mode="bilinear", align_corners=False)

    # Convert cluster_edge_masks to tensor if it's a tuple before interpolating
    #cluster_edge_masks = F.interpolate(torch.stack(cluster_edge_masks) if isinstance(cluster_edge_masks, tuple) else cluster_edge_masks, (args.image_size, args.image_size), mode="bilinear", align_corners=False)

    # Check the shapes of cluster_edge_masks
    if isinstance(cluster_edge_masks, tuple):
        print("Shapes of tensors in cluster_edge_masks:")
        valid_masks = []
        for i, mask in enumerate(cluster_edge_masks):
            print(f"cluster_edge_masks[{i}] shape: {mask.shape}")
            if len(mask.shape) == 4:
                valid_masks.append(mask)
        
        if valid_masks:
            # Find the maximum shape
            max_shape = max([mask.shape for mask in valid_masks], key=lambda x: (x[2], x[3]))
            print(f"Max shape in cluster_edge_masks: {max_shape}")

            # Resize all valid masks to the maximum shape
            resized_cluster_edge_masks = [F.interpolate(mask, size=max_shape[2:], mode='bilinear', align_corners=False) for mask in valid_masks]
            cluster_edge_masks = torch.stack(resized_cluster_edge_masks)
        else:
            raise ValueError("No valid masks found in cluster_edge_masks.")

    
    #return masks, low_res_masks, iou_predictions
    return seg_masks, low_res_seg_masks, seg_iou_predictions, norm_edge_masks, low_res_norm_edge_masks, norm_edge_iou_predictions, cluster_edge_masks, low_res_cluster_edge_masks, cluster_edge_iou_predictions


class MaskPredictor:

    def __init__(self, model, pred_iou_thresh: float, stability_score_thresh: float,
                 points_per_side: int = 32, points_per_batch: int = 256, **kwargs):
        self.generator = SamAutomaticMaskGenerator(
            model=model,
            pred_iou_thresh=pred_iou_thresh,
            stability_score_thresh=stability_score_thresh,
            stability_score_offset=1.0,
            box_nms_thresh=0.7,
            min_mask_region_area=10,
            points_per_side=points_per_side,
            points_per_batch=points_per_batch
        )

    def predict(self, image: np.ndarray):
        """
        image: RGB image
        """
        self.generator.predictor.model.eval()
        masks = self.generator.generate(image)
        pred_mask = np.zeros(shape=(image.shape[0], image.shape[1]), dtype=np.uint16)
        for i, mask_data in enumerate(masks):
            mask = mask_data["segmentation"]
            pred_mask[mask] = i + 1

        return pred_mask

    def batch_predict(self, images: List[np.ndarray] | List[str]):
        """
        image: RGB images or image paths
        """
        if isinstance(images[0], str):
            images = [cv2.cvtColor(cv2.imread(_), cv2.COLOR_BGR2RGB) for _ in images]
        masks = [self.predict(_) for _ in images]
        return masks


def generate_cluster_edge_mask(mask, current_value, iterations):
    """
    Generate cluster edge masks from grayscale instance segmentation masks.
    Identifies edges between touching cells.

    Args:
    - label (numpy.ndarray): Grayscale instance segmentation mask.

    Returns:
    - numpy.ndarray: Binary cluster edge mask.
    """
    # Dilate the label to make adjacent instances overlap
    mask = mask.copy()
    kernel = np.ones((3, 3), np.uint8)
    dilated_label = cv2.dilate(mask, kernel, iterations=iterations)

    # Identify where the dilated label differs from the original, indicating potential cluster edges
    potential_cluster_edges = dilated_label != mask

    # Initialize cluster edge mask
    cluster_edge_mask = np.zeros(mask.shape, dtype=np.uint8)

    # Iterate through potential cluster edges and validate against original label
    for y in range(mask.shape[0]):
        for x in range(mask.shape[1]):
            if potential_cluster_edges[y, x]:
                # Get the original and dilated values
                original_value = mask[y, x]
                dilated_value = dilated_label[y, x]

                # If the dilated value is different from the original, and neither is background
                if original_value != dilated_value and original_value != 0 and dilated_value != 0 and dilated_value == current_value:
                    # Mark as cluster edge
                    cluster_edge_mask[y, x] = 1
    return cluster_edge_mask


if __name__ == "__main__":
    # test
    from segment_anything.build_sam import _build_sam
    model = _build_sam(
        encoder_embed_dim=768,
        encoder_depth=12,
        encoder_num_heads=12,
        encoder_global_attn_indexes=[2, 5, 8, 11],
        image_size=256,
        checkpoint="epoch0077_test-loss0.1181_sam.pth",
        encoder_adapter=True
    ).to("cpu")
    mask_predictor = MaskPredictor(
        model=model, pred_iou_thresh=0.8, stability_score_thresh=0.9
    )
    image = cv2.imread("CoNIC_image_0001.png")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mask = mask_predictor.predict(image)
    print("mask values: ", np.unique(mask))
    mask = mask / mask.max() * 255
    mask = mask.astype(np.uint8)
    cv2.imshow("mask", mask)
    cv2.waitKey(0)




