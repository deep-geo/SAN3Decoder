import os
import sys
import re
import random
import datetime
import glob
import numpy as np
import torch
import wandb
import cv2

#from segment_anything.build_sam import sam_model_registry
from check_sam_registry import sam_model_registry
from torch import optim
from torch.utils.data import DataLoader

from DataLoader2 import TrainingDataset, TestingDataset, TrainingDatasetFolder, TestingDatasetFolder, stack_dict_batched #, CombineBatchSampler 
from utils import get_logger, generate_point, setting_prompt_none, save_masks, postprocess_masks, to_device, prompt_and_decoder
from loss import FocalDiceloss_IoULoss, Total_Loss
from arguments import parse_train_args
from metrics import SegMetrics, AggregatedMetrics
from tqdm import tqdm
import torch.multiprocessing as mp

torch.set_default_dtype(torch.float32)
max_num_chkpt = 2
global_step = 0
global_metrics_dict = {}
global_train_losses = []

@torch.no_grad()
def eval_model(args, model, test_loader, output_dataset_metrics: bool = False):
    global global_step

    model.eval()
    criterion = FocalDiceloss_IoULoss()
    
    dataset_names = []
    test_loss = []
    prompt_dict = {}
    all_eval_metrics = []

    for i, batched_input in enumerate(tqdm(test_loader,
                                           desc=f"Testing(step={global_step})",
                                           mininterval=0.5, ascii=True)):

        batched_input = to_device(batched_input, args.device)
        dataset_names.append(batched_input["dataset_name"])
        ori_labels = batched_input["ori_label"]
        batch_original_size = batched_input["original_size"]
        original_size = batch_original_size[0][0], batch_original_size[1][0]
        labels = batched_input["label"]
        img_name = batched_input['name'][0]
        if args.prompt_path is None:
            prompt_dict[img_name] = {
                "boxes": batched_input["boxes"].squeeze(1).cpu().numpy().tolist(),
                "point_coords": batched_input["point_coords"].squeeze(1).cpu().numpy().tolist(),
                "point_labels": batched_input["point_labels"].squeeze(1).cpu().numpy().tolist(),
                "cluster_edge_coords": batched_input["cluster_edge_coords"].squeeze(1).cpu().numpy().tolist() if "cluster_edge_coords" in batched_input else None
            }

        with torch.no_grad():
            image_embeddings = model.image_encoder(batched_input["image"])

        if args.boxes_prompt:
            save_path = os.path.join(args.work_dir, args.run_name, "boxes_prompt")
            batched_input["point_coords"], batched_input["point_labels"] = None, None
            # masks, low_res_masks, iou_predictions = prompt_and_decoder(
            #     args, batched_input, model, image_embeddings)
            seg_masks, low_res_seg_masks, seg_iou_predictions, norm_edge_masks, low_res_norm_edge_masks, norm_edge_iou_predictions, cluster_edge_masks, low_res_cluster_edge_masks, cluster_edge_iou_predictions = prompt_and_decoder(args, batched_input, model, image_embeddings)
            points_show = None

        else:
            save_path = os.path.join(
                f"{args.work_dir}", args.run_name,
                f"iter{args.iter_point if args.iter_point > 1 else args.point_num}_prompt")
            batched_input["boxes"] = None
            point_coords, point_labels = [batched_input["point_coords"]], [
                batched_input["point_labels"]]

            for p in range(args.iter_point):
                seg_masks, low_res_seg_masks, seg_iou_predictions, norm_edge_masks, low_res_norm_edge_masks, norm_edge_iou_predictions, cluster_edge_masks, low_res_cluster_edge_masks, cluster_edge_iou_predictions = prompt_and_decoder(args, batched_input, model, image_embeddings) #Added by Ray 0716
                if p != args.iter_point - 1:
                    batched_input = generate_point(
                        masks, labels, low_res_masks, batched_input,
                        args.point_num)
                    batched_input = to_device(batched_input, args.device)
                    point_coords.append(batched_input["point_coords"])
                    point_labels.append(batched_input["point_labels"])
                    batched_input["point_coords"] = torch.concat(point_coords,dim=1)
                    batched_input["point_labels"] = torch.concat(point_labels,dim=1)

            points_show = (torch.concat(point_coords, dim=1),
                           torch.concat(point_labels, dim=1))

        #masks, pad = postprocess_masks(low_res_masks, args.image_size, original_size)
        if args.save_pred:
            save_masks(seg_masks, save_path, img_name, args.image_size,
                       original_size, pad, batched_input.get("boxes", None),
                       points_show)

        loss = criterion(seg_masks, ori_labels, seg_iou_predictions)
        test_loss.append(loss.item())

        seg_metrics = SegMetrics(args.metrics, seg_masks, labels)
        all_eval_metrics.append(seg_metrics.result())

    average_loss = np.mean(test_loss)

    agg_metrics = AggregatedMetrics(args.metrics, all_eval_metrics, dataset_names)
    metrics_overall = agg_metrics.aggregate()
    if output_dataset_metrics:
        metrics_datasets = agg_metrics.aggregate_by_datasets()
    else:
        metrics_datasets = None

    return average_loss, metrics_overall, metrics_datasets


def train_one_epoch(args, model, optimizer, train_loader, epoch, criterion, total_loss_fn, 
                    test_loader, run_dir):

    global global_metrics_dict
    global global_step
    global global_train_losses

    pbar = tqdm(total=len(train_loader), desc="Training", mininterval=0.5)
    dataloader_iter = iter(train_loader)

    while True:

        model.train()

        try:
            batched_input = next(dataloader_iter)
        except StopIteration:
            break

        global_step += 1
        batched_input = stack_dict_batched(batched_input)
        batched_input = to_device(batched_input, args.device)

        if random.random() > 0.5:
            batched_input["point_coords"] = None
            flag = "boxes"
        else:
            batched_input["boxes"] = None
            flag = "point"

        for n, value in model.image_encoder.named_parameters():
            if "Adapter" in n:
                value.requires_grad = True
            else:
                value.requires_grad = False

        labels = batched_input["label"]
        image_embeddings = model.image_encoder(batched_input["image"])

        batch, _, _, _ = image_embeddings.shape
        image_embeddings_repeat = []
        for i in range(batch):
            image_embed = image_embeddings[i]
            image_embed = image_embed.repeat(args.mask_num, 1, 1, 1)
            image_embeddings_repeat.append(image_embed)

        image_embeddings = torch.cat(image_embeddings_repeat, dim=0)

        #masks, low_res_masks, iou_predictions = prompt_and_decoder(args, batched_input, model, image_embeddings, decoder_iter=False
        seg_masks, low_res_seg_masks, seg_iou_predictions, norm_edge_masks, low_res_norm_edge_masks, norm_edge_iou_predictions, cluster_edge_masks, low_res_cluster_edge_masks, cluster_edge_iou_predictions = prompt_and_decoder(args, batched_input, model, image_embeddings, decoder_iter=False)

        #loss = criterion(masks, labels, iou_predictions)
        loss = total_loss_fn(seg_masks, labels, seg_iou_predictions, 
                             norm_edge_masks, labels, norm_edge_iou_predictions, 
                             cluster_edge_masks, labels, cluster_edge_iou_predictions)
        
        loss.backward(retain_graph=False)

        optimizer.step()
        optimizer.zero_grad()

        point_num = random.choice(args.point_list)
        batched_input = generate_point(seg_masks, labels, low_res_seg_masks, batched_input, point_num)
        batched_input = to_device(batched_input, args.device)

        image_embeddings = image_embeddings.detach().clone()
        for n, value in model.named_parameters():
            if "image_encoder" in n:
                value.requires_grad = False
            else:
                value.requires_grad = True

        init_mask_num = np.random.randint(1, args.iter_point - 1)
        for p in range(args.iter_point):
            if p == init_mask_num or p == args.iter_point - 1:
                batched_input = setting_prompt_none(batched_input)

            #masks, low_res_masks, iou_predictions = prompt_and_decoder(
            #    args, batched_input, model, image_embeddings, decoder_iter=True)

            seg_masks, low_res_seg_masks, seg_iou_predictions, norm_edge_masks, low_res_norm_edge_masks, norm_edge_iou_predictions, cluster_edge_masks, low_res_cluster_edge_masks, cluster_edge_iou_predictions = prompt_and_decoder(args, batched_input, model, image_embeddings, decoder_iter=True)

            optimizer.zero_grad()
            loss = total_loss_fn(seg_masks, labels, seg_iou_predictions,
                             norm_edge_masks, labels, norm_edge_iou_predictions, 
                             cluster_edge_masks, labels, cluster_edge_iou_predictions)

            loss.backward(retain_graph=False)
            optimizer.step()

            if p != args.iter_point - 1:
                point_num = random.choice(args.point_list)
                #batched_input = generate_point(masks, labels, low_res_masks, batched_input, point_num)
                batched_input = generate_point(seg_masks, labels, low_res_seg_masks, batched_input, point_num)
                batched_input = to_device(batched_input, args.device)

        if int(batch + 1) % 200 == 0:
            print(
                f"epoch:{epoch + 1}, iteration:{batch + 1}, loss:{loss.item()}")
            save_path = os.path.join(f"{args.work_dir}/models", args.run_name,
                                     f"epoch{epoch + 1}_batch{batch + 1}.pth")
            state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict()}
            torch.save(state, save_path)

        global_train_losses.append(loss.item())

        pbar.update()
        pbar.set_postfix(train_loss=loss.item(), epoch=epoch)

        if global_step % args.eval_interval == 0:
            average_test_loss, test_metrics_overall, test_metrics_datasets = \
                eval_model(args, model, test_loader, output_dataset_metrics=True)

            for metric in args.metrics:
                global_metrics_dict[f"Overall/{metric}"] = test_metrics_overall.get(metric, None)
                if test_metrics_datasets:
                    for dataset_name in test_metrics_datasets.keys():
                        global_metrics_dict[f"{dataset_name}/{metric}"] = \
                            test_metrics_datasets[dataset_name].get(metric, None)

            average_train_loss = np.mean(global_train_losses)
            global_metrics_dict["Loss/train"] = average_train_loss
            global_metrics_dict["Loss/test"] = average_test_loss

            wandb.log(global_metrics_dict, step=global_step, commit=True)

            #global_metrics_dict = {}
            #global_train_losses = []

            chkpts = [
                (p, float(re.search(r"loss(\d+\.\d+|nan)", os.path.basename(p)).group(1)))
                for p in glob.glob(os.path.join(run_dir, "*.pth"))
            ]
            chkpts = sorted(chkpts, key=lambda x: x[-1])
            if not chkpts or average_test_loss < chkpts[-1][-1]:
                save_path = os.path.join(
                    run_dir,
                    f"epoch{epoch + 1:04d}_step{global_step}_loss{average_test_loss:.4f}.pth"
                )
                state = {
                    'model': model.float().state_dict(),
                    'optimizer': optimizer,
                    'train-loss': average_train_loss,
                    'test-loss': average_test_loss,
                    'epoch': epoch + 1,
                    'step': global_step
                }
                torch.save(state, save_path)
                print("\nsave new checkpoint: ", save_path)

                chkpts.append((save_path, average_test_loss))
                chkpts = sorted(chkpts, key=lambda x: x[-1])
                for chkpt, _ in chkpts[max_num_chkpt:]:
                    print("remove checkpoint: ", chkpt)
                    os.remove(chkpt)
            model.train()
            print("train status after eval: ", model.training)


def main(args):

    global global_metrics_dict
    global global_step

    model = sam_model_registry[args.model_type](args).to(args.device)
    print(f"args.model_type:{args.model_type}")
    for key in sam_model_registry.keys():
        print(f"Sam_model_registry:{key}")
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = FocalDiceloss_IoULoss()
    total_loss_fn = Total_Loss(weight=20.0, iou_scale=1.0)

    #print("Model's segmentation_decoder:", model.segmentation_decoder)
    
    args.run_name = f"{args.run_name}_{datetime.datetime.now().strftime('%m-%d_%H-%M')}"

    run_dir = os.path.join(args.work_dir, "models", args.run_name)
    os.makedirs(run_dir, exist_ok=True)

    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[5, 10], gamma=0.5)

    resume_chkpt = None
    if args.resume:
        if os.path.isfile(args.resume):
            resume_chkpt = args.resume
        else:   # dir
            chkpts = sorted(
                glob.glob(os.path.join(run_dir, "*.pth")),
                key=lambda p: float(
                    re.search(r"epoch(\d+)", os.path.basename(p)).group(1))
            )
            if chkpts:
                resume_chkpt = chkpts[-1]

    print("\nresume_chkpt: ", resume_chkpt)

    resume_epoch = 0
    if resume_chkpt:
        with open(resume_chkpt, "rb") as f:
            checkpoint = torch.load(f, map_location=args.device)
            #optimizer.load_state_dict(checkpoint['optimizer'].state_dict())
            resume_epoch = checkpoint["epoch"]

    params = ["seed", "epochs", "batch_size", "image_size", "mask_num", "lr",
              "resume", "model_type", "checkpoint", "boxes_prompt",
              "point_num", "iter_point", "lr_scheduler", "point_list",
              "multimask", "encoder_adapter"]
    config = {p: getattr(args, p) for p in params}
    config["resume_checkpoint"] = resume_chkpt
    wandb.init(project="SAM_3decoder", name=args.run_name, config=config)

    # Random seed Setting
    if args.seed is not None:
        print(f"Random Seed: {args.seed}")
        np.random.seed(args.seed)
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    if args.data_root:
        train_set_gt = TrainingDatasetFolder(
            data_root=args.data_root,
            train_size=1-args.test_size,
            point_num=1,
            edge_point_num=args.edge_point_num,
            mask_num=args.mask_num,
            requires_name=False,
            random_seed=args.seed
        )
        test_set = TestingDatasetFolder(
            data_root=args.data_root,
            test_size=args.test_size,
            requires_name=True,
            point_num=args.point_num,
            edge_point_num=args.edge_point_num,
            return_ori_mask=True,
            prompt_path=args.prompt_path,
            sample_rate=args.test_sample_rate
        )
    else:
        raise ValueError(f"No dataset provided!")

    print("\nGT dataset length: ", len(train_set_gt))

    train_loader = DataLoader(train_set_gt,  batch_size=args.batch_size, num_workers=args.num_workers)
    test_loader = DataLoader(dataset=test_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    loggers = get_logger(
        os.path.join(
            args.work_dir, "logs",
            f"{args.run_name}_{datetime.datetime.now().strftime('%Y%m%d-%H%M.log')}"
        )
    )

    # print("\nEvaluate model using initial checkpoint...")
    # average_test_loss, test_metrics_overall, test_metrics_datasets = \
    #     eval_model(args, model, test_loader, output_dataset_metrics=True)
    #
    # for metric in args.metrics:
    #     global_metrics_dict[f"Overall/{metric}"] = test_metrics_overall.get(metric, None)
    #     if test_metrics_datasets:
    #         for dataset_name in test_metrics_datasets.keys():
    #             global_metrics_dict[f"{dataset_name}/{metric}"] = \
    #                 test_metrics_datasets[dataset_name].get(metric, None)
    #
    # global_metrics_dict["Loss/train"] = average_test_loss
    # global_metrics_dict["Loss/test"] = average_test_loss

    wandb.log(global_metrics_dict, step=global_step, commit=True)
    global_metrics_dict = {}
    
    for epoch in range(resume_epoch, args.epochs):

        model.train()

        train_one_epoch(
            args, model, optimizer, train_loader, epoch, criterion, total_loss_fn, 
            test_loader, run_dir
        )

        if args.lr_scheduler is not None:
            scheduler.step()

        for param_group in optimizer.param_groups:
            print(f"Learning Rate: {param_group['lr']}")
            sys.stdout.flush()  # Ensure the output is displayed

        for name, param in model.named_parameters():
            if param.grad is not None:
                print(f"{name} grad norm: {param.grad.norm()}")
                sys.stdout.flush()  # Ensure the output is displayed
            else:
                print("grad norm is None")

        # Optionally, perform gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # Print the average loss for the epoch
        average_loss = np.mean(global_train_losses)
        print(f"Epoch [{epoch+1}/{args.epochs}], Average Loss: {average_loss}")
        sys.stdout.flush()  # Ensure the output is displayed



if __name__ == '__main__':
    mp.set_start_method('spawn')
    args = parse_train_args()
    args.encoder_adapter = True
    args.multimask = False

    # args.batch_size = 2
    # args.test_sample_rate = 0.01
    # args.test_size = 0.1
    # args.num_workers = 2
    # args.image_size = 256
    # args.data_root = "/Users/zhaojq/Datasets/ALL_Multi"
    # args.checkpoint = "/Users/zhaojq/PycharmProjects/SAN3Decoder/sam_vit_b_01ec64.pth"
    # args.resume = "/Users/zhaojq/PycharmProjects/SAN3Decoder/epoch0079_test-loss0.1429_sam.pth"
    # args.model_type = "three_decoder"
    # args.edge_point_num = 3
    # args.boxes_prompt = True

    main(args)
