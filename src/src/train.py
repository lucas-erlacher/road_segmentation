# This is the training file that I have used to produce the best model so far.

# IDEAS
# - incorporate the rotating into the training and not just the inference
# - can images in rotate_model be improved with postprocessing? (e.g. thresholding)

from torch.utils.tensorboard.writer import SummaryWriter
import torchvision
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import sys
from networks.modded_unet import ModNet
from dataloader.segdataset import SegmentationDataset, SegmentationDatasetBuilder
import torch.utils.data
from networks.skip_connection import UnetSkipConnectionBlock
from utils import seed_everything
import torchmetrics
import torchvision.transforms.v2 as transforms
import argparse

directory = os.path.dirname(os.path.realpath(__file__))
sys.path.append(directory)
torchvision.disable_beta_transforms_warning()

seed_everything(42)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--split_factor", type=int, default=8)
    parser.add_argument("--n_eval_steps", type=int, default=100)
    parser.add_argument("--n_save_steps", type=int, default=300)
    parser.add_argument("--n_save_example_img_steps", type=int, default=50)
    parser.add_argument("--checkpoint", type=str, default="")
    return parser.parse_args()


class DeepLab(torch.nn.Module):
    def __init__(self):
        super(DeepLab, self).__init__()
        self.model = torch.hub.load(
            "pytorch/vision:v0.10.0", "deeplabv3_resnet50", pretrained=False
        )
        self.model.classifier[4] = torch.nn.Conv2d(
            256, 1, kernel_size=(1, 1), stride=(1, 1)
        )

    def forward(self, x):
        return self.model(x)["out"]


if __name__ == "__main__":
    args = get_args()
    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device: ", args.device)
    device = torch.device(args.device)

    # Dataset
    transform = transforms.Compose(
        [
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.RandomAffine(
                degrees=15,
                translate=(0.05, 0.05),
                scale=(0.95, 1.05),
            ),
            transforms.ColorJitter(
                brightness=0.15, contrast=0.15, saturation=0.15, hue=0.05
            ),
            transforms.RandomVerticalFlip(),
            transforms.RandomHorizontalFlip(),
        ]
    )
    seg_builder = SegmentationDatasetBuilder(transform=transform)
    seg_builder.add_path(
        "cil-road-segmentation-2022/training/images",
        "cil-road-segmentation-2022/training/groundtruth",
        "Official Data",
    )
    # seg_builder.add_path(
    #     "dataset_40K/source_sample",
    #     "dataset_40K/target_sample",
    #     "Artifical Data",
    # )
    seg_builder.set_test_dataset("Official Data", ratio=0.3)
    train_set, test_set = seg_builder.get_datasets()

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
    )
    test_loader = DataLoader(
        test_set, batch_size=args.batch_size, shuffle=False, num_workers=2
    )

    # Model
    model = DeepLab()
    if os.path.isfile(args.checkpoint):
        model.load_state_dict(torch.load(args.checkpoint))

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", patience=5)

    # Loss
    loss_function = torch.nn.BCEWithLogitsLoss()

    # Logging for tensorboard
    writer = SummaryWriter()
    os.makedirs(os.path.join(writer.get_logdir(), "checkpoint"), exist_ok=True)
    print("Logging to: ", writer.get_logdir())

    tensorboard_gt_results = []
    tensorboard_eval_imgs = []
    for i in range(9):
        data = test_set[i]
        tensorboard_eval_imgs.append(data[0])
        tensorboard_gt_results.append(data[1])
    tensorboard_eval_imgs = torch.stack(tensorboard_eval_imgs, dim=0)
    tensorboard_gt_results = torch.stack(tensorboard_gt_results, dim=0).unsqueeze(1)
    writer.add_image(
        "Groundtruth", torchvision.utils.make_grid(tensorboard_gt_results, nrow=3), 0
    )
    writer.add_image(
        "Input", torchvision.utils.make_grid(tensorboard_eval_imgs, nrow=3), 0
    )

    # train loop
    model.to(device)
    model.train()

    step = 0

    for e in range(args.epochs):
        accum_loss = 0
        for X_data, y_data in tqdm(
            train_loader, total=len(train_loader), position=1, leave=False
        ):
            optimizer.zero_grad()

            X_data = X_data.to(device)
            y_data = y_data.to(device).unsqueeze(1).type(torch.float32)

            # FORWARD PASS
            pred = model(X_data)

            # LOSS
            loss = loss_function(pred, y_data)
            accum_loss += loss.item()

            # BACKWARD PASS
            loss.backward()
            optimizer.step()

            writer.add_scalar("loss", loss.item(), step)
            writer.add_scalar("lr", optimizer.param_groups[0]["lr"], step)

            # Display example images
            if step % args.n_save_example_img_steps == 0:
                model.eval()

                with torch.no_grad():
                    # Evaluation examples
                    pred = model(tensorboard_eval_imgs.to(device))
                    writer.add_image(
                        "Output", torchvision.utils.make_grid(pred, nrow=3), step
                    )
                model.train()

            # Save model
            if step % args.n_save_steps == 0:
                torch.save(
                    model.state_dict(),
                    writer.get_logdir() + "/checkpoint/model_" + str(step) + ".pt",
                )

            # Evaluation
            if step % args.n_eval_steps == 0:
                model.eval()
                acc = torchmetrics.Accuracy(task="binary").to(device)
                f1 = torchmetrics.F1Score(task="binary").to(device)
                precision = torchmetrics.Precision(task="binary").to(device)
                recall = torchmetrics.Recall(task="binary").to(device)
                with torch.no_grad():
                    for img in tqdm(
                        test_loader, total=len(test_loader), position=2, leave=False
                    ):
                        pred = model(img[0].to(device))
                        pred = pred > 0.0
                        gt = img[1].to(device).unsqueeze(1)
                        acc(gt, pred)
                        f1(gt, pred)
                        precision(gt, pred)
                        recall(gt, pred)

                    writer.add_scalar("Evaluation accuracy", acc.compute().item(), step)
                    writer.add_scalar("Evaluation f1", f1.compute().item(), step)
                    writer.add_scalar(
                        "Evaluation precision", precision.compute().item(), step
                    )
                    writer.add_scalar(
                        "Evaluation recall", recall.compute().item(), step
                    )

                model.train()
                print(
                    "Evaluation Step: ",
                    step,
                    "Accuracy: ",
                    acc.compute().item(),
                    "F1: ",
                    f1.compute().item(),
                    "Precision: ",
                    precision.compute().item(),
                    "Recall: ",
                    recall.compute().item(),
                )
            step += 1
        print(
            "Train Epoch: ",
            e,
            "Step: ",
            step,
            "Loss: ",
            accum_loss / len(train_loader),
            "LR: ",
            optimizer.param_groups[0]["lr"],
        )
        accum_loss = accum_loss / len(train_loader)
        scheduler.step(accum_loss)
    writer.close()

    # save model
    torch.save(model.state_dict(), writer.get_logdir() + "/model_final.pt")
