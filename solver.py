import glob
import os

import numpy as np
import torch
import losses as additional_losses
from torch.optim import lr_scheduler

import utils.common_utils as common_utils
from utils.log_utils import LogWriter
import gc
import wandb

"""wandb.init(project="quicknat-original")
wandb.config = {
  "learning_rate": 1e-5,
  "epochs": 20,
  "batch_size": 2
}"""
CHECKPOINT_EXTENSION = "pth.tar"
CHECKPOINT_DIR = "checkpoints"



class Solver(object):
    def __init__(
        self,
        model,
        exp_name,
        device,
        num_class,
        optim=torch.optim.Adam,
        optim_args={},
        # add weights
        # loss_func=additional_losses.CombinedLoss(),
        model_name="quicknat",
        labels=None,
        num_epochs=10,
        log_nth=5,
        lr_scheduler_step_size=5,
        lr_scheduler_gamma=0.5,
        use_last_checkpoint=True,
        exp_dir="experiments",
        log_dir="logs",
    ):

        self.device = device
        self.model = model

        self.model_name = model_name
        self.labels = labels
        self.num_epochs = num_epochs

        self.optim = optim(model.parameters(), **optim_args)
        self.scheduler = lr_scheduler.StepLR(
            self.optim, step_size=lr_scheduler_step_size, gamma=lr_scheduler_gamma
        )

        exp_dir_path = os.path.join(exp_dir, exp_name)
        print(exp_dir_path)
        common_utils.create_if_not(exp_dir_path)

        #Changed
        CHECKPOINT_DIR = exp_dir_path +"/checkpoints"
        print(CHECKPOINT_DIR)
        common_utils.create_if_not(CHECKPOINT_DIR)

        self.exp_dir_path = exp_dir_path

        self.log_nth = log_nth
        self.logWriter = LogWriter(
            num_class, log_dir, exp_name, use_last_checkpoint, labels
        )

        self.use_last_checkpoint = use_last_checkpoint

        self.start_epoch = 1
        self.start_iteration = 1

        self.best_ds_mean = 0
        self.best_ds_mean_epoch = 0

        if use_last_checkpoint:
            self.load_checkpoint()

    def train(self, train_loader, val_loader):
        """
        Train a given model with the provided data.

        Inputs:
        - train_loader: train data in torch.utils.data.DataLoader
        - val_loader: val data in torch.utils.data.DataLoader
        """

        os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

        # Comented for now
        # model, optim, scheduler = self.model, self.optim, self.scheduler
        model, optim, scheduler = self.model, self.optim, self.scheduler

        dataloaders = {"train": train_loader, "val": val_loader}

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            model.cuda(self.device)

        print("START TRAINING. : model name = ", self.model_name)
        # print(
        #     "START TRAINING. : model name = %s, device = %s"
        #     % (self.model_name, torch.cuda.get_device_name(self.device))
        # )
        current_iteration = self.start_iteration
        # -------------------------------
        val_iteration = 0
        val_dice_loops = int(552 / 4)
        # -------------------------------
        for epoch in range(self.start_epoch, self.num_epochs + 1):

            print(
                "==== Epoch ["
                + str(epoch)
                + " / "
                + str(self.num_epochs)
                + "] DONE ===="
            )
            checkpoint_name = self.exp_dir_path + "/" + CHECKPOINT_DIR + "/checkpoint_epoch_" + str(epoch) + "." + CHECKPOINT_EXTENSION
            self.save_checkpoint(
                {
                    "epoch": epoch + 1,
                    "start_iteration": current_iteration + 1,
                    "arch": self.model_name,
                    "state_dict": model.state_dict(),
                    "optimizer": optim.state_dict(),
                    "scheduler": scheduler.state_dict(),
                },
                checkpoint_name
            )


            print("\n==== Epoch [ %d  /  %d ] START ====" % (epoch, self.num_epochs))
            for phase in ["train", "val"]:
                print("<<<= Phase: %s =>>>" % phase)
                loss_arr = []
                out_list = []
                y_list = []
                if phase == "train":
                    model.train()
                    scheduler.step()

                else:
                    model.eval()    
                # Chunk input data to 3000 images
                """i = 0
                while i < 40: 
                    ind = np.random.choice(len(dataloaders['train']), min(len(dataloaders['train']),2), replace=False)
                    np.save('/Users/carlottaholzle/Desktop/monai_networks_local/Quicknat_codes/data/train/' + str(i), ind)
                    ind = np.random.choice(len(dataloaders['val']), min(len(dataloaders['val']),2), replace=False)
                    np.save('/Users/carlottaholzle/Desktop/monai_networks_local/Quicknat_codes/data/val/' + str(i), ind)
                    i = i + 1"""

                ind = np.random.choice(len(dataloaders[phase]), min(len(dataloaders[phase]),2), replace=False)
                ind = np.sort(ind, axis=0)
                print('Selected {0}/{1} images from Dataloader..'
                        .format(len(ind), dataloaders[phase].dataset.data_files['data'].shape[0]))
               
                for i_batch, sample_batched in enumerate(dataloaders[phase]):
                    """
                    if phase == 'train':
                    """
                    if i_batch not in ind:
                        continue
                    
                    
                    X = sample_batched[0].type(torch.FloatTensor)
                    y = sample_batched[1].type(torch.LongTensor)
                    w = sample_batched[2].type(torch.FloatTensor)

                    if model.is_cuda:
                        X, y, w = (
                            X.cuda(self.device, non_blocking=True),
                            y.cuda(self.device, non_blocking=True),
                            w.cuda(self.device, non_blocking=True),
                        )
                    print(X.shape)
                    print(type(X))

                    output = model(X)

                    loss_func = additional_losses.CombinedLoss()
                    # loss_func = additional_losses.DiceLoss()

                    if torch.cuda.is_available():
                        loss_func = loss_func.cuda(self.device)
                    else:
                        loss_func = loss_func

                    # loss = loss_func.forward(output, y)
                    loss = loss_func(output, y, w)
                    #wandb.log({"loss": loss})
                    if phase == 'train':
                        optim.zero_grad()
                        loss.backward()
                        optim.step()
                        if i_batch % self.log_nth == 0:
                            self.logWriter.loss_per_iter(loss.item(), i_batch, current_iteration)
                        current_iteration += 1

                    loss_arr.append(loss.item())
                    """if phase == "train":
                        optim.zero_grad()
                        loss.backward()
                        optim.step()
                        if i_batch % self.log_nth == 0:
                            self.logWriter.loss_per_iter(
                                loss.item(), i_batch, current_iteration
                            )

                        # if current_iteration == 4:
                        #    break
                        current_iteration += 1
                        # Move this inside train phase
                        loss_arr.append(loss.item())
                    else:
                        val_iteration += 1
                        loss_arr.append(loss.item())"""
                    # Calculate dice score per patient
                    # else:
                    #    val_iteration += 1
                    #    if val_iteration == val_dice_loops:
                    #        print('Calculating Dice Score for patient..')
                    #        loss_arr.append(loss.item())
                    #        val_iteration = 0

                    _, batch_output = torch.max(output, dim=1)
                    out_list.append(batch_output.cpu())
                    y_list.append(y.cpu())

                    del X, y, w, output, batch_output, loss
                    torch.cuda.empty_cache()
                    gc.collect()

                    if phase == "val":
                        if i_batch != len(dataloaders[phase]) - 1:
                            print("#", end="", flush=True)
                        else:
                            print("100%", flush=True)

                with torch.no_grad():
                    out_arr, y_arr = torch.cat(out_list), torch.cat(y_list)
                    self.logWriter.loss_per_epoch(loss_arr, phase, epoch)

                    # Specify specific images to show in Tensorboard
                    if phase == "train":
                        index = np.random.choice(
                            len(dataloaders[phase].dataset), 3, replace=False
                        )
                        # index = np.random.choice(ind, 3, replace=False)
                        index = np.sort(index, axis=0)
                    else:
                        index = [87, 358, 416]

                    print(self.device)
                    self.logWriter.image_per_epoch(
                        model.predict(
                            dataloaders[phase].dataset.data_files["data"][list(index)][
                                :, np.newaxis, :, :
                            ],
                            self.device,
                        ),
                        dataloaders[phase].dataset.labels["label"][list(index)],
                        dataloaders[phase].dataset.data_files["data"][list(index)],
                        phase,
                        epoch,
                    )
                    self.logWriter.cm_per_epoch(phase, out_arr, y_arr, epoch)
                    ds_mean = self.logWriter.dice_score_per_epoch(
                        phase, out_arr, y_arr, epoch
                    )
                    if phase == "val":
                        if ds_mean > self.best_ds_mean:
                            self.best_ds_mean = ds_mean
                            self.best_ds_mean_epoch = epoch
                        print("Best mean dice score is: ", self.best_ds_mean)
                    print("Current mean dice score is: ", ds_mean)

        print("FINISH.")
        self.logWriter.close()

    def save_best_model(self, path):
        """
        Save model with its parameters to the given path. Conventionally the
        path should end with "*.model".
        Inputs:
        - path: path string
        """
        print("Saving model... %s" % path)
        self.load_checkpoint(self.best_ds_mean_epoch)

        torch.save(self.model, path)

    def save_checkpoint(self, state, filename):
        torch.save(state, filename)

    def load_checkpoint(self, epoch=None):
        if epoch is not None:
            checkpoint_path = os.path.join(
                self.exp_dir_path,
                CHECKPOINT_DIR,
                "checkpoint_epoch_" + str(epoch) + "." + CHECKPOINT_EXTENSION,
            )
            self._load_checkpoint_file(checkpoint_path)
        else:
            all_files_path = os.path.join(
                self.exp_dir_path, CHECKPOINT_DIR, "*." + CHECKPOINT_EXTENSION
            )
            list_of_files = glob.glob(all_files_path)
            if len(list_of_files) > 0:
                checkpoint_path = max(list_of_files, key=os.path.getctime)
                self._load_checkpoint_file(checkpoint_path)
            else:
                self.logWriter.log(
                    "=> no checkpoint found at '{}' folder".format(
                        os.path.join(self.exp_dir_path, CHECKPOINT_DIR)
                    )
                )

    def _load_checkpoint_file(self, file_path):
        self.logWriter.log("=> loading checkpoint '{}'".format(file_path))
        checkpoint = torch.load(file_path)
        self.start_epoch = checkpoint["epoch"]
        self.start_iteration = checkpoint["start_iteration"]
        self.model.load_state_dict(checkpoint["state_dict"])
        self.optim.load_state_dict(checkpoint["optimizer"])

        for state in self.optim.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.to(self.device)

        self.scheduler.load_state_dict(checkpoint["scheduler"])
        self.logWriter.log(
            "=> loaded checkpoint '{}' (epoch {})".format(
                file_path, checkpoint["epoch"]
            )
        )
    def predict(self, X, device=0, enable_dropout=False):
        """
        Predicts the output after the model is trained.
        Inputs:
        - X: Volume to be predicted
        """
        self.eval()
        print("tensor size before transformation", X.shape)

        if type(X) is np.ndarray:
            # X = torch.tensor(X, requires_grad=False).type(torch.FloatTensor)
            X = (
                torch.tensor(X, requires_grad=False)
                .type(torch.FloatTensor)
                .cuda(device, non_blocking=True)
            )
        elif type(X) is torch.Tensor and not X.is_cuda:
            X = X.type(torch.FloatTensor).cuda(device, non_blocking=True)

        print("tensor size ", X.shape)

        if enable_dropout:
            self.enable_test_dropout()

        with torch.no_grad():
            out = self.forward(X)

        max_val, idx = torch.max(out, 1)
        idx = idx.data.cpu().numpy()
        prediction = np.squeeze(idx)
        print("prediction shape", prediction.shape)
        del X, out, idx, max_val
        return prediction

