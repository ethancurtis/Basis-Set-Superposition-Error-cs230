import torch
import os
import torch_funcs
import glob
import numpy as np



class EarlyStopping():
    """
    Get a callback for early-stopping of the training-process

    Parameters:
    monitor (string): Must be of the options ["val_loss", "train_loss"]. Default: "val_loss"
    patience (int): Number of epoch to wait before terminating training process

    Returns:
    Early stopping callable class that takes callback_params as argument
    """
    def __init__(self, monitor="val_loss", patience=20, v=True):
        assert(monitor in ["val_loss", "train_loss"])
        self.monitor = monitor
        self.patience = patience
        self.v = v

    def pre_train(self, callback_params, v=False):
        history = callback_params["history"]
        if history["epoch"]:
            self.best_epoch = history["epoch"][-1]
        else:
            self.best_epoch = 0
        return False

    def post_epoch(self, callback_params, v=False):
        history = callback_params["history"]
        current_epoch = history["epoch"][-1]
        if history["val_loss"][-1] <= history["val_loss"][self.best_epoch]:
            if v:
                print(", Best epoch", end="")
            self.best_epoch = current_epoch
            return False
        if current_epoch - self.best_epoch > self.patience:
            print("\nEarly stopping at epoch {}".format(history["epoch"][-1]))
            print("Best epoch: {}, train loss: {:1.4e}, val loss: {:1.4e}".format(
                self.best_epoch, history["train_loss"][self.best_epoch],history["val_loss"][self.best_epoch]))
            return True
        return False

    def post_train(self, callback_params, v=False):
        return False

class Checkpoints():
    """
    Get a callback to restore the make checkpoints of modelweights, the optimizer and history

    Parameters:
    monitor (string): Must be of the options ["val_loss", "train_loss"]. Default: "val_loss"
    checkpoint_dir (str): Where to save the create the model checkpoint dir
    model_dit (str): If spefified, the model dir with be used even if it exists. Default None
    checkpoint_start: No checkpoints will be saved before epoch

    Returns:
    Checkpoints callable class that takes callback_params as argument
    """
    def __init__(self, monitor="val_loss", checkpoint_dir="Checkpoints/", model_dir=None, checkpoint_start=200, checkpoint_frequency=20, checkpoint_best_epoch=True, restore_best_epoch=False):
        assert(monitor in ["val_loss", "train_loss"])
        self.monitor = monitor
        self.checkpoint_start = checkpoint_start
        self.checkpoint_frequency = checkpoint_frequency
        self.checkpoint_best_epoch = checkpoint_best_epoch
        self.restore_best_epoch = restore_best_epoch
        self.saved_checkpoint = False
        self.checkpoint_filename = None
        self.train_end_checkpoint_filename = None
        self.checkpoint_dir = checkpoint_dir
        self.model_dir = model_dir
        self.save_path = None

        assert(os.path.isdir(checkpoint_dir))
        self.set_model_dir(model_dir)
        

    def pre_train(self, callback_params, v=False):
        history = callback_params["history"]
        if history["epoch"]:
            self.best_epoch = history["epoch"][-1]
        else:
            self.best_epoch = 0
        return False

    def post_epoch(self, callback_params, v=False):
        history = callback_params["history"]
        model = callback_params["model"]
        optimizer = callback_params["optimizer"]
        current_epoch = history["epoch"][-1]

        make_checkpoint = False
        if current_epoch < self.checkpoint_start:
            return False
        if self.checkpoint_frequency:
            if current_epoch % self.checkpoint_frequency == 0:
                make_checkpoint = True
        if self.checkpoint_best_epoch:
            if history[self.monitor][-1] < history[self.monitor][self.best_epoch]:
                self.best_epoch = current_epoch
                make_checkpoint = True

        if make_checkpoint:
            self.saved_checkpoint = True
            if v:
                print(", Checkpoint", end="")
            checkpoint_filename = "checkpoint_e{}_{:.4e}.pt".format(current_epoch, history[self.monitor][-1])
            path = os.path.join(self.save_path, checkpoint_filename)
            save_model_checkpoint(path, model, optimizer, history=history)
    
        return False
        
    def post_train(self, callback_params, v=False):
        history = callback_params["history"]
        model = callback_params["model"]
        optimizer = callback_params["optimizer"]
        current_epoch = history["epoch"][-1]

        self.train_end_checkpoint_filename = os.path.join(self.save_path, "checkpoint_e{}_{:.4e}_end.pt".format(current_epoch, history[self.monitor][-1]))
        save_model_checkpoint(self.train_end_checkpoint_filename, model, optimizer, history=history)

        if self.saved_checkpoint:
            if self.restore_best_epoch:
                if self.load_best_epoch(model, optimizer=optimizer):
                    if v:
                        print("Weights restored to epoch {}, train loss: {:1.4e}, val loss: {:1.4e}".format(
                            self.best_epoch, history["train_loss"][self.best_epoch],history["val_loss"][self.best_epoch]))
                else:
                    print("Could not load checkpoint corresponding to the best epoch ({})".format(self.best_epoch))
       
        path = os.path.join(self.save_path, "model.pt")
        torch_funcs.save_model(path, model, optimizer, history, overwrite=True)

        return False

    def load_best_epoch(self, model, optimizer=None, history=None):
        best_epoch_filename = glob.glob(os.path.join(self.save_path, "*_e{}_*".format(self.best_epoch)))
        if len(best_epoch_filename) == 1:
            print("Best epoch checkpoint \"{}\" loaded".format(os.path.basename(best_epoch_filename[0])))
            try:
                load_model_checkpoint(best_epoch_filename[0], model, optimizer)
                return 0
            except:
                return 1
        print(best_epoch_filename)
        return 1

    def load_train_end(self, model, optimizer=None, history=None):
        load_path = ""
        if self.train_end_checkpoint_filename is None:
            train_end_files = glob.glob(os.path.join(self.save_path, "*end.pt"))
            if not train_end_files:
                print("No end of train checkpoints found")
                return 1
            if len(train_end_files) == 1:
                load_path = train_end_files
            else:
                end_epochs = [int(os.path.basename(path).split("_")[1][1:]) for path in train_end_files]   
                argmax = np.argmax(end_epochs)
                load_path = train_end_files[argmax]
        else:
            load_path = self.train_end_checkpoint_filename

        try:
            load_model_checkpoint(load_path, model, optimizer=optimizer, history=history)
            print("Train end checkpoint \"{}\" loaded".format(os.path.basename(load_path)))
            return 0
        except:
            return 1

    def load_epoch_checkpoint(self, epoch, model, optimizer=None, history=None):
        epoch_filename = glob.glob(os.path.join(self.save_path, "*_e{}_*".format(epoch)))
        if not epoch_filename:
            print("Could not find checkpoint corresponding to epoch {}".format(epoch))
            print("Looking for checkpoints in {}".format(self.save_path))
            return 1
        if len(epoch_filename) != 1:
            print("Found {} checkpoints corresponding to epoch {}".format(len(epoch_filename, epoch)))
            return 1
        load_model_checkpoint(epoch_filename[0], model, optimizer=optimizer, history=history)
        print("Checkpoint \"{}\" loaded".format(os.path.basename(epoch_filename[0])))

    def set_model_dir(self, model_dir):
        if model_dir is None:
            for i in range(1, 1000):
                model_dir = "model_{}".format(i)
                path = os.path.join(self.checkpoint_dir, model_dir)
                if not os.path.isdir(path):
                    self.save_path = path
                    break
        else:
            self.save_path = os.path.join(self.checkpoint_dir, model_dir)

        if self.save_path is None:
            print("Delete some checkpoints in {} before continuing!".format(checkpoint_dir))
            return
        print("Saving checkpoints in {}".format(self.save_path))
        self.model_dir = model_dir
        if not os.path.isdir(self.save_path):
            os.makedirs(self.save_path)


def save_model_checkpoint(path, model, optimizer, history=None):
    torch.save({"model_state_dict":model.state_dict(), 
                "optimizer_state_dict":optimizer.state_dict(),
                "history":history
    },
    path)

def load_model_checkpoint(path, model, optimizer=None, history=None):
    checkpoint = torch.load(path)
    if "history" in checkpoint:
        history =  checkpoint["history"]
    else:
        history = None

    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    return {"model": model, "optimizer":optimizer, "history":history}

def load_checkpoint(path):
    checkpoint = torch.load(path)
    return checkpoint

    # load_checkpoint