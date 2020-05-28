import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import os


def filter_partition(data, partition):
    index = data.index

    if type(partition) != set:
        partition = set(partition)

    
    drop_indices = []
    for i in index:
        if data.loc[i,"molecule"] not in partition:
            drop_indices.append(i)
    
    return data.drop(drop_indices)

def plot_history(history, start_epoch=0, size=None):
    train_loss = np.array(history["train_loss"])
    val_loss = np.array(history["val_loss"])
    epoch = np.array(history["epoch"])

    start_index = np.argwhere(epoch == start_epoch).flatten()[0]

    train_loss = train_loss[start_index:]
    val_loss = val_loss[start_index:]
    epoch = epoch[start_index:]

    if size:
        plt.figure(figsize=size)
    plt.plot(epoch, train_loss, label="Train loss")
    plt.plot(epoch, val_loss, label="Val loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
    plt.xlim(np.min(epoch), np.max(epoch))

    plt.legend()
    plt.show()

def train_model(model, loss_function, optimizer, train_dataloader, val_dataloader, n_epochs=1000, callbacks=[], history=None, output="long", device="cpu"):
    if history is None:
        history = {"train_loss":[],"val_loss":[], "epoch":[]}
        start_epoch = 0
    else:
        start_epoch = history["epoch"][-1] + 1
    
    callback_params = {"model":model, "optimizer":optimizer, "history":history}

    assert(output in [None, "long", "short"])
    v = True
    if output is None:
        v = False
    elif output == "long":
        end = "\n"
    else: #output == "short"
        end = ""

    for callback in callbacks:
        callback.pre_train(callback_params, v=v)

    for epoch in range(start_epoch, start_epoch + n_epochs):
        history["epoch"].append(epoch)

        #First run through the train batches
        cum_loss = 0  
        n_train = 0      
        model.train()
        for batch in train_dataloader:
            x_batch, target = model.unpack_batch(batch)#, device=device)
            prediction = model(x_batch)
            loss = loss_function(prediction, target)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            n_batch = target.shape[0]
            cum_loss += loss.item() * n_batch
            n_train += n_batch
        history["train_loss"].append(cum_loss / n_train)
            
        #Then run through the validation batches
        cum_loss = 0 #torch.tensor(0, dtype=torch.float, device=device) #0
        n_val = 0 #torch.tensor(0, device=device) # 0
        model.eval()
        for batch in val_dataloader:
            x_batch, target = model.unpack_batch(batch)#, device=device)
            prediction = model(x_batch)
            loss = loss_function(prediction, target)
            
            n_batch = target.shape[0]
            cum_loss += loss.item() * n_batch
            n_val += n_batch
        history["val_loss"].append(cum_loss / n_val)

        #Print output
        if v:
            print("\rEpoch: {}, train loss: {:1.4e}, val loss: {:1.4e}".format(
            epoch, history["train_loss"][epoch],history["val_loss"][epoch]),end="")

        stop_train = False
        for callback in callbacks:
            stop_train += callback.post_epoch(callback_params, v=v)
        if v:
            print("                                              ", end=end) #Newline character after epoch. Make sure to overwrite everything
        if stop_train:
            break
    print("\n") #Just an extra newline after training finishes

    for callback in callbacks:
        callback.post_train(callback_params, v=v)

    return history


def save_model(path, model, optimizer=None, history=None, overwrite=False):
    """
    Save model dict at path
    """
    model_dict = {
        "model":model,
        "optimizer":optimizer,
        "history":history
    }

    if not overwrite:
        if os.path.isfile(path):
            print("File is already found at {}".format(path))
            response = "no"
            while response != "yes":
                response = input("Overwrite model? (yes/no)")
                if response == "no":
                    print("Model not saved")
                    return
    torch.save(model_dict, path)

def load_model(path):
    """
    Load model dict at path
    """
    return torch.load(path)