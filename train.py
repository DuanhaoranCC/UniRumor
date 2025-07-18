# -*- coding: utf-8 -*-
# @Author  : Alisa
# @File    : main(pretrain).py
# @Software: PyCharm
import warnings
from evaluate import evaluate, train_test_split_few, context_inference
from torch_geometric.nn import global_add_pool
from torch_geometric.loader import DataLoader
import time
import numpy as np
import torch
import itertools
from torch.optim import Adam
from pargs import pargs
from load_data import load_datasets_with_prompts, TreeDataset, HugeDataset, TreeDataset_PHEME, TreeDataset_UPFD, \
    CovidDataset
from model import BiGCN_graphcl
from augmentation import augment
from torch_geometric import seed_everything

warnings.filterwarnings("ignore")


def get_scheduler(optimizer, use_scheduler=True, epochs=1000):
    if use_scheduler:
        scheduler = lambda epoch: (1 + np.cos(epoch * np.pi / epochs)) * 0.5
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=scheduler)
    else:
        scheduler = None

    return scheduler


def pre_train(loaders, aug1, aug2, model, optimizer, device):
    """
    Pre-train the model with multiple DataLoaders.

    :param loaders: List of DataLoaders for the datasets.
    :param aug1: String specifying the first set of augmentations.
    :param aug2: String specifying the second set of augmentations.
    :param model: The model to train.
    :param optimizer: Optimizer for the training process.
    :param device: Device to perform computations on (e.g., 'cuda' or 'cpu').
    :return: Average loss over all datasets.
    """
    model.train()
    total_loss = 0

    # Split augmentation strategies
    augs1 = aug1.split('||')
    augs2 = aug2.split('||')

    # Iterate through batches from each DataLoader, using itertools.zip_longest to handle different lengths
    for i, batches in enumerate(itertools.zip_longest(*loaders, fillvalue=None)):
        optimizer.zero_grad()

        augmented_data1 = []
        augmented_data2 = []

        # Process each batch from the different loaders
        for idx, batch in enumerate(batches):
            if batch is not None:  # Ensure the batch is not None (handle shorter datasets)
                batch = batch.to(device)
                aug_data1 = augment(batch, augs1)
                aug_data2 = augment(batch, augs2)
                # Attach prompt_key to the batch
                aug_data1.prompt_key = loaders[idx].prompt_key
                aug_data2.prompt_key = loaders[idx].prompt_key
                augmented_data1.append(aug_data1)
                augmented_data2.append(aug_data2)

        # Model forward pass
        out1 = model(*augmented_data1)
        out2 = model(*augmented_data2)
        #############################################
        # augmented_data1 = []
        #
        # # Process each batch from the different loaders
        # for idx, batch in enumerate(batches):
        #     if batch is not None:  # Ensure the batch is not None (handle shorter datasets)
        #         batch = batch.to(device)
        #         aug_data1 = augment(batch, augs1)
        #         augmented_data1.append(aug_data1)
        # # Model forward pass
        # loss = model(*augmented_data1)

        # Compute the loss using the contrastive loss function
        loss = model.loss_graphcl(out1, out2)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    total_loss /= len(loaders)
    return total_loss


def pre_trains(loaders, aug1, aug2, model, optimizer, device):
    """
    Pre-train the model with multiple DataLoaders.

    :param loaders: List of DataLoaders for the datasets.
    :param aug1: String specifying the first set of augmentations.
    :param aug2: String specifying the second set of augmentations.
    :param model: The model to train.
    :param optimizer: Optimizer for the training process.
    :param device: Device to perform computations on (e.g., 'cuda' or 'cpu').
    :return: Average loss over all datasets.
    """
    model.train()
    total_loss = 0

    # Split augmentation strategies
    augs1 = aug1.split('||')
    augs2 = aug2.split('||')

    # Iterate through batches from each DataLoader, using itertools.zip_longest to handle different lengths
    for data in loaders:
        optimizer.zero_grad()
        data = data.to(device)

        aug_data1 = augment(data, augs1)
        aug_data2 = augment(data, augs2)

        out1 = model(aug_data1)
        out2 = model(aug_data2)
        loss = model.loss_graphcl(out1, out2)
        print(loss.item())
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    total_loss /= len(loaders)
    return total_loss


if __name__ == '__main__':

    f1_macros_5 = []

    args = pargs()
    seed_everything(0)
    dataset = args.dataset
    device = torch.device(f'cuda:{args.cuda}' if torch.cuda.is_available() else 'cpu')

    batch_size = args.batch_size

    weight_decay = args.weight_decay
    epochs = args.epochs

    # Initialize datasets
    # data = TreeDataset("./Data/DRWeiboV3/")
    # data = TreeDataset("../ACL/Data/Weibo/")
    # data = TreeDataset("./Data/Twitter15-tfidf/")
    # data = TreeDataset("./Data/Twitter16-tfidf/")
    # data = TreeDataset_PHEME("./Data/pheme/")
    # data = TreeDataset_UPFD("./Data/politifact/")
    # data = TreeDataset_UPFD("./Data/gossipcop/")
    # data = CovidDataset("./Data/Twitter-COVID19/Twittergraph")
    # data = CovidDataset("./Data/Weibo-COVID19/Weibograph")
    # data = HugeDataset("./Data/Tree/")
    # train_loaders = DataLoader(data, batch_size=args.batch_size, shuffle=True, num_workers=48)
    # target_loader = DataLoader(data, batch_size=args.batch_size, shuffle=False)

    train_loaders, target_loader = load_datasets_with_prompts(args)

    # Model and optimizer initialization
    t = 0.5
    u = 0.5
    model = BiGCN_graphcl(768, args.out_feat, t, u).to(device)
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=weight_decay)
    # scheduler = get_scheduler(optimizer, epochs=epochs)

    for epoch in range(1, epochs + 1):
        pretrain_loss = pre_train(train_loaders,
                                  args.aug1, args.aug2, model, optimizer, device)
        # scheduler.step()
        print(f"Epoch: {epoch}, loss: {pretrain_loss}")
    # torch.save(model.state_dict(), f"./{dataset}_19.pt")
    print(dataset)

    # model.load_state_dict(torch.load(f"./{dataset}_3.pt", map_location=device))

    # Evaluation
    model.eval()
    x_list = []
    y_list = []
    for data in target_loader:
        data = data.to(device)
        embeds = model.get_embeds(data).detach()
        # embeds = global_add_pool(data.x, data.batch).detach()
        ################################
        # root_indices = []
        # batch_size = max(data.batch) + 1
        # for num_batch in range(batch_size):
        #     root_indices.append(torch.nonzero(data.batch == num_batch, as_tuple=False)[0].item())
        # embeds = data.x[root_indices]
        ################################
        y_list.append(data.y)
        x_list.append(embeds)
    x = torch.cat(x_list, dim=0)
    y = torch.cat(y_list, dim=0)

    ################################################################################################
    for r in [1, 5]:
        mask = train_test_split_few(y.cpu().numpy(), seed=0,
                                    train_examples_per_class=r,
                                    val_size=500, test_size=None)
        train_mask_l = f"{r}_train_mask"
        train_mask = mask['train'].astype(bool)
        val_mask_l = f"{r}_val_mask"
        val_mask = mask['val'].astype(bool)

        test_mask_l = f"{r}_test_mask"
        test_mask = mask['test'].astype(bool)

        evaluate(x, y, device, 0.01, 0.0, train_mask, val_mask, test_mask)
    ############################################################################################
    # for i in range(10):
    #     for r in [1, 5]:
    #         mask = train_test_split_few(y.cpu().numpy(), seed=i,
    #                                     train_examples_per_class=r,
    #                                     val_size=500, test_size=None)
    #         train_mask_l = f"{r}_train_mask"
    #         train_mask = mask['train'].astype(bool)
    #         val_mask_l = f"{r}_val_mask"
    #         val_mask = mask['val'].astype(bool)
    #
    #         test_mask_l = f"{r}_test_mask"
    #         test_mask = mask['test'].astype(bool)
    #
    #         # evaluate(x, y, device, 0.01, 0.0, train_mask, val_mask, test_mask)
    #         f1_macro, f1_micro = context_inference(x, y, device, train_mask, test_mask)
    #         if r == 1:
    #             f1_macros_1.append(f1_macro)
    #         if r == 5:
    #             f1_macros_5.append(f1_macro)
    # # Compute mean and standard deviation
    # f1_macro_1_mean, f1_macro_1_std = np.mean(f1_macros_1), np.std(f1_macros_1)
    #
    # f1_macro_5_mean, f1_macro_5_std = np.mean(f1_macros_5), np.std(f1_macros_5)
    #
    # # Print final results
    # print("\nFinal Results (10 runs) for 1-shot:")
    # print(f"Macro-F1: Mean = {f1_macro_1_mean:.4f}, Std = {f1_macro_1_std:.4f}")
    #
    # print("\nFinal Results (10 runs) for 5-shot:")
    # print(f"Macro-F1: Mean = {f1_macro_5_mean:.4f}, Std = {f1_macro_5_std:.4f}")
