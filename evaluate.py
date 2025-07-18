import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score
from torch.nn.functional import softmax
from sklearn.metrics import roc_auc_score

class LogReg(nn.Module):
    def __init__(self, num_features, num_classes):
        super(LogReg, self).__init__()
        self.fc = nn.Linear(num_features, num_classes)
        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)

    def forward(self, seq):
        return self.fc(seq)


def sample_per_class(random_state, labels, num_examples_per_class, forbidden_indices=None):
    num_samples = labels.shape[0]
    num_classes = labels.max() + 1
    sample_indices_per_class = {index: [] for index in range(num_classes)}

    # get indices sorted by class
    for class_index in range(num_classes):
        for sample_index in range(num_samples):
            if labels[sample_index] == class_index:
                if forbidden_indices is None or sample_index not in forbidden_indices:
                    sample_indices_per_class[class_index].append(sample_index)

    # get specified number of indices for each class
    return np.concatenate(
        [random_state.choice(sample_indices_per_class[class_index], num_examples_per_class, replace=False)
         for class_index in range(len(sample_indices_per_class))
         ])


def get_train_val_test_split(random_state,
                             labels,
                             train_examples_per_class=None, val_examples_per_class=None,
                             test_examples_per_class=None,
                             train_size=None, val_size=None, test_size=None):
    num_samples = labels.shape[0]
    num_classes = labels.max() + 1
    remaining_indices = list(range(num_samples))

    if train_examples_per_class is not None:
        train_indices = sample_per_class(
            random_state, labels, train_examples_per_class)
    else:
        # select train examples with no respect to class distribution
        train_indices = random_state.choice(
            remaining_indices, train_size, replace=False)

    if val_examples_per_class is not None:
        val_indices = sample_per_class(
            random_state, labels, val_examples_per_class, forbidden_indices=train_indices)
    else:
        remaining_indices = np.setdiff1d(remaining_indices, train_indices)
        if val_size is not None and len(remaining_indices) < val_size:
            val_size = min(100, len(remaining_indices))
        val_indices = random_state.choice(
            remaining_indices, val_size, replace=False)

    forbidden_indices = np.concatenate((train_indices, val_indices))
    if test_examples_per_class is not None:
        test_indices = sample_per_class(random_state, labels, test_examples_per_class,
                                        forbidden_indices=forbidden_indices)
    elif test_size is not None:
        remaining_indices = np.setdiff1d(remaining_indices, forbidden_indices)
        test_indices = random_state.choice(
            remaining_indices, test_size, replace=False)
    else:
        test_indices = np.setdiff1d(remaining_indices, forbidden_indices)

    # assert that there are no duplicates in sets
    assert len(set(train_indices)) == len(train_indices)
    assert len(set(val_indices)) == len(val_indices)
    assert len(set(test_indices)) == len(test_indices)
    # assert sets are mutually exclusive
    assert len(set(train_indices) - set(val_indices)
               ) == len(set(train_indices))
    assert len(set(train_indices) - set(test_indices)
               ) == len(set(train_indices))
    assert len(set(val_indices) - set(test_indices)) == len(set(val_indices))
    if test_size is None and test_examples_per_class is None:
        # all indices must be part of the split
        assert len(np.concatenate(
            (train_indices, val_indices, test_indices))) == num_samples

    if train_examples_per_class is not None:
        train_labels = labels[train_indices]
        train_sum = np.sum(train_labels, axis=0)
        # assert all classes have equal cardinality
        assert np.unique(train_sum).size == 1

    if val_examples_per_class is not None:
        val_labels = labels[val_indices]
        val_sum = np.sum(val_labels, axis=0)
        # assert all classes have equal cardinality
        assert np.unique(val_sum).size == 1

    if test_examples_per_class is not None:
        test_labels = labels[test_indices]
        test_sum = np.sum(test_labels, axis=0)
        # assert all classes have equal cardinality
        assert np.unique(test_sum).size == 1

    return train_indices, val_indices, test_indices


def train_test_split_few(labels, seed, train_examples_per_class=None, val_examples_per_class=None,
                     test_examples_per_class=None, train_size=None, val_size=None, test_size=None):
    random_state = np.random.RandomState(seed=seed)
    train_indices, val_indices, test_indices = get_train_val_test_split(
        random_state, labels, train_examples_per_class, val_examples_per_class, test_examples_per_class, train_size,
        val_size, test_size)

    # print('number of training: {}'.format(len(train_indices)))
    # print('number of validation: {}'.format(len(val_indices)))
    # print('number of testing: {}'.format(len(test_indices)))

    train_mask = np.zeros((labels.shape[0], 1), dtype=int)
    train_mask[train_indices, 0] = 1
    train_mask = np.squeeze(train_mask, 1)
    val_mask = np.zeros((labels.shape[0], 1), dtype=int)
    val_mask[val_indices, 0] = 1
    val_mask = np.squeeze(val_mask, 1)
    test_mask = np.zeros((labels.shape[0], 1), dtype=int)
    test_mask[test_indices, 0] = 1
    test_mask = np.squeeze(test_mask, 1)
    mask = {}
    mask['train'] = train_mask
    mask['val'] = val_mask
    mask['test'] = test_mask
    return mask


##################################################
# This section of code adapted from pcy1302/DMGI #
##################################################

def evaluate(embeds, label, device, lr, wd, train_mask, val_mask, test_mask):
    num_features = embeds.shape[1]
    num_classes = label.max() + 1
    embeds = embeds.to(device)
    label = label.to(device)
    xent = nn.CrossEntropyLoss()

    train_lbls = label[train_mask]
    val_lbls = label[val_mask]
    test_lbls = label[test_mask]
    train_embs = embeds[train_mask]
    val_embs = embeds[val_mask]
    test_embs = embeds[test_mask]

    micro_f1s = []
    macro_f1s = []
    macro_f1s_val = []
    auc_score_list = []

    for _ in range(10):
        log = LogReg(num_features, num_classes).to(device)
        opt = torch.optim.Adam(log.parameters(), lr=lr, weight_decay=wd)

        val_micro_f1s = []
        test_micro_f1s = []
        val_macro_f1s = []
        test_macro_f1s = []
        logits_list = []
        train_embs = train_embs.to(device)
        train_lbls = train_lbls.to(device)
        for i in range(200):
            log.train()
            opt.zero_grad()
            logits = log(train_embs)
            loss = xent(logits, train_lbls)
            loss.backward()
            opt.step()

            ##########################################################################
            # Val
            logits = log(val_embs.to(device))
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            val_f1_macro = f1_score(val_lbls.cpu().numpy(), preds, average='macro')
            val_f1_micro = f1_score(val_lbls.cpu().numpy(), preds, average='micro')

            val_macro_f1s.append(val_f1_macro)
            val_micro_f1s.append(val_f1_micro)

            # #####################################################################
            # Test
            logits = log(test_embs)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            test_f1_macro = f1_score(test_lbls.cpu().numpy(), preds, average='macro')
            test_f1_micro = f1_score(test_lbls.cpu().numpy(), preds, average='micro')

            test_macro_f1s.append(test_f1_macro)
            test_micro_f1s.append(test_f1_micro)
            logits_list.append(logits)
        #################################################################################
        max_iter = val_macro_f1s.index(max(val_macro_f1s))
        macro_f1s.append(test_macro_f1s[max_iter])
        macro_f1s_val.append(val_macro_f1s[max_iter])
        # #################################################################################
        max_iter = val_micro_f1s.index(max(val_micro_f1s))
        micro_f1s.append(test_micro_f1s[max_iter])
        # auc
        best_logits = logits_list[max_iter]
        best_proba = softmax(best_logits, dim=1)
        auc_score_list.append(
            roc_auc_score(
                y_true=F.one_hot(test_lbls).detach().cpu().numpy(),
                y_score=best_proba.detach().cpu().numpy(),
                multi_class='ovr'
            )
        )

    print("Macro-F1_mean: {:.4f} var: {:.4f}  Micro-F1_mean: {:.4f} var: {:.4f} auc {:.4f} var: {:.4f}"
    .format(
        np.mean(macro_f1s),
        np.std(macro_f1s),
        np.mean(micro_f1s),
        np.std(micro_f1s),
        np.mean(auc_score_list),
        np.std(auc_score_list)
    )
    )
    return np.mean(micro_f1s)


def context_inference(embeds, labels, device, train_mask, test_mask, method='mean', k=5):
    """
    Implements in-context inference for Few-Shot classification.

    Parameters:
    - embeds: [N, d], node embeddings for all samples.
    - labels: [N], labels for all nodes.
    - device: device to run the computation on.
    - train_mask: boolean mask for the training set.
    - test_mask: boolean mask for the test set.
    - method: 'mean' for class center-based classification, 'knn' for KNN classification.
    - k: number of nearest neighbors (only used when method='knn').

    Returns:
    - Macro-F1 and Micro-F1 scores.
    """
    embeds = embeds.to(device)
    labels = labels.to(device)

    # Extract support (train) and query (test) embeddings and labels
    support_labels = labels[train_mask]  # [num_train]
    query_labels = labels[test_mask]  # [num_test]
    support_embeddings = embeds[train_mask]  # [num_train, d]
    query_embeddings = embeds[test_mask]  # [num_test, d]

    if method == 'mean':
        # **Mean Embedding Classification**
        unique_classes = torch.unique(support_labels)  # Extract unique classes
        class_embeddings = []

        # Compute class centers (mean embeddings)
        for cls in unique_classes:
            class_mask = support_labels == cls  # Select samples of current class
            class_embedding = support_embeddings[class_mask].mean(dim=0)  # Compute class mean
            class_embeddings.append(class_embedding)

        class_embeddings = torch.stack(class_embeddings)  # [num_classes, d]

        # Compute cosine similarity between query embeddings and class centers
        similarities = F.cosine_similarity(query_embeddings.unsqueeze(1), class_embeddings.unsqueeze(0),
                                           dim=2)  # [num_test, num_classes]

        # Assign the class with the highest similarity
        pred_classes = unique_classes[torch.argmax(similarities, dim=1)]  # [num_test]

    elif method == 'knn':
        # **KNN Classification**
        num_query = query_embeddings.shape[0]

        # Compute Euclidean distance between query and support embeddings
        distances = torch.cdist(query_embeddings, support_embeddings, p=2)  # [num_test, num_train]

        # Select the k nearest neighbors
        knn_indices = torch.argsort(distances, dim=1)[:, :k]  # [num_test, k]

        # Retrieve labels of nearest neighbors
        knn_labels = support_labels[knn_indices]  # [num_test, k]

        # Majority voting to determine the predicted class
        pred_classes = torch.mode(knn_labels, dim=1)[0]  # [num_test]

    else:
        raise ValueError("The method parameter must be 'mean' or 'knn'.")

    # Compute F1 scores
    f1_macro = f1_score(query_labels.cpu().numpy(), pred_classes.cpu().numpy(), average='macro')
    f1_micro = f1_score(query_labels.cpu().numpy(), pred_classes.cpu().numpy(), average='micro')

    # print(f"Method: {method} | Macro-F1: {f1_macro:.4f}  Micro-F1: {f1_micro:.4f}")

    return f1_macro, f1_micro
