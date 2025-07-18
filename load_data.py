import os
import json
import pickle
import numpy
import torch
import numpy as np
from datetime import datetime
import re
import scipy.sparse as sp
from torch_geometric.io import read_txt_array
from torch_geometric.data import Data, InMemoryDataset, Dataset
from torch_geometric.data import DataLoader
from torch_geometric.utils import degree, to_undirected, coalesce, cumsum
from transformers import XLMRobertaTokenizer, XLMRobertaModel
from sklearn.decomposition import PCA
from torch_scatter import scatter


def pca_compression(seq, k):
    pca = PCA(n_components=k)
    seq = pca.fit_transform(seq)

    # print(pca.explained_variance_ratio_.sum())
    return seq


def min_max_normalize(features):
    """
    对特征进行最大-最小归一化，将值缩放到 [0, 1]。
    """
    min_vals = features.min(axis=0)
    max_vals = features.max(axis=0)
    normalized = (features - min_vals) / (max_vals - min_vals + 1e-8)  # 避免除以零
    return normalized


def standardize(features):
    """
    对特征进行标准化，调整为均值为 0，标准差为 1。
    """
    mean_vals = features.mean(axis=0)
    std_vals = features.std(axis=0) + 1e-8  # 避免除以零
    standardized = (features - mean_vals) / std_vals
    return standardized


def svd_compression(seq, k):
    res = np.zeros_like(seq)
    # 进行奇异值分解, 从svd函数中得到的奇异值sigma 是从大到小排列的
    U, Sigma, VT = np.linalg.svd(seq, full_matrices=False)
    # print(U[:, :k].shape)
    # print(VT[:k, :].shape)
    # res = U[:, :k].dot(np.diag(Sigma[:k]))
    # 只保留前 k 个奇异值对应的部分
    U_k = U[:, :k]
    Sigma_k = np.diag(Sigma[:k])
    # VT_k = VT[:k, :]

    # 压缩后的特征
    compressed_seq = U_k.dot(Sigma_k)

    return compressed_seq


def text_to_vector(texts, model, tokenizer, max_length=128, batch_size=16):
    """
    使用 XLM-RoBERTa 将文本列表转化为向量，支持批量处理。
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    vectors = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        inputs = tokenizer(batch_texts, return_tensors="pt", truncation=True, padding=True, max_length=max_length).to(
            device)
        with torch.no_grad():
            outputs = model(**inputs)
        vectors.append(outputs.last_hidden_state[:, 0, :].cpu())  # 使用 [CLS] token 的输出
    return torch.cat(vectors, dim=0)


def parse_date(date_str):
    """
    Parses the date string in the format 'Sat Aug 09 22:33:06 +0000 2014'.
    """
    return datetime.strptime(date_str, "%a %b %d %H:%M:%S %z %Y")


def preprocess_text(value):
    """综合清洗函数：处理 Unicode 引号、非 ASCII 字符、URL、提及、主题标签、标点和空格"""
    if isinstance(value, str):
        # 替换特殊的 Unicode 引号
        value = value.replace('\u201c', '').replace('\u201d', '')
        value = value.replace('\u2018', "").replace('\u2019', "")

        # 移除非 ASCII 字符
        value = re.sub(r'[^\x00-\x7F]+', '', value)

        # 移除 URL
        value = re.sub(r'http\S+|www\.\S+', '', value)

        # 移除提及和主题标签
        value = re.sub(r'@\w+', '', value)  # 删除 @提及
        value = re.sub(r'#\w+', '', value)  # 删除 #主题标签

        # 标准化标点符号
        value = re.sub(r'\.{2,}', '.', value)  # 替换多个点

        # 删除反斜杠
        value = value.replace('//', '')  # 删除反斜杠

        # 去除多余空格
        value = ' '.join(value.split())

    return value


def trans_time(t, t_init):
    """
    Converts time strings into seconds since a reference time (t_init).
    """
    try:
        t_seconds = int(datetime.strptime(t, "%a %b %d %H:%M:%S +0000 %Y").timestamp())
        t_init_seconds = int(datetime.strptime(t_init, "%a %b %d %H:%M:%S +0000 %Y").timestamp())
        return t_seconds - t_init_seconds
    except Exception as e:
        print(f"Error parsing time: {e}")
        return None


class TreeDataset(InMemoryDataset):
    def __init__(self, root, centrality_metric="PageRank", undirected=False, transform=None, pre_transform=None,
                 pre_filter=None):
        self.undirected = undirected
        self.centrality_metric = centrality_metric
        # self.tokenizer = XLMRobertaTokenizer.from_pretrained('xlm-roberta-base', cache_dir="./")
        # self.model = XLMRobertaModel.from_pretrained('xlm-roberta-base', cache_dir="./")
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return os.listdir(self.raw_dir)

    @property
    def processed_file_names(self):
        return ['data.pt']

    def download(self):
        pass

    def process(self):
        data_list = []
        all_data_names = self.raw_file_names
        xx = []
        # for filename in all_data_names:
        #     tweets = os.listdir(os.path.join(self.raw_dir, filename))
        #     print(filename)
        filename = self.root
        for tweet in all_data_names:
            print(tweet)
            y = []
            centrality = None
            row = []
            col = []
            no_root_row = []
            no_root_col = []
            edges = []
            post = json.load(open(os.path.join(self.raw_dir, tweet), 'r', encoding='utf-8'))

            if "15" in filename or "16" in filename:
                tfidf = post['source']['content']
                indices = [[0, int(index_freq.split(':')[0])] for index_freq in tfidf.split()]
                values = [int(index_freq.split(':')[1]) for index_freq in tfidf.split()]
            else:
                # 提取内容
                texts = [post["source"]["content"]]  # 添加主内容
                # texts += [comment["content"] for comment in post["comment"] if comment["content"].strip()]  # 添加非空评论

            if 'label' in post['source'].keys():
                y.append(post['source']['label'])
            else:
                y.append(-1)

            if "time" in post["source"]:
                # 定义原始时间格式
                original_format = "%y-%m-%d %H:%M"

                # 解析为 datetime 对象
                parsed_time = datetime.strptime(post["source"]["time"], original_format)

                # 转换为目标格式
                init_time = parsed_time.strftime("%a %b %d %H:%M:%S +0000 %Y")

            for i, comment in enumerate(post['comment']):
                if "time" in post["source"]:
                    # 解析为 datetime 对象
                    parsed_time = datetime.strptime(comment["time"], original_format)

                    # 转换为目标格式
                    post_time = parsed_time.strftime("%a %b %d %H:%M:%S +0000 %Y")
                    edge_time = trans_time(post_time, init_time)
                    # edge_time = np.log(1 + np.abs(edge_time))
                    edges.append(edge_time)
                else:
                    edges.append(1.0)
                if "15" in filename or "16" in filename:
                    indices += [[i + 1, int(index_freq.split(':')[0])] for index_freq in comment['content'].split()]
                    values += [int(index_freq.split(':')[1]) for index_freq in comment['content'].split()]
                elif comment['content'] == "":
                    txt = "转发"
                    texts += [txt]
                else:
                    texts += [comment["content"]]  # 添加非空评论

                if comment['parent'] != -1:
                    no_root_row.append(comment['parent'] + 1)
                    no_root_col.append(comment['comment id'] + 1)
                row.append(comment['parent'] + 1)
                col.append(comment['comment id'] + 1)

            # if self.centrality_metric == "Degree":
            #     centrality = torch.tensor(post['centrality']['Degree'], dtype=torch.float32)
            # elif self.centrality_metric == "PageRank":
            centrality = torch.tensor(post['centrality']['Pagerank'], dtype=torch.float32)
            # elif self.centrality_metric == "Eigenvector":
            #     centrality = torch.tensor(post['centrality']['Eigenvector'], dtype=torch.float32)
            # elif self.centrality_metric == "Betweenness":
            #     centrality = torch.tensor(post['centrality']['Betweenness'], dtype=torch.float32)

            edge_index = [row, col]
            y = torch.LongTensor(y)
            edge_index = to_undirected(torch.LongTensor(edge_index)) if self.undirected else torch.LongTensor(
                edge_index)
            if "15" in filename or "16" in filename:
                x = torch.sparse_coo_tensor(torch.tensor(indices).t(), values, (len(post['comment']) + 1, 5000),
                                            dtype=torch.float32).to_dense()
            else:
                x = text_to_vector(texts, self.model, self.tokenizer)
            if "Weibo" in filename:
                lang = torch.tensor([1])
            else:
                lang = torch.tensor([0])

            one_data = Data(x=x, y=y, edge_index=edge_index,
                            edge_attr=torch.FloatTensor(edges), lang=lang, centrality=centrality)
            data_list.append(one_data)

        if "15" in filename or "16" in filename:
            # 合并所有图的节点特征
            for data in data_list:
                xx.append(data.x)  # data.x 是节点特征矩阵
            all_features = torch.cat(xx).numpy()
            # normalized_features = min_max_normalize(all_features)
            # all_features = standardize(normalized_features)

            # 对合并的特征矩阵进行 PCA 降维
            pca = PCA(n_components=768)
            reduced_features = pca.fit_transform(all_features)
            reduced_features = torch.FloatTensor(reduced_features)

            # 将降维后的特征分配回每个图
            start_idx = 0
            for data in data_list:
                num_nodes = data.x.shape[0]
                data.x = reduced_features[start_idx: start_idx + num_nodes]  # 取对应行
                start_idx += num_nodes

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]
        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]
        all_data, slices = self.collate(data_list)
        torch.save((all_data, slices), self.processed_paths[0])


def read_json_file(file_path):
    """
    Reads a JSON file and returns its content as a dictionary.
    Args:
        file_path (str): Path to the JSON file.
    Returns:
        dict or None: The content of the JSON file, or None if an error occurs.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error reading JSON file '{file_path}': {e}")
        return None


# need normalization
# root centrality = no children 1 level reply centrality
def pagerank_centrality(data, damp=0.85, k=10):
    device = data.x.device
    bu_edge_index = data.edge_index.clone()
    bu_edge_index[0], bu_edge_index[1] = data.edge_index[1], data.edge_index[0]

    num_nodes = data.num_nodes
    deg_out = degree(bu_edge_index[0])
    centrality = torch.ones((num_nodes,)).to(device).to(torch.float32)

    for i in range(k):
        edge_msg = centrality[bu_edge_index[0]] / deg_out[bu_edge_index[0]]
        agg_msg = scatter(edge_msg, bu_edge_index[1], reduce='sum')
        pad = torch.zeros((len(centrality) - len(agg_msg),)).to(device).to(torch.float32)
        agg_msg = torch.cat((agg_msg, pad), 0)

        centrality = (1 - damp) * centrality + damp * agg_msg

    centrality[0] = centrality.min().item()
    return centrality


class TreeDataset_PHEME(InMemoryDataset):
    def __init__(self, root, centrality_metric="Pagerank", undirected=False, transform=None, pre_transform=None,
                 pre_filter=None):
        self.undirected = undirected
        self.centrality_metric = centrality_metric
        # self.tokenizer = XLMRobertaTokenizer.from_pretrained('xlm-roberta-base', cache_dir="./")
        # self.model = XLMRobertaModel.from_pretrained('xlm-roberta-base', cache_dir="./")
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return os.listdir(self.raw_dir)

    @property
    def processed_file_names(self):
        return ['data.pt']

    def download(self):
        pass

    def process(self):
        data_list = []
        event_list = ['germanwings-crash-all-rnr-threads', 'charliehebdo-all-rnr-threads',
                      'sydneysiege-all-rnr-threads', 'ebola-essien-all-rnr-threads',
                      'gurlitt-all-rnr-threads', 'putinmissing-all-rnr-threads',
                      'ferguson-all-rnr-threads', 'ottawashooting-all-rnr-threads',
                      'prince-toronto-all-rnr-threads']
        for event in event_list:
            event_path = os.path.join(self.raw_dir, self.raw_file_names[0], event)
            if not os.path.exists(event_path):
                continue

            # Process non-rumor data
            non_rumor_path = os.path.join(event_path, 'non-rumours')
            if os.path.exists(non_rumor_path):

                # Iterate through each news item in the event directory
                for news in os.listdir(non_rumor_path):
                    print(news)
                    # if news == "580320890086383616":
                    #     print("DDDD")
                    json_content = {"source": {}, "comment": []}
                    if not news.startswith('._') and news != '.DS_Store':
                        if len(os.listdir(os.path.join(event_path, 'non-rumours', news, 'reactions'))) == 0:
                            continue
                        source_tweets_path = os.path.join(event_path, 'non-rumours', news, 'source-tweets',
                                                          f'{news}.json')
                        try:
                            source_tweets_data = read_json_file(source_tweets_path)
                            if source_tweets_data:
                                json_content["source"]["t"] = source_tweets_data["created_at"]
                                json_content["source"]["id"] = source_tweets_data["id"]
                                json_content["source"]["parent"] = source_tweets_data["in_reply_to_status_id"]
                                json_content["source"]["text"] = source_tweets_data["text"]

                        except Exception as e:
                            print(f"Error reading source tweet {source_tweets_path}: {e}")
                            continue

                        reactions_path = os.path.join(event_path, 'non-rumours', news, 'reactions')
                        node_index = {}
                        node_index[int(news)] = 0
                        json_content["source"]["id"] = 0
                        if os.path.exists(reactions_path):

                            for comment in os.listdir(reactions_path):
                                if not comment.startswith('._') and comment != '.DS_Store':
                                    comment_path = os.path.join(reactions_path, comment)
                                    try:
                                        comment_data = read_json_file(comment_path)
                                        if comment_data:
                                            if int(comment_data["id"]) not in node_index:
                                                node_index[int(comment_data["id"])] = len(node_index)
                                    except Exception as e:
                                        print(f"Error reading comment {comment_path}: {e}")
                                        continue
                            for comment in os.listdir(reactions_path):
                                if not comment.startswith('._') and comment != '.DS_Store':
                                    comment_path = os.path.join(reactions_path, comment)
                                    try:
                                        comment_data = read_json_file(comment_path)
                                        if comment_data:
                                            content = {
                                                "t": comment_data["created_at"],
                                                "id": node_index[int(comment_data["id"])],
                                                "text": comment_data["text"],
                                                # "parent": node_index[int(comment_data["in_reply_to_status_id"])]
                                            }
                                            if comment_data["in_reply_to_status_id"] == None:
                                                content["parent"] = 0
                                            elif int(comment_data["in_reply_to_status_id"]) in node_index.keys():
                                                content["parent"] = node_index[
                                                    int(comment_data["in_reply_to_status_id"])]
                                            else:
                                                content["parent"] = 0
                                            json_content["comment"].append(content)
                                    except Exception as e:
                                        print(f"Error reading comment {comment_path}: {e}")
                                        continue
                        row = []
                        col = []
                        edge_attr = []
                        texts = [json_content["source"]["text"]]
                        # 定义原始时间格式
                        original_format = "%a %b %d %H:%M:%S %z %Y"

                        # 解析为 datetime 对象
                        parsed_time = datetime.strptime(json_content["source"]["t"], original_format)

                        # 转换为目标格式
                        init_time = parsed_time.strftime("%a %b %d %H:%M:%S +0000 %Y")
                        for post in json_content["comment"]:
                            row.append(post["parent"])
                            col.append(post["id"])

                            parsed_time = datetime.strptime(post["t"], original_format)

                            # 转换为目标格式
                            post_time = parsed_time.strftime("%a %b %d %H:%M:%S +0000 %Y")
                            edge_time = trans_time(post_time, init_time)
                            edge_attr.append(edge_time)

                            if post['text'] == "":
                                txt = "Relay"
                                texts += [txt]
                            else:
                                texts += [post["text"]]  # 添加非空评论
                        x = text_to_vector(texts, self.model, self.tokenizer)
                        lang = torch.tensor([0])
                        edge_index = torch.LongTensor([row, col])
                        one_data = Data(x=x, y=torch.LongTensor([0]), edge_index=edge_index,
                                        edge_attr=torch.FloatTensor(edge_attr), lang=lang)
                        if one_data.num_nodes > 1:
                            pc = pagerank_centrality(one_data)

                        centrality = torch.tensor(pc, dtype=torch.float32)
                        one_data["centrality"] = centrality
                        print(one_data)
                        data_list.append(one_data)
            # Process rumor data
            rumor_path = os.path.join(event_path, 'rumours')
            if os.path.exists(rumor_path):

                # Iterate through each news item in the event directory
                for news in os.listdir(rumor_path):
                    # if news == "580320890086383616":
                    #     print("DDDD")
                    print(news)

                    json_content = {"source": {}, "comment": []}
                    if not news.startswith('._') and news != '.DS_Store':
                        if len(os.listdir(os.path.join(event_path, 'rumours', news, 'reactions'))) == 0:
                            continue
                        source_tweets_path = os.path.join(event_path, 'rumours', news, 'source-tweets',
                                                          f'{news}.json')
                        try:
                            source_tweets_data = read_json_file(source_tweets_path)
                            if source_tweets_data:
                                json_content["source"]["t"] = source_tweets_data["created_at"]
                                json_content["source"]["id"] = source_tweets_data["id"]
                                json_content["source"]["parent"] = source_tweets_data[
                                    "in_reply_to_status_id"]
                                json_content["source"]["text"] = source_tweets_data["text"]


                        except Exception as e:
                            print(f"Error reading source tweet {source_tweets_path}: {e}")
                            continue

                        reactions_path = os.path.join(event_path, 'rumours', news, 'reactions')
                        node_index = {}
                        node_index[int(news)] = 0
                        json_content["source"]["id"] = 0
                        if os.path.exists(reactions_path):

                            for comment in os.listdir(reactions_path):
                                if not comment.startswith('._') and comment != '.DS_Store':
                                    comment_path = os.path.join(reactions_path, comment)
                                    try:
                                        comment_data = read_json_file(comment_path)
                                        if comment_data:
                                            if int(comment_data["id"]) not in node_index:
                                                node_index[int(comment_data["id"])] = len(node_index)
                                    except Exception as e:
                                        print(f"Error reading comment {comment_path}: {e}")
                                        continue
                            for comment in os.listdir(reactions_path):
                                if not comment.startswith('._') and comment != '.DS_Store':
                                    comment_path = os.path.join(reactions_path, comment)
                                    try:
                                        comment_data = read_json_file(comment_path)
                                        if comment_data:
                                            content = {
                                                "t": comment_data["created_at"],
                                                "id": node_index[int(comment_data["id"])],
                                                "text": comment_data["text"],
                                                # "parent": node_index[
                                                #     int(comment_data["in_reply_to_status_id"])]
                                            }
                                            if comment_data["in_reply_to_status_id"] == None:
                                                content["parent"] = 0
                                            elif int(comment_data["in_reply_to_status_id"]) in node_index.keys():
                                                content["parent"] = node_index[
                                                    int(comment_data["in_reply_to_status_id"])]
                                            else:
                                                content["parent"] = 0
                                            json_content["comment"].append(content)
                                    except Exception as e:
                                        print(f"Error reading comment {comment_path}: {e}")
                                        continue
                        row = []
                        col = []
                        edge_attr = []
                        texts = [json_content["source"]["text"]]
                        # 定义原始时间格式
                        original_format = "%a %b %d %H:%M:%S %z %Y"

                        # 解析为 datetime 对象
                        parsed_time = datetime.strptime(json_content["source"]["t"], original_format)

                        # 转换为目标格式
                        init_time = parsed_time.strftime("%a %b %d %H:%M:%S +0000 %Y")
                        for post in json_content["comment"]:
                            row.append(post["parent"])
                            col.append(post["id"])

                            parsed_time = datetime.strptime(post["t"], original_format)

                            # 转换为目标格式
                            post_time = parsed_time.strftime("%a %b %d %H:%M:%S +0000 %Y")
                            edge_time = trans_time(post_time, init_time)
                            edge_attr.append(edge_time)

                            if post['text'] == "":
                                txt = "Relay"
                                texts += [txt]
                            else:
                                texts += [post["text"]]  # 添加非空评论
                        x = text_to_vector(texts, self.model, self.tokenizer)
                        lang = torch.tensor([0])
                        edge_index = torch.LongTensor([row, col])
                        one_data = Data(x=x, y=torch.LongTensor([1]), edge_index=edge_index,
                                        edge_attr=torch.FloatTensor(edge_attr), lang=lang)
                        if one_data.num_nodes > 1:
                            pc = pagerank_centrality(one_data)

                        centrality = torch.tensor(pc, dtype=torch.float32)
                        one_data["centrality"] = centrality
                        print(one_data)
                        data_list.append(one_data)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]
        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]
        all_data, slices = self.collate(data_list)
        torch.save((all_data, slices), self.processed_paths[0])


class TreeDataset_UPFD(InMemoryDataset):
    def __init__(self, root, centrality_metric="Pagerank", undirected=False, transform=None, pre_transform=None,
                 pre_filter=None):
        self.undirected = undirected
        self.centrality_metric = centrality_metric
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return os.listdir(self.raw_dir)

    @property
    def processed_file_names(self):
        return ['data.pt']

    def download(self):
        pass

    def process(self):

        x = sp.load_npz(os.path.join(self.raw_dir, 'new_bert_feature.npz'))
        x = torch.from_numpy(x.todense()).to(torch.float)

        edge_index = read_txt_array(os.path.join(self.raw_dir, 'A.txt'), sep=',',
                                    dtype=torch.long).t()
        edge_index = coalesce(edge_index, num_nodes=x.size(0))

        y = np.load(os.path.join(self.raw_dir, 'graph_labels.npy'))
        y = torch.from_numpy(y).to(torch.long)
        _, y = y.unique(sorted=True, return_inverse=True)

        batch = np.load(os.path.join(self.raw_dir, 'node_graph_id.npy'))
        batch = torch.from_numpy(batch).to(torch.long)

        # Create individual graphs as Data objects
        data_list = []
        for graph_id in batch.unique():
            node_mask = batch == graph_id
            edge_mask = node_mask[edge_index[0]] & node_mask[edge_index[1]]

            graph_x = x[node_mask]
            graph_edge_index = edge_index[:, edge_mask] - node_mask.nonzero(as_tuple=False)[0].min()
            graph_y = y[graph_id]

            data = Data(x=graph_x, edge_index=graph_edge_index, y=graph_y)
            if data.num_nodes > 1:
                pc = pagerank_centrality(data)

            centrality = torch.tensor(pc, dtype=torch.float32)
            data["centrality"] = centrality
            data["lang"] = torch.tensor([0])
            data.edge_attr = torch.ones(graph_edge_index.size(1), dtype=torch.float)
            data_list.append(data)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]
        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]
        all_data, slices = self.collate(data_list)
        torch.save((all_data, slices), self.processed_paths[0])


class HugeDataset(Dataset):
    def __init__(self, root, centrality_metric="Pagerank", undirected=False, transform=None, pre_transform=None,
                 pre_filter=None):
        self.undirected = undirected
        self.centrality_metric = centrality_metric
        # self.tokenizer = XLMRobertaTokenizer.from_pretrained('xlm-roberta-base', cache_dir="./")
        # self.model = XLMRobertaModel.from_pretrained('xlm-roberta-base', cache_dir="./")
        super().__init__(root, transform, pre_transform, pre_filter)

    @property
    def raw_file_names(self):
        """List all raw files in the raw directory."""
        return os.listdir(self.raw_dir)

    @property
    def processed_file_names(self):
        """Define processed file names based on dataset size."""
        return [f'data_{i}.pt' for i in range(len(self.raw_file_names))]

    def download(self):
        """Skip download logic as data is assumed to be locally available."""
        pass

    def process(self):
        """Process raw files into individual PyG Data objects."""
        for i, raw_file in enumerate(self.raw_file_names):
            print(f"Processing file: {raw_file}")
            post = json.load(open(os.path.join(self.raw_dir, raw_file), 'r', encoding='utf-8'))
            data = self.process_single_file(post)

            if self.pre_filter is not None and not self.pre_filter(data):
                continue

            if self.pre_transform is not None:
                data = self.pre_transform(data)

            # Save each processed graph individually
            torch.save(data, os.path.join(self.processed_dir, f'data_{i}.pt'))

    def len(self):
        """Return the number of processed graphs."""
        return len(self.processed_file_names)

    def get(self, idx):
        """Load a single graph from processed files."""
        data_path = os.path.join(self.processed_dir, f'data_{idx}.pt')
        return torch.load(data_path)

    def process_single_file(self, post):
        """Process a single JSON file into a PyG Data object."""
        try:
            # Extract label
            y = [post['source'].get('label', -1)]

            # Extract node texts
            texts = [post["source"]["content"]]
            row, col, edges = [], [], []
            for i, comment in enumerate(post['comment']):
                row.append(comment['parent'] + 1)
                col.append(comment['comment id'] + 1)
                edges.append(self.calculate_edge_time(comment, post))

                if comment['content']:
                    texts.append(comment['content'])
                else:
                    texts.append('回复')

            # Create edge index tensors
            edge_index = torch.LongTensor([row, col])
            if self.undirected:
                edge_index = to_undirected(edge_index)

            # Generate node features
            x = self.create_node_features(texts)

            # Extract centrality
            # centrality = torch.tensor(post['centrality'].get(self.centrality_metric, []), dtype=torch.float32)

            # Language type
            lang = torch.tensor([1 if "Weibo" in self.root else 0])
            data = Data(
                x=x,
                y=torch.LongTensor(y),
                edge_index=edge_index,
                edge_attr=torch.FloatTensor(edges),
                lang=lang
            )
            if data.num_nodes > 1:
                pc = pagerank_centrality(data)

            centrality = torch.tensor(pc, dtype=torch.float32)
            data["centrality"] = centrality
            # Create the Data object
            return data
        except Exception as e:
            print(f"Error processing file: {e}")
            return None

    def calculate_edge_time(self, comment, post):
        """Calculate edge time or assign a default value."""
        try:
            original_format = "%y-%m-%d %H:%M"
            source_time = datetime.strptime(post["source"]["time"], original_format)
            comment_time = datetime.strptime(comment["time"], original_format)
            return (comment_time - source_time).total_seconds()
        except:
            return 1.0

    def create_node_features(self, texts):
        """Convert texts into node feature tensors."""
        return text_to_vector(texts, self.model, self.tokenizer)


class CovidDataset(Dataset):
    def __init__(self, root, transform=None, pre_transform=None):
        """
        Custom dataset class for graph data.
        :param root: Root directory containing .npz files.
        :param transform: Transformation function for the data.
        :param pre_transform: Preprocessing function for the data.
        """
        super().__init__(root, transform, pre_transform)
        self.data_dir = os.path.join(root)  # Directory for dataset files

    @property
    def raw_file_names(self):
        """
        Returns a list of all .npz files in the directory.
        """
        return [f for f in os.listdir(self.data_dir) if f.endswith('.npz')]

    def len(self):
        """
        Returns the number of samples in the dataset.
        """
        return len(self.raw_file_names)

    def get(self, idx):
        """
        Loads a single data sample by index.
        :param idx: Index of the sample.
        :return: A torch_geometric.data.Data object.
        """
        file_name = self.raw_file_names[idx]  # Get the file name
        file_path = os.path.join(self.data_dir, file_name)  # Construct the full file path

        data = np.load(file_path, allow_pickle=True)

        # Extract features, edges, and labels
        x = torch.stack(data['x'].tolist(), dim=0).float()
        edge_index = torch.tensor(data['edgeindex'], dtype=torch.long)
        y = torch.tensor([int(data['y'])], dtype=torch.long)

        # Create a torch_geometric.data.Data object
        data = Data(x=x, edge_index=edge_index, y=y)

        # Compute PageRank centrality if the graph has more than one node
        if data.num_nodes > 1:
            pc = pagerank_centrality(data)

        # Add centrality and language attributes to the data object
        centrality = torch.tensor(pc, dtype=torch.float32)
        data["centrality"] = centrality
        lang = torch.tensor([1 if "Weibo" in self.root else 0])
        data["lang"] = lang
        data.edge_attr = torch.ones(data.edge_index.size(1), dtype=torch.float)

        return data


def load_datasets_with_prompts(args):
    """
    Load all datasets, dynamically exclude the target dataset based on args.dataset,
    and assign corresponding prompt keys to each DataLoader.

    Args:
        args: Argument parser containing the target dataset name and batch_size.

    Returns:
        train_loaders: List of DataLoaders for training datasets, each with a `prompt_key` attribute.
        target_loader: DataLoader for the target dataset.
    """
    # Define all datasets and their corresponding paths
    dataset_mapping = {
        "DRWeiboV3": {"dataset": TreeDataset("../ACL/Data/DRWeiboV3/"), "prompt_key": "DRWeiboV3_prompt"},
        "Weibo": {"dataset": TreeDataset("../ACL/Data/Weibo/"), "prompt_key": "Weibo_prompt"},
        "WeiboCOVID19": {"dataset": CovidDataset("../ACL/Data/Weibo-COVID19/Weibograph"), "prompt_key": "W19_prompt"},
        "PHEME": {"dataset": TreeDataset_PHEME("../ACL/Data/pheme/"), "prompt_key": "PHEME_prompt"},
        "Politifact": {"dataset": TreeDataset_UPFD("../ACL/Data/politifact/"), "prompt_key": "Politifact_prompt"},
        "Gossipcop": {"dataset": TreeDataset_UPFD("../ACL/Data/gossipcop/"), "prompt_key": "Gossipcop_prompt"},
        "TwitterCOVID19": {"dataset": CovidDataset("../ACL/Data/Twitter-COVID19/Twittergraph"), "prompt_key": "T19_prompt"},
        # "Twitter15-tfidf": {"dataset": TreeDataset("../ACL/Data/Twitter15-tfidf/"), "prompt_key": "en_prompt"}
    }

    # Check if the target dataset exists
    if args.dataset not in dataset_mapping:
        raise ValueError(f"Dataset '{args.dataset}' not found in the available datasets.")

    # Split the datasets into training and target (test)
    target_info = dataset_mapping[args.dataset]
    train_datasets = {key: info for key, info in dataset_mapping.items() if key != args.dataset}

    # Create DataLoaders
    target_loader = DataLoader(target_info["dataset"], batch_size=args.batch_size, shuffle=False)
    target_loader.prompt_key = target_info["prompt_key"]  # Attach the prompt_key for the target dataset

    train_loaders = []
    for key, info in train_datasets.items():
        loader = DataLoader(info["dataset"], batch_size=args.batch_size, shuffle=True)
        loader.prompt_key = info["prompt_key"]  # Attach the prompt_key for each training dataset
        loader.name = key
        train_loaders.append(loader)

    return train_loaders, target_loader


def preprocess_data(raw_dir, cache_file):
    """
    Preprocess raw data and save it as a pickle file.

    Args:
    raw_dir (str): Path to the raw data directory.
    model: The model used to convert text to vectors.
    tokenizer: The tokenizer used for text tokenization.
    trans_time: Function to convert time format.
    cache_file (str): Path to save the processed data as a pickle file.

    Returns:
    None
    """
    tokenizer = XLMRobertaTokenizer.from_pretrained('xlm-roberta-base', cache_dir="./")
    model = XLMRobertaModel.from_pretrained('xlm-roberta-base', cache_dir="./")

    datas = []
    text = []
    ys = []

    # Iterate through all tweets in the raw directory
    for tweet in os.listdir(raw_dir):
        print(f"Processing tweet: {tweet}")
        post = json.load(open(os.path.join(raw_dir, tweet), "r", encoding="utf-8"))
        row = []
        col = []
        edges = []
        combined_text = post["source"]["content"]
        texts = [post["source"]["content"]]
        label = post["source"]["label"]
        ys.append(label)

        # Handle the timestamp if available
        if "time" in post["source"]:
            original_format = "%y-%m-%d %H:%M"
            parsed_time = datetime.strptime(post["source"]["time"], original_format)
            init_time = parsed_time.strftime("%a %b %d %H:%M:%S +0000 %Y")

        # Process comments and related information
        for i, comment in enumerate(post['comment']):
            if "time" in post["source"]:
                parsed_time = datetime.strptime(comment["time"], original_format)
                post_time = parsed_time.strftime("%a %b %d %H:%M:%S +0000 %Y")
                edge_time = trans_time(post_time, init_time)
                edges.append(edge_time)
            else:
                edges.append(1.0)
            if comment['content'] == "":
                txt = "转发"  # "Forward" for empty comments
                texts += [txt]
            else:
                texts += [comment["content"]]

            row.append(comment['parent'] + 1)
            col.append(comment['comment id'] + 1)

        # Construct graph data
        edge_index = [row, col]
        y = torch.LongTensor(label)
        edge_index = torch.LongTensor(edge_index)
        x = text_to_vector(texts, model, tokenizer)

        # Create a Data object to store the graph data
        one_data = Data(x=x, y=y, edge_index=edge_index, edge_attr=torch.FloatTensor(edges))
        datas.append(one_data)
        text.append(combined_text)

    # Save processed data to a pickle file
    with open(cache_file, "wb") as f:
        pickle.dump({
            "graph": datas,
            "texts": text,
            "labels": ys
        }, f)

    print(f"Data has been saved to {cache_file}")


if __name__ == '__main__':
    # data = HugeDataset("./Data/UWeibo/")
    # data = TreeDataset_PHEME("./Data/pheme/")
    # data = UPFD("./Data/", "gossipcop", "bert")
    # data = TreeDataset_UPFD('./Data/politifact/')
    # root_path = "D://ACLR4RUMOR_datasets//Twitter-COVID19//Twitter-COVID19//Twittergraph"
    # data = CovidDataset('./Data/Twitter-COVID19/Twittergraph')
    data = TreeDataset("./Data/Weibo/")
    print(data[0])
