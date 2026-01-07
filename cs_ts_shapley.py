import numpy as np
from random import shuffle, seed, randint, sample, choice
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KDTree, KNeighborsClassifier


def class_conditional_sampling(Y, label_set):
    Idx_nonlabel = []
    for label in label_set:
        label_indices = list(np.where(Y == label)[0])
        s = randint(1, len(label_indices))
        Idx_nonlabel += sample(label_indices, s)
    shuffle(Idx_nonlabel) # shuffle the sampled indices
    # print('len(Idx_nonlabel) = {}'.format(len(Idx_nonlabel)))
    return Idx_nonlabel


def cs_ts_shapley(trnX, trnY, devX, devY, label, clf, T=200,
                   epsilon=1e-4, normalized_score=True, resample=1,
                   trustscore_k=10, trustscore_alpha=0.0, trustscore_filtering="none"):

    # 初始化参数
    k = trustscore_k
    alpha = trustscore_alpha
    filtering = trustscore_filtering
    min_dist = 1e-12
    
    n_labels = np.max(trnY) + 1
    kdtrees = [None] * n_labels
    
    # 初始化Shapley值和迭代计数
    total_N = trnX.shape[0]  # 总训练样本数量
    M = devX.shape[0]  # 验证样本数量
    
    # Select data based on the class label
    orig_indices = np.array(list(range(total_N)))[trnY == label]
    print("The number of training data with label {} is {}".format(label, len(orig_indices)))
    trnX_label = trnX[trnY == label]
    trnY_label = trnY[trnY == label]
    trnX_nonlabel = trnX[trnY != label]
    trnY_nonlabel = trnY[trnY != label]
    devX_label = devX[devY == label]
    devY_label = devY[devY == label]
    devX_nonlabel = devX[devY != label]
    devY_nonlabel = devY[devY != label]
    N_nonlabel = trnX_nonlabel.shape[0]
    nonlabel_set = list(set(trnY_nonlabel))
    print("Labels on the other side: {}".format(nonlabel_set))
    
    # 初始化当前标签样本的Shapley值向量
    N = trnX_label.shape[0]
    val = np.zeros(N)
    # 初始化记录每轮迭代val_t值的数组
    val_T = np.zeros((T, N))
    
    # 实现不同的过滤方法
    if filtering == "uncertainty":
        neigh = KNeighborsClassifier(n_neighbors=k)
        neigh.fit(trnX, trnY)
        confidence = neigh.predict_proba(trnX)
        cutoff = np.percentile(confidence, alpha * 100)
        unfiltered_idxs = np.where(confidence >= cutoff)[0]
        X_filtered = trnX[unfiltered_idxs]
        y_filtered = trnY[unfiltered_idxs]
    
    # 根据过滤方法获取指定类别的数据
    def apply_filtering(X, label):
        if filtering == "none":
            return X[trnY == label]
        elif filtering == "density":
            label_mask = trnY == label
            kdtree = KDTree(X[label_mask])
            knn_radii = kdtree.query(X[label_mask], k=k)[0][:, -1]
            eps = np.percentile(knn_radii, (1 - alpha) * 100)
            return X[label_mask & (knn_radii <= eps)]
        elif filtering == "uncertainty":
            return X_filtered[y_filtered == label]
        return X[trnY == label]
    
    # 构建每个类别的KDTree，使用欧氏距离
    for label_idx in range(n_labels):
        X_to_use = apply_filtering(trnX, label_idx)
        
        # 使用KDTree和欧氏距离
        kdtrees[label_idx] = KDTree(X_to_use)
        if len(X_to_use) == 0:
            print("Filtered too much or missing examples from a label! Please lower alpha or check data.")
    
    # 定义TrustScore计算函数
    def compute_trustscore(X, y_pred):
        # 初始化距离矩阵
        d = np.zeros((X.shape[0], n_labels))
        for label_idx in range(n_labels):
            # 查询第一个最近邻的距离（使用欧氏距离）
            d[:, label_idx] = kdtrees[label_idx].query(X, k=1)[0][:, 0]
        
        sorted_d = np.sort(d, axis=1)
        d_to_pred = d[np.arange(d.shape[0]), y_pred]
        # 找到不是预测类的最近距离
        d_to_closest_not_pred = np.where(
            sorted_d[:, 0] != d_to_pred, sorted_d[:, 0], sorted_d[:, 1]
        )
        
        # 计算原始TrustScore
        ts_values = d_to_closest_not_pred / (d_to_pred + min_dist)
        
        # 对TrustScore值进行平滑处理
        # 1. 归一化到[0, 1]范围
        min_ts = np.min(ts_values)
        max_ts = np.max(ts_values)
        if max_ts - min_ts > min_dist:  # 避免除以零
            ts_values = (ts_values - min_ts) / (max_ts - min_ts)
        
        # 2. 使用指数平滑进一步减少噪声
        ts_values = np.exp(ts_values) / np.exp(1.0)  # 平滑到[1/e, 1]范围
        
        return ts_values

    # Create indices and shuffle them
    Idx = list(range(N))
    
    M = len(devY)
    # Shapley values, number of permutations, total number of iterations
    val, k = np.zeros((N)), 0
    for t in tqdm(range(1, T+1)):
        # print("t = {}".format(t))
        # Shuffle the data
        shuffle(Idx)
        # For each permutation, resample I times from the other classes
        for i in range(resample):
            k += 1
            # value container for iteration i
            val_i = np.zeros((N+1))
            val_i_non = np.zeros((N+1))

            # --------------------
            # Sample a subset of training data from other labels for each i
            if len(nonlabel_set) == 1:
                s = randint(1, N_nonlabel)
                # print('s = {}'.format(s))
                Idx_nonlabel = sample(list(range(N_nonlabel)), s)
            else:
                Idx_nonlabel = class_conditional_sampling(trnY_nonlabel, nonlabel_set)
            trnX_nonlabel_i = trnX_nonlabel[Idx_nonlabel, :]
            trnY_nonlabel_i = trnY_nonlabel[Idx_nonlabel]

            # --------------------
            # With no data from the target class and the sampled data from other classes
            val_i[0] = 0.0
            # 无数据情况特殊处理，使用默认trustscore
            val_i_non[0] = 0.0  # 无数据情况的默认值
            
            # --------------------- 
            # With all data from the target class and the sampled data from other classes
            tempX = np.concatenate((trnX_nonlabel_i, trnX_label))
            tempY = np.concatenate((trnY_nonlabel_i, trnY_label))
            clf.fit(tempX, tempY)
            val_i[N] = accuracy_score(devY_label, clf.predict(devX_label), normalize=False) / M
            # 对训练集预测用于计算trustscore
            trn_y_pred = clf.predict(trnX_label)
            # 为所有训练样本计算trustscore
            ts_values = compute_trustscore(trnX_label, trn_y_pred)
            
            # 标准化TrustScore值
            if ts_values.sum() != 0:
                ts_values = ts_values / ts_values.sum()
            
            # 使用所有训练样本的trustscore值，按照打乱后的索引顺序
            for i in range(N):
                original_idx = Idx[i]  # 获取当前样本在原始trnX_label中的索引
                val_i_non[i+1] = ts_values[original_idx]  # 每个样本对应自己的trustscore值
            
            # --------------------
            # 
            for j in range(1,N+1):
                if abs(val_i[N] - val_i[j-1]) < epsilon:
                    val_i[j] = val_i[j-1]
                    val_i_non[j] = val_i_non[j-1]  
                else:
                    # Extract the first $j$ data points
                    trnX_j = trnX_label[Idx[:j],:]
                    trnY_j = trnY_label[Idx[:j]]
                    try:
                        # ---------------------------------
                        tempX = np.concatenate((trnX_nonlabel_i, trnX_j))
                        tempY = np.concatenate((trnY_nonlabel_i, trnY_j))
                        clf.fit(tempX, tempY)
                        val_i[j] = accuracy_score(devY_label, clf.predict(devX_label), normalize=False) / M
                        # 直接使用之前计算好的trustscore值，不需要重新计算
                        val_i_non[j] = val_i_non[j]  # 保持当前样本的trustscore值不变
                    except ValueError:
                        # 处理单一类别情况
                        # 对目标类别验证集预测用于计算准确率
                        y_pred_single_label = [trnY_j[0]] * len(devX_label)
                        val_i[j] = accuracy_score(devY_label, y_pred_single_label, normalize=False) / M
                        # 直接使用之前计算好的trustscore值，不需要重新计算
                        val_i_non[j] = val_i_non[j]  # 保持当前样本的trustscore值不变
            # ==========================================
            # New implementation
            wvalues = np.exp(val_i_non) * val_i
            # print("wvalues = {}".format(wvalues))
            diff = wvalues[1:] - wvalues[:N]
            val[Idx] = ((1.0*(k-1)/k))*val[Idx] + (1.0/k)*(diff)
    
    # 标准化Shapley值
    if normalized_score:
        val = val / val.sum() if val.sum() != 0 else val
        clf.fit(trnX, trnY)
        score = accuracy_score(devY, clf.predict(devX), normalize=False) / M
        val = val * score
    return val, orig_indices
