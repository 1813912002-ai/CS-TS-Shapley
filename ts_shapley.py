import numpy as np
from random import shuffle
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KDTree, KNeighborsClassifier

def ts_shapley(trnX, trnY, devX, devY, clf, T=200,
                   epsilon=1e-4, normalized_score=True,
                   trustscore_k=10, trustscore_alpha=0.0, trustscore_filtering="none"):
    '''
    不区分类别的CS-TS-Shapley实现，使用TrustScore评估数据价值
    
    Args:
        trnX: 训练数据特征
        trnY: 训练数据标签
        devX: 验证数据特征
        devY: 验证数据标签
        clf: 分类器
        T: 采样轮数
        epsilon: 收敛阈值
        normalized_score: 是否归一化Shapley值
        trustscore_k: TrustScore的k近邻参数
        trustscore_alpha: TrustScore的过滤参数
        trustscore_filtering: TrustScore的过滤方法，可选值："none", "density", "uncertainty"
    
    Returns:
        val: 每个训练样本的Shapley值
        ts_values_list: 每轮迭代中每个验证样本的trustscore值列表
    '''
    # 初始化参数
    k = trustscore_k
    alpha = trustscore_alpha
    filtering = trustscore_filtering
    min_dist = 1e-12
    
    n_labels = np.max(trnY) + 1
    kdtrees = [None] * n_labels
    
    # 初始化Shapley值和迭代计数
    N = trnX.shape[0]  # 训练样本数量
    M = devX.shape[0]  # 验证样本数量
    val = np.zeros(N)  # 初始化Shapley值向量
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
    for label in range(n_labels):
        X_to_use = apply_filtering(trnX, label)
        kdtrees[label] = KDTree(X_to_use)
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

    # 初始化主要变量
    N = trnX.shape[0]
    M = len(devY)
    Idx = list(range(N))
    val = np.zeros(N)

    
    # 主循环计算Shapley值
    for t in tqdm(range(1, T+1)):
        shuffle(Idx)
        
        # 初始化价值容器
        val_i = np.zeros(N+1)
        val_i_non = np.zeros(N+1)

        # 无数据情况初始化
        val_i[0] = 0.0
        # 随机预测 - 对于j=0（无数据情况），所有样本使用默认trustscore
        val_i_non[0] = 0.0  # 无数据情况的默认值
        
        # 使用全部训练数据 - 对于j=N（全部数据情况）
        clf.fit(trnX, trnY)
        # 对验证集预测用于计算准确率
        y_pred = clf.predict(devX)
        val_i[N] = accuracy_score(devY, y_pred, normalize=False) / M
        # 对训练集预测用于计算trustscore
        trn_y_pred = clf.predict(trnX)
        # 为所有训练样本计算trustscore
        ts_values = compute_trustscore(trnX, trn_y_pred)
        
        # 标准化TrustScore值
        if ts_values.sum() != 0:
            ts_values = ts_values / ts_values.sum()
        
        # 使用所有训练样本的trustscore值，按照打乱后的索引顺序
        for i in range(N):
            sample_idx = Idx[i]  # 获取当前样本在原始训练集中的索引
            val_i_non[i+1] = ts_values[sample_idx]  # 每个样本对应自己的trustscore值
        
        # 计算每个子集的价值
        for j in range(1, N+1):
            if abs(val_i[N] - val_i[j-1]) < epsilon:
                val_i[j] = val_i[j-1]
                val_i_non[j] = val_i_non[j-1]  
            else:
                # 提取前j个数据点
                trnX_j = trnX[Idx[:j],:]
                trnY_j = trnY[Idx[:j]]
                try:
                    # 训练模型并评估
                    clf.fit(trnX_j, trnY_j)
                    # 对验证集预测用于计算准确率
                    y_pred = clf.predict(devX)
                    val_i[j] = accuracy_score(devY, y_pred, normalize=False) / M
                    # 直接使用之前计算好的trustscore值，不需要重新计算
                    val_i_non[j] = val_i_non[j]  # 保持当前样本的trustscore值不变
                except ValueError:
                    # 处理单一类别情况
                    # 对验证集预测用于计算准确率
                    y_pred_single = [trnY_j[0]] * M
                    val_i[j] = accuracy_score(devY, y_pred_single, normalize=False) / M
                    # 直接使用之前计算好的trustscore值，不需要重新计算
                    val_i_non[j] = val_i_non[j]  # 保持当前样本的trustscore值不变
        

        
        # 更新Shapley值
        wvalues = np.exp(val_i_non) * val_i
        diff = wvalues[1:] - wvalues[:N]
        val[Idx] = ((t-1)/t) * val[Idx] + (1/t) * diff
        # 记录每轮迭代的val_t值，类似于TMC-Shapley的实现
        val_T[t-1,:] = (np.array(val_i)[1:])[Idx]
    
    # 标准化Shapley值
    if normalized_score:
        val = val / val.sum() if val.sum() != 0 else val
        clf.fit(trnX, trnY)
        score = accuracy_score(devY, clf.predict(devX), normalize=False) / M
        val = val * score
    return val