# hier_loss.py
import tensorflow as tf
from tensorflow.keras.layers import Layer

class HIERLoss(Layer):
    def __init__(self, nb_proxies, sz_embed, mrg, tau=0.1, hyp_c=0.1, clip_r=2.3, **kwargs):
        super(HIERLoss, self).__init__(**kwargs)
        self.nb_proxies = nb_proxies  # 代理点数量
        self.sz_embed = sz_embed  # 嵌入向量的大小
        self.tau = tau  # Gumbel softmax 的温度参数
        self.hyp_c = hyp_c  # 双曲空间的曲率
        self.mrg = mrg  # HIER 损失的边距参数
        self.clip_r = clip_r  # 初始化代理点的剪辑范围

        # 初始化代理点
        initial_lcas = tf.random.normal((self.nb_proxies, self.sz_embed))
        initial_lcas = initial_lcas / tf.math.sqrt(float(sz_embed)) * clip_r * 0.9
        self.lcas = tf.Variable(initial_lcas, trainable=True)

    def poincare_dist(self, x, y, c):
        """
        计算 Poincare 距离
        """
        sqrt_c = tf.math.sqrt(c)
        norm_x = tf.norm(x, axis=-1, keepdims=True)
        norm_y = tf.norm(y, axis=-1, keepdims=True)

        # 添加一个小常数以避免除以零
        eps = 1e-7
        numerator = sqrt_c * tf.norm(x[:, None, :] - y[None, :, :], axis=-1)
        denominator = 1 - c * norm_x * norm_y
        denominator = tf.where(denominator < eps, eps, denominator)  # 避免除以接近零的值

        dist_c = 2 / sqrt_c * tf.math.atanh(
            tf.clip_by_value(numerator / denominator, -1 + eps, 1 - eps)
        )
        return dist_c

    def dist_matrix(self, x, y):
        """
        计算距离矩阵
        """
        if self.hyp_c > 0:
            return self.poincare_dist(x, y, self.hyp_c)  # 使用 Poincare 距离
        else:
            x2 = tf.reduce_sum(tf.square(x), axis=1, keepdims=True)  # 计算 x 的平方和
            y2 = tf.reduce_sum(tf.square(y), axis=1, keepdims=True)  # 计算 y 的平方和
            xy = tf.matmul(x, y, transpose_b=True)  # 计算 x 和 y 的点积
            dist = x2 + tf.transpose(y2) - 2 * xy  # 计算欧氏距离
            return tf.maximum(dist, 1e-12)  # 确保距离矩阵中没有负值

    def compute_gHHC(self, z_s, lcas, dist_matrix, indices_tuple):
        """
        计算 gHHC 损失
        """
        i, j, k = indices_tuple  # 三元组的索引
        cp_dist = dist_matrix  # 距离矩阵

        # 计算到代理点的距离
        max_dists_ij = tf.maximum(tf.gather(cp_dist, i), tf.gather(cp_dist, j))  # 取两点之间的最大距离
        lca_ij_prob = self.gumbel_softmax(-max_dists_ij / self.tau, temperature=self.tau, hard=True)  # 计算 LCA 概率
        lca_ij_idx = tf.argmax(lca_ij_prob, axis=-1)  # 计算 LCA 的索引

        max_dists_ijk = tf.maximum(tf.gather(cp_dist, k), max_dists_ij)  # 取三点之间的最大距离
        lca_ijk_prob = self.gumbel_softmax(-max_dists_ijk / self.tau, temperature=self.tau, hard=True)  # 计算 LCA 概率
        lca_ijk_idx = tf.argmax(lca_ijk_prob, axis=-1)  # 计算 LCA 的索引

        # 计算到 LCA 的距离
        dist_i_lca_ij = tf.reduce_sum(tf.gather(cp_dist, i) * lca_ij_prob, axis=-1)  # 计算 i 到 LCA 的距离
        dist_i_lca_ijk = tf.reduce_sum(tf.gather(cp_dist, i) * lca_ijk_prob, axis=-1)  # 计算 i 到 LCA 的距离
        dist_j_lca_ij = tf.reduce_sum(tf.gather(cp_dist, j) * lca_ij_prob, axis=-1)  # 计算 j 到 LCA 的距离
        dist_j_lca_ijk = tf.reduce_sum(tf.gather(cp_dist, j) * lca_ijk_prob, axis=-1)  # 计算 j 到 LCA 的距离
        dist_k_lca_ij = tf.reduce_sum(tf.gather(cp_dist, k) * lca_ij_prob, axis=-1)  # 计算 k 到 LCA 的距离
        dist_k_lca_ijk = tf.reduce_sum(tf.gather(cp_dist, k) * lca_ijk_prob, axis=-1)  # 计算 k 到 LCA 的距离

        # 计算 HIER 损失
        hc_loss_1 = tf.keras.activations.relu(dist_i_lca_ij - dist_i_lca_ijk + self.mrg)  # 计算第一个损失
        hc_loss_2 = tf.keras.activations.relu(dist_j_lca_ij - dist_j_lca_ijk + self.mrg)  # 计算第二个损失
        hc_loss_3 = tf.keras.activations.relu(dist_k_lca_ijk - dist_k_lca_ij + self.mrg)  # 计算第三个损失

        hc_loss = hc_loss_1 + hc_loss_2 + hc_loss_3  # 合并损失

        # 添加条件：仅当 lca_ij_idx 和 lca_ijk_idx 不相同时才计算损失
        hc_loss = hc_loss * tf.cast(tf.not_equal(lca_ij_idx, lca_ijk_idx), tf.float32)

        # 确保损失是有限的
        hc_loss = tf.where(tf.math.is_finite(hc_loss), hc_loss, tf.zeros_like(hc_loss))  # 如果有 NaN 或 Inf，将其置为 0

        return tf.reduce_mean(hc_loss)  # 返回平均损失

    def gumbel_softmax(self, logits, temperature, hard=False):
        """
        计算 Gumbel Softmax
        """
        gumbel_noise = -tf.math.log(
            -tf.math.log(tf.random.uniform(tf.shape(logits), minval=0, maxval=1) + 1e-20) + 1e-20)  # 生成 Gumbel 噪声
        logits = (logits + gumbel_noise) / temperature  # 调整 logits
        y_soft = tf.nn.softmax(logits, axis=-1)  # 计算 softmax

        if hard:
            y_hard = tf.cast(tf.equal(y_soft, tf.reduce_max(y_soft, axis=-1, keepdims=True)), y_soft.dtype)  # 生成硬样本
            y_soft = tf.stop_gradient(y_hard - y_soft) + y_soft  # 停止梯度

        return y_soft

    def get_reciprocal_triplets(self, sim_matrix, topk=10, t_per_anchor=30):
        """
        获取相互三元组
        """
        sim_matrix = tf.cast(sim_matrix, tf.float32)  # 确保 sim_matrix 是 float32
        sim_matrix_shape = tf.shape(sim_matrix)

        anchor_idx, positive_idx, negative_idx = [], [], []
        topk_index = tf.argsort(sim_matrix, direction='DESCENDING')[:, :topk]  # 获取 topk 索引

        def gather_topk(i):
            pos_indices = tf.where(sim_matrix[i] == 1.0)  # 获取正例索引
            neg_indices = tf.where(sim_matrix[i] < 1.0)  # 获取负例索引

            pos_indices = tf.cast(pos_indices, tf.int32)
            neg_indices = tf.cast(neg_indices, tf.int32)

            # 确保 pos_indices 和 neg_indices 形状为一维
            pos_indices = tf.reshape(pos_indices, [-1])
            neg_indices = tf.reshape(neg_indices, [-1])

            # 调整 pos_indices 和 neg_indices 的大小为 t_per_anchor
            if tf.size(pos_indices) > t_per_anchor:
                pos_indices = pos_indices[:t_per_anchor]
            else:
                # 添加维度以确保形状一致
                pos_indices = tf.concat([pos_indices, tf.fill([t_per_anchor - tf.size(pos_indices)], -1)], axis=0)

            if tf.size(neg_indices) > t_per_anchor:
                neg_indices = neg_indices[:t_per_anchor]
            else:
                # 添加维度以确保形状一致
                neg_indices = tf.concat([neg_indices, tf.fill([t_per_anchor - tf.size(neg_indices)], -1)], axis=0)

            return (tf.fill([t_per_anchor], i), pos_indices, neg_indices)

        results = tf.map_fn(
            gather_topk,
            tf.range(sim_matrix_shape[0], dtype=tf.int32),
            fn_output_signature=(
                tf.TensorSpec(shape=[t_per_anchor], dtype=tf.int32),
                tf.TensorSpec(shape=[t_per_anchor], dtype=tf.int32),
                tf.TensorSpec(shape=[t_per_anchor], dtype=tf.int32)
            )
        )

        anchor_idx = tf.concat(results[0], axis=0)
        positive_idx = tf.concat(results[1], axis=0)
        negative_idx = tf.concat(results[2], axis=0)
        return anchor_idx, positive_idx, negative_idx

    @tf.function
    def call(self, inputs, y, topk=10):
        """
        前向传播功能
        """
        z_s = inputs  # 输入嵌入
        bs = tf.shape(z_s)[0]  # 批量大小
        lcas = self.lcas  # 代理点
        # tf.print("z_s shape (should be [batch_size, embedding_size]):", tf.shape(z_s))
        # tf.print("lcas shape:", tf.shape(lcas))
        if len(z_s.shape) == 1:
            z_s = tf.reshape(z_s, [bs, self.sz_embed])
        # 确保 y 的形状正确
        # z_s = tf.reshape(z_s, [bs, self.sz_embed])  # Adjust this line if needed based on z_s shape
        y = tf.reshape(y, [bs])

        # 合并嵌入和代理点
        all_nodes = tf.concat([z_s, lcas], axis=0)
        all_dist_matrix = self.dist_matrix(all_nodes, all_nodes)  # 计算组合距离矩阵

        # 计算相似度矩阵
        sim_matrix = tf.exp(-tf.cast(all_dist_matrix[:bs, :bs], tf.float32))

        # 扩展 y 以进行广播
        y_expanded_1 = tf.expand_dims(y, 1)
        y_expanded_0 = tf.expand_dims(y, 0)

        one_hot_mat = tf.cast(tf.equal(y_expanded_1, y_expanded_0), tf.float32)

        # 确保 one_hot_mat 是二维的
        tf.debugging.assert_equal(tf.rank(one_hot_mat), 2, f"one_hot_mat must be 2D, but got shape {tf.shape(one_hot_mat)}")

        # 确保 sim_matrix 和 one_hot_mat 具有相同的形状
        tf.debugging.assert_equal(tf.shape(sim_matrix), tf.shape(one_hot_mat),
                                  f"sim_matrix and one_hot_mat must have the same shape, but got sim_matrix shape {tf.shape(sim_matrix)} and one_hot_mat shape {tf.shape(one_hot_mat)}")

        sim_matrix += one_hot_mat

        # 对称性操作，确保相似度矩阵是对称的
        sim_matrix = (sim_matrix + tf.transpose(sim_matrix)) / 2
        sim_matrix = tf.linalg.set_diag(sim_matrix, tf.fill([bs], -1.0))

        # 获取三元组
        indices_tuple = self.get_reciprocal_triplets(sim_matrix, topk=topk, t_per_anchor=30)
        # ASCAD_fixed 数据集，topk 和 t_per_anchor 的初始值建议分别设置为 10-20 和 30-50
        hier_loss_value = self.compute_gHHC(z_s, lcas, all_dist_matrix[:bs, bs:], indices_tuple)

        return hier_loss_value  # 返回 HIER 损失值

    def get_config(self):
        """
        获取配置
        """
        config = super().get_config()
        config.update({
            "nb_proxies": self.nb_proxies,
            "sz_embed": self.sz_embed,
            "mrg": self.mrg,
            "tau": self.tau,
            "hyp_c": self.hyp_c,
            "clip_r": self.clip_r
        })
        return config

    @classmethod
    def from_config(cls, config):
        """
        从配置创建实例
        """
        return cls(**config)