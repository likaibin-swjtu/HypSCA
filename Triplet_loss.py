# Triplet_loss.py
import tensorflow as tf
import sys
import numpy as np

def pairwise_distance(feature, labels, num_classes, alpha_value, squared=False):
    pairwise_distances_squared = (
        tf.math.add(
            tf.math.reduce_sum(tf.math.square(feature), axis=[1], keepdims=True),   #tf.math.reduce_sum计算张量各维度上元素的总和
            tf.math.reduce_sum(tf.math.square(tf.transpose(feature)), axis=[0], keepdims=True),
        )
        - 2.0 * tf.matmul(feature, tf.transpose(feature))
    )
    #标签距离
    lable_pairwise_distances_squared = (
        tf.math.add(
            tf.math.square(labels),           #标签平方(bs, 1)
            tf.math.square(tf.transpose(labels))         #转置后平方(1, bs)
        )
        - 2.0 * tf.matmul(labels, tf.transpose(labels)))     #转置后平方(bs, bs)

    pairwise_distances_squared = pairwise_distances_squared / tf.pow(alpha_value, tf.math.sqrt(lable_pairwise_distances_squared)/num_classes)

    # Deal with numerical inaccuracies. Set small negatives to zero.处理数字不准确的问题。 将小负数设置为零。
    pairwise_distances_squared = tf.math.maximum(pairwise_distances_squared, 0.0)
    # Get the mask where the zero distances are at.获取零距离处的掩码。
    error_mask = tf.math.less_equal(pairwise_distances_squared, 0.0)
    # Optionally take the sqrt.
    if squared:
        pairwise_distances = pairwise_distances_squared
    else:
        pairwise_distances = tf.math.sqrt(
            pairwise_distances_squared
            + tf.cast(error_mask, dtype=tf.dtypes.float32) * 1e-16     #tf.cast(x, dtype, name=None)释义：数据类型转换
        )

    # Undo conditionally adding 1e-16.有条件地撤消添加 1e-16。
    pairwise_distances = tf.math.multiply(
        pairwise_distances,
        tf.cast(tf.math.logical_not(error_mask), dtype=tf.dtypes.float32), #tf.math.logical_notx和y两个张量在相应位置上做非（!）操作
    )

    num_data = tf.shape(feature)[0]      #bs
    # Explicitly set diagonals to zero.显式将对角线设置为零
    mask_offdiagonals = (tf.ones_like(pairwise_distances)#tf.ones_like函数目的是创建一个和输入参数（tensor）维度一样，元素都为1的张量
                         - tf.linalg.diag(tf.ones([num_data])))  #tf.linalg.diag返回具有给定对角线值的对角线张量,tf.ones生成给定形状的全1的tensor张量
    pairwise_distances = tf.math.multiply(pairwise_distances, mask_offdiagonals)   #tf.math.multiply逐个元素相乘
    return pairwise_distances

def pairwise_distance_unlabel(feature, squared=False):
  pairwise_distances_squared = (
          tf.math.add(
              tf.math.reduce_sum(tf.math.square(feature), axis=[1], keepdims=True),  # tf.math.reduce_sum计算张量各维度上元素的总和
              tf.math.reduce_sum(tf.math.square(tf.transpose(feature)), axis=[0], keepdims=True),
          )
          - 2.0 * tf.matmul(feature, tf.transpose(feature))
  )

  # Deal with numerical inaccuracies. Set small negatives to zero.
  pairwise_distances_squared = tf.math.maximum(pairwise_distances_squared, 0.0)
  # Get the mask where the zero distances are at.
  error_mask = tf.math.less_equal(pairwise_distances_squared, 0.0)

  # Optionally take the sqrt.
  if squared:
      pairwise_distances = pairwise_distances_squared
  else:
      pairwise_distances = tf.math.sqrt(
          pairwise_distances_squared
          + tf.cast(error_mask, dtype=tf.dtypes.float64) * 1e-16  # tf.cast(x, dtype, name=None)释义：数据类型转换
      )

  pairwise_distances = tf.math.multiply(
      pairwise_distances,
      tf.cast(tf.math.logical_not(error_mask), dtype=tf.dtypes.float64),  # tf.math.logical_notx和y两个张量在相应位置上做非（!）操作
  )

  num_data = tf.shape(feature)[0]  # bs
  # Explicitly set diagonals to zero.显式将对角线设置为零
  mask_offdiagonals = (tf.ones_like(pairwise_distances)  # tf.ones_like函数目的是创建一个和输入参数（tensor）维度一样，元素都为1的张量
                       - tf.linalg.diag(
              tf.ones([num_data])))  # tf.linalg.diag返回具有给定对角线值的对角线张量,tf.ones生成给定形状的全1的tensor张量
  pairwise_distances = tf.math.multiply(pairwise_distances, mask_offdiagonals)  # tf.math.multiply逐个元素相乘
  return pairwise_distances


def masked_maximum(data, mask, dim=1):
    axis_minimums = tf.math.reduce_min(data, dim, keepdims=True)
    masked_maximums = (
        tf.math.reduce_max(
            tf.math.multiply(data - axis_minimums, mask), dim, keepdims=True
        )
        + axis_minimums
    )
    return masked_maximums


def masked_minimum(data, mask, dim=1):
    axis_maximums = tf.math.reduce_max(data, dim, keepdims=True)
    masked_minimums = (
        tf.math.reduce_min(
            tf.math.multiply(data - axis_maximums, mask), dim, keepdims=True
        )
        + axis_maximums
    )
    return masked_minimums


def triplet_semihard_loss(alpha_value, margin, num_classes):
    def loss(y_true, y_pred):
        labels = tf.convert_to_tensor(y_true, name="labels")
        embeddings = tf.convert_to_tensor(y_pred, name="embeddings")

        # loss = tf.losses.categorical_crossentropy()
        convert_to_float32 = (
            embeddings.dtype == tf.dtypes.float16 or embeddings.dtype == tf.dtypes.bfloat16
        )
        precise_embeddings = (
            tf.cast(embeddings, tf.dtypes.float32) if convert_to_float32 else embeddings
        )

        # Reshape label tensor to [batch_size, 1].
        lshape = tf.shape(labels)
        labels = tf.cast(tf.reshape(labels, [lshape[0], 1]), tf.dtypes.float32)
        pdist_matrix = pairwise_distance(precise_embeddings, labels, num_classes, alpha_value, squared=True)
        adjacency = tf.math.equal(labels, tf.transpose(labels))
        adjacency_not = tf.math.logical_not(adjacency)
        batch_size = tf.size(labels)
        pdist_matrix_tile = tf.tile(pdist_matrix, [batch_size, 1])
        mask = tf.math.logical_and(
            tf.tile(adjacency_not, [batch_size, 1]),
            tf.math.greater(
                pdist_matrix_tile, tf.reshape(tf.transpose(pdist_matrix), [-1, 1])
            ),
        )
        mask_final = tf.reshape(
            tf.math.greater(
                tf.math.reduce_sum(
                    tf.cast(mask, dtype=tf.dtypes.float32), 1, keepdims=True
                ),
                0.0,
            ),
            [batch_size, batch_size],
        )
        mask_final = tf.transpose(mask_final)
        adjacency_not = tf.cast(adjacency_not, dtype=tf.dtypes.float32)
        mask = tf.cast(mask, dtype=tf.dtypes.float32)

        # negatives_outside: smallest D_an where D_an > D_ap.
        negatives_outside = tf.reshape(
            masked_minimum(pdist_matrix_tile, mask), [batch_size, batch_size]
        )
        negatives_outside = tf.transpose(negatives_outside)

        # negatives_inside: largest D_an.
        negatives_inside = tf.tile(
            masked_maximum(pdist_matrix, adjacency_not), [1, batch_size]
        )
        semi_hard_negatives = tf.where(mask_final, negatives_outside, negatives_inside)
        loss_mat = tf.math.add(margin, pdist_matrix - semi_hard_negatives)

        mask_positives = tf.cast(adjacency, dtype=tf.dtypes.float32) - tf.linalg.diag(
            tf.ones([batch_size])
        )

        # In lifted-struct, the authors multiply 0.5 for upper triangular
        #   in semihard, they take all positive pairs except the diagonal.
        num_positives = tf.math.reduce_sum(mask_positives)

        triplet_loss = tf.math.truediv(
            tf.math.reduce_sum(
                tf.math.maximum(tf.math.multiply(loss_mat, mask_positives), 0.0)
            ),
            num_positives,
        )

        if convert_to_float32:
            return tf.cast(triplet_loss, embeddings.dtype)
        else:
            return triplet_loss
    return loss

