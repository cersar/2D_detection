import tensorflow as tf
import numpy as np


def decode_box_outputs(box_out, anchors):
    pred_x, pred_y, pred_w, pred_h = tf.unstack(box_out, num=4, axis=-1)
    anchor_x, anchor_y, anchor_w, anchor_h = tf.unstack(anchors, num=4, axis=-1)

    box_w = tf.math.exp(pred_w) * anchor_w
    box_h = tf.math.exp(pred_h) * anchor_h
    box_x = pred_x * anchor_w + anchor_x
    box_y = pred_y * anchor_h + anchor_y
    boxes = tf.stack([box_x, box_y, box_w, box_h], axis=-1)
    return boxes


def xywh_to_x1y1x2y2(bbox):
    c_x, c_y, w, h = tf.unstack(bbox, axis=-1)
    c_x, c_y, w, h = tf.expand_dims(c_x, axis=-1), tf.expand_dims(c_y, axis=-1), \
                     tf.expand_dims(w, axis=-1), tf.expand_dims(h, axis=-1)
    bbox = tf.concat([c_x - w / 2,
                      c_y - h / 2,
                      c_x + w / 2,
                      c_y + h / 2], -1)
    return bbox


def x1y1x2y2_to_xywh(bbox):
    x1, y1, x2, y2 = tf.unstack(bbox, axis=-1)
    x1, y1, x2, y2 = tf.expand_dims(x1, axis=-1), tf.expand_dims(y1, axis=-1), \
                     tf.expand_dims(x2, axis=-1), tf.expand_dims(y2, axis=-1)
    bbox = tf.concat([(x1 + x2) / 2,
                      (y1 + y2) / 2,
                      x2 - x1,
                      y2 - y1], -1)
    return bbox


def get_envelope_box(bbox1, bbox2):
    i_lt = tf.math.minimum(bbox1[..., :2], bbox2[..., :2])
    i_rd = tf.math.maximum(bbox1[..., 2:], bbox2[..., 2:])
    box = tf.concat([i_lt, i_rd], -1)
    return box


def get_intersec_box(bbox1, bbox2):
    i_lt = tf.math.maximum(bbox1[..., :2], bbox2[..., :2])
    i_rd = tf.math.minimum(bbox1[..., 2:], bbox2[..., 2:])
    box = tf.concat([i_lt, i_rd], -1)
    return box


def compute_iou(bbox1, bbox2, iou_type='iou'):
    if iou_type not in ['iou', 'giou', 'diou', 'ciou']:
        raise ValueError('Invalid iou_type {} !'.format(iou_type))
    eps = 1.e-10

    intersec_box = get_intersec_box(bbox1, bbox2)

    i_w = intersec_box[..., 2] - intersec_box[..., 0]
    i_h = intersec_box[..., 3] - intersec_box[..., 1]

    i_w *= tf.cast(i_w > 0, tf.float32)
    i_h *= tf.cast(i_h > 0, tf.float32)

    intersec = tf.abs(i_w * i_h)
    s1 = (bbox1[..., 2] - bbox1[..., 0]) * (bbox1[..., 3] - bbox1[..., 1])
    s2 = (bbox2[..., 2] - bbox2[..., 0]) * (bbox2[..., 3] - bbox2[..., 1])
    union = s1 + s2 - intersec
    iou = intersec / (union + eps)

    if iou_type == 'iou':
        return iou

    C = get_envelope_box(bbox1, bbox2)
    if iou_type == 'giou':
        C_area = (C[..., 2] - C[..., 0]) * (C[..., 3] - C[..., 1])
        iou -= (1 - union / C_area)
        return iou

    bbox1_xywh = x1y1x2y2_to_xywh(bbox1)
    bbox2_xywh = x1y1x2y2_to_xywh(bbox2)
    D_box = tf.norm(bbox1_xywh[..., :2] - bbox2_xywh[..., :2], axis=-1)
    D = tf.norm(C[..., :2] - C[..., 2:], axis=-1)
    if iou_type == 'diou':
        iou -= (D_box / D) ** 2
        return iou

    arc_tan1 = tf.math.atan2(bbox1_xywh[..., 2], bbox1_xywh[..., 3])
    arc_tan2 = tf.math.atan2(bbox2_xywh[..., 2], bbox2_xywh[..., 3])
    V = (4 / np.pi ** 2) * (arc_tan1 - arc_tan2) ** 2
    alpha = V / (1 - iou + V)

    iou -= (D_box / D) ** 2 + alpha * V

    return iou


def compute_iou_array(anchors, box_label):
    batch_size = tf.shape(box_label)[0]
    anchors_num = anchors.shape[0]
    box_num = box_label.shape[1]

    anchors_coord = xywh_to_x1y1x2y2(anchors)
    anchors_coord = tf.tile(tf.expand_dims(tf.expand_dims(anchors_coord, 1), 0), (batch_size,1, box_num, 1))

    box_label_coord = xywh_to_x1y1x2y2(box_label)
    box_label_coord = tf.tile(tf.expand_dims(box_label_coord, 1), (1,anchors_num, 1, 1))

    iou = compute_iou(anchors_coord, box_label_coord)

    return iou
