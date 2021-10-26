import tensorflow as tf
from tensorflow.keras.losses import binary_crossentropy
from detector.src.box_utils import *


def focal_loss(
        y_true,
        y_pred,
        alpha=0.25,
        gamma=2.0,
        label_smoothing=0.0,
        from_logits=False,
) -> tf.Tensor:
    """Implements the focal loss function.
    Focal loss was first introduced in the RetinaNet paper
    (https://arxiv.org/pdf/1708.02002.pdf). Focal loss is extremely useful for
    classification when you have highly imbalanced classes. It down-weights
    well-classified examples and focuses on hard examples. The loss value is
    much high for a sample which is misclassified by the classifier as compared
    to the loss value corresponding to a well-classified example. One of the
    best use-cases of focal loss is its usage in object detection where the
    imbalance between the background class and other classes is extremely high.
    Args:
        y_true: true targets tensor.
        y_pred: predictions tensor.
        alpha: balancing factor.
        gamma: modulating factor.
    Returns:
        Weighted loss float `Tensor`. If `reduction` is `NONE`,this has the
        same shape as `y_true`; otherwise, it is scalar.
    """
    if gamma and gamma < 0:
        raise ValueError("Value of gamma should be greater than or equal to zero.")

    tf.assert_equal(tf.shape(y_true),tf.shape(y_pred))
    y_pred = tf.convert_to_tensor(y_pred)
    y_true = tf.cast(y_true, dtype=y_pred.dtype)

    num_class = y_pred.shape[2]
    feature_size = y_pred.shape[1]

    # Get the cross_entropy for each entry
    ce = binary_crossentropy(tf.reshape(y_true, (-1, 1)), tf.reshape(y_pred, (-1, 1)), from_logits=from_logits, label_smoothing=label_smoothing)
    ce = tf.reshape(ce, (-1, feature_size, num_class))
    # If logits are provided then convert the predictions into probabilities
    if from_logits:
        pred_prob = tf.sigmoid(y_pred)
    else:
        pred_prob = y_pred

    p_t = (y_true * pred_prob) + ((1 - y_true) * (1 - pred_prob))

    alpha = tf.cast(alpha, dtype=y_true.dtype)
    alpha_factor = y_true * alpha + (1 - y_true) * (1 - alpha)

    gamma = tf.cast(gamma, dtype=y_true.dtype)
    modulating_factor = tf.pow((1.0 - p_t), gamma)

    loss = alpha_factor * modulating_factor * ce
    loss = tf.reduce_mean(loss,-1)
    return loss


def smooth_l1_loss(box_out,box_label,anchors):
    reg_x = (box_label[..., 0] - anchors[..., 0]) / anchors[..., 2]
    reg_y = (box_label[..., 1] - anchors[..., 1]) / anchors[..., 3]
    reg_w = tf.math.log(box_label[..., 2] / anchors[..., 2])
    reg_h = tf.math.log(box_label[..., 3] / anchors[..., 3])
    reg_target = tf.stack([reg_x, reg_y, reg_w, reg_h], axis=1)

    diff = tf.math.abs(box_out-reg_target)
    mask = tf.cast(diff<1,tf.float32)
    loss = 0.5*diff**2*mask+(diff-0.5)*(1-mask)
    return loss

def iou_loss(box_out,box_label,anchors,CFG):
    box_label_coord = xywh_to_x1y1x2y2(box_label)
    box_pred = decode_box_outputs(box_out,anchors)
    box_pred_coord = xywh_to_x1y1x2y2(box_pred)

    iou = compute_iou(box_pred_coord,box_label_coord,CFG['iou_type'])
    loss = 1-iou
    return loss

def compute_cls_loss(cls_out,cls_label,CFG):
    cls_label = tf.one_hot(tf.cast(cls_label,tf.int32),CFG['num_classes']+1)
    cls_label = cls_label[...,:-1]
    if CFG['cls_loss'] == 'focal_loss':
        loss = focal_loss(cls_label,cls_out,alpha=CFG['focal_alpha'],gamma=CFG['focal_gamma'],label_smoothing=CFG['label_smooth'])
    else:
        raise ValueError('invalid cls_loss {}!'.format(CFG['cls_loss']))
    return loss

def compute_reg_loss(box_out,box_label,anchors,CFG):
    if CFG['reg_loss'] == 'smooth_l1':
        loss = smooth_l1_loss(box_out,box_label,anchors)
    elif CFG['reg_loss'] == 'iou':
        loss = iou_loss(box_out,box_label,anchors,CFG)
    else:
        raise ValueError('invalid cls_loss {}!'.format(CFG['cls_loss']))

    return loss

def compute_detect_loss(labels,outputs,anchors,CFG):
    eps = 1e-10
    cls_outputs,bbox_outputs = outputs[...,:CFG['num_classes']],outputs[...,CFG['num_classes']:]
    batch_size = tf.shape(cls_outputs)[0]
    feature_size = anchors.shape[0]

    cls_labels = labels[..., 0]
    box_labels = labels[..., 1:]
    cls_labels = tf.reshape(cls_labels, (-1, CFG['obj_max_num']))
    box_labels = tf.reshape(box_labels,(-1,CFG['obj_max_num'],4))

    iou_anchor = compute_iou_array(anchors, box_labels)

    ind1 = tf.reshape(tf.tile(tf.reshape(tf.range(batch_size), (-1, 1)), (1, feature_size)), (-1, 1))
    max_iou = tf.reduce_max(iou_anchor, axis=-1)
    max_ind = tf.argmax(iou_anchor, axis=-1, output_type=ind1.dtype)
    ind2 = tf.reshape(max_ind,(-1,1))
    matched_ind = tf.concat([ind1, ind2], -1)

    cls_labels = tf.gather_nd(cls_labels, matched_ind)
    box_labels = tf.gather_nd(box_labels, matched_ind)

    cls_labels = tf.reshape(cls_labels,(-1,feature_size))
    box_labels = tf.reshape(box_labels, (-1, feature_size,4))

    valid_mask = cls_labels != -1

    pos_mask = max_iou > CFG['pos_thres']
    neg_mask = max_iou < CFG['neg_thres']

    cls_mask = tf.logical_or(pos_mask,neg_mask)
    cls_mask = tf.logical_and(cls_mask, valid_mask)
    reg_mask = tf.logical_and(pos_mask, valid_mask)

    cls_mask = tf.cast(cls_mask, outputs.dtype)
    reg_mask = tf.cast(reg_mask, outputs.dtype)

    cls_labels = cls_labels*reg_mask+(1-reg_mask)*CFG['num_classes']


    cls_loss = compute_cls_loss(cls_outputs, cls_labels, CFG)
    reg_loss = compute_reg_loss(bbox_outputs, box_labels, anchors, CFG)

    cls_loss = tf.reduce_sum(cls_loss*cls_mask,-1)/(tf.reduce_sum(reg_mask,-1)+eps)
    reg_loss = tf.reduce_sum(reg_loss*reg_mask,-1)/(tf.reduce_sum(reg_mask,-1)+eps)

    # print(tf.reduce_sum(reg_mask,-1))
    # print('cls loss: {}, reg loss: {}'.format(tf.reduce_mean(CFG['cls_weight']*cls_loss),tf.reduce_mean(CFG['reg_weight']*reg_loss)))

    loss = tf.reduce_mean(CFG['cls_weight']*cls_loss+CFG['reg_weight']*reg_loss)
    return loss
