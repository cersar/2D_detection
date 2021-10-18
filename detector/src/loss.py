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

    num_class = tf.shape(y_pred)[-1]

    # Get the cross_entropy for each entry
    ce = binary_crossentropy(tf.reshape(y_true, (-1, 1)), tf.reshape(y_pred, (-1, 1)), from_logits=from_logits, label_smoothing=label_smoothing)
    ce = tf.reshape(ce, (-1, num_class))
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
    loss = tf.reduce_mean(loss)
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
    loss = tf.reduce_mean(loss)
    return loss

def iou_loss(box_out,box_label,anchors,CFG):
    box_label_coord = xywh_to_x1y1x2y2(box_label)
    box_pred = decode_box_outputs(box_out,anchors)
    box_pred_coord = xywh_to_x1y1x2y2(box_pred)

    iou = compute_iou(box_pred_coord,box_label_coord,CFG['iou_type'])

    loss = tf.reduce_mean(1-iou)

    return loss

def compute_cls_loss(cls_out,cls_label,CFG):
    cls_label = tf.one_hot(tf.cast(cls_label,tf.int32),CFG['num_classes'])
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

def compute_detect_loss(cls_outputs,bbox_outputs,anchors,labels,CFG):
    batch_size = cls_outputs.shape[0]

    cls_loss = []
    reg_loss = []
    for i in range(batch_size):
        cls_label = labels[i,:,0]
        box_label = labels[i,:,1:]
        cls_out = cls_outputs[i]
        bbox_out = bbox_outputs[i]
        valid_ind = tf.where(cls_label!=-1)
        cls_label = tf.gather_nd(cls_label,valid_ind)
        box_label = tf.gather_nd(box_label, valid_ind)

        iou_anchor = compute_iou_array(anchors,box_label)
        max_iou = tf.reduce_max(iou_anchor,axis=-1)
        max_ind = tf.expand_dims(tf.argmax(iou_anchor,axis=-1),-1)
        pos_ind = tf.where(max_iou > CFG['pos_thres'])
        neg_ind = tf.where(max_iou < CFG['neg_thres'])

        pos_num = tf.shape(pos_ind)[0]
        neg_num = tf.shape(neg_ind)[0]

        cls_label = tf.tensor_scatter_nd_update(cls_label,neg_ind,tf.ones(neg_num)*(-1.))
        cls_loss_ind = tf.concat([pos_ind,neg_ind],axis=0)
        reg_loss_ind = pos_ind

        box_ind = tf.gather_nd(max_ind, reg_loss_ind)
        cls_ind = tf.gather_nd(max_ind, cls_loss_ind)

        box_label = tf.gather_nd(box_label, box_ind)
        cls_label = tf.gather_nd(cls_label, cls_ind)

        cls_out = tf.gather_nd(cls_out, cls_ind)
        box_out = tf.gather_nd(bbox_out, box_ind)

        anchors_matched = tf.gather_nd(anchors, box_ind)
        cls_loss.append(compute_cls_loss(cls_out,cls_label,CFG))
        reg_loss.append(compute_reg_loss(box_out,box_label,anchors_matched,CFG))

    cls_loss = tf.reduce_mean(cls_loss)
    reg_loss = tf.reduce_mean(reg_loss)

    loss = cls_loss+reg_loss
    return loss
