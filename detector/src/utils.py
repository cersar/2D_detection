import tensorflow as tf
from detector.src.box_utils import decode_box_outputs,compute_iou,xywh_to_x1y1x2y2

def post_process(outputs ,anchors ,CFG):
    batch_size = tf.shape(outputs)[0]
    cls_outputs, bbox_outputs = outputs[..., :CFG['num_classes']], outputs[..., CFG['num_classes']:]
    bbox_outputs = decode_box_outputs(bbox_outputs, anchors)
    bbox_outputs = xywh_to_x1y1x2y2(bbox_outputs)
    bbox_outputs = tf.clip_by_value(bbox_outputs,0,1)
    cls_preds = tf.argmax(cls_outputs,axis=-1,output_type=tf.int32)
    preds = []
    for i in tf.range(batch_size):
        pred = []
        for j in tf.range(CFG['num_classes']):
            ind = tf.where(cls_preds[i]==j)
            box_per_cls = tf.gather_nd(bbox_outputs[i],ind)
            score_per_cls = tf.gather_nd(cls_outputs[i],ind)
            nms_ind = tf.image.non_max_suppression(box_per_cls, score_per_cls[:,j],CFG['max_output_obj_num'],
                                                   iou_threshold=CFG['nms_iou_thres'],score_threshold=CFG['nms_score_thres'])

            if tf.shape(nms_ind)[0]>0:
                pred.append(tf.concat([tf.gather(score_per_cls,nms_ind),tf.gather(box_per_cls,nms_ind)],-1))
        if len(pred):
            pred = tf.concat(pred, 0)
        else:
            pred = tf.zeros((0,24))

        if tf.shape(pred)[0] < CFG['max_output_obj_num']:
            pred = tf.pad(pred, ((0, CFG['obj_max_num'] - tf.shape(pred)[0]), (0, 0)), constant_values=0)
        else:
            pred = pred[:CFG['max_output_obj_num']]
        pred = tf.reshape(pred, (CFG['max_output_obj_num'], 24))
        preds.append(pred)
    preds = tf.stack(preds)
    return preds