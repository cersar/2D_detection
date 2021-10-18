import tensorflow as tf
import time
from functools import partial
from tensorflow.keras.utils import plot_model
from tensorflow.keras.optimizers import Adam

from detector.src.loss import compute_detect_loss
from detector.src.model import get_detector

obj_max_num = 50
CFG = {
    'backbone_name':'EfficientNetB0',
    'input_shape' : (512,512,3),
    'num_classes' : 2,
    'anchors_scales':[2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)],
    'anchors_aspects':[(1.0, 1.0), (1.4, 0.7), (0.7, 1.4)],
    'anchor_scale': 4.0,
    'pyramid_levels' : [3,4,5,6,7],
    'neck' : 'bifpn',
    'neck_width' : 64,
    'neck_len' : 3,
    'detect_head' : 'retina',
    'detect_head_len' : 3,
    'pos_thres': 0.5,
    'neg_thres': 0.4,
    'focal_alpha': [0.25,0.25],
    'focal_gamma': 2.0,
    'label_smooth': 0,
    'cls_loss': 'focal_loss',
    'reg_loss': 'iou',
    'iou_type': 'ciou',
}

CFG['num_anchors'] = len(CFG['anchors_scales'])*len(CFG['anchors_aspects'])


model,anchors = get_detector(CFG)

# print(anchors)
plot_model(model)


data_description = {
    'image_id': tf.io.FixedLenFeature([], tf.string),
    'image': tf.io.FixedLenFeature([], tf.string),
    "label": tf.io.VarLenFeature(tf.float32),
}

def _parse_data(example_proto,obj_max_num=50):
    # Parse the input tf.Example proto using the dictionary above.
    sample = tf.io.parse_single_example(example_proto, data_description)
    image = tf.io.decode_image(sample['image'])
    label = tf.sparse.to_dense(sample['label'])
    obj_num = tf.shape(label)[0]//5
    label = tf.reshape(label,(obj_num,5))
    if obj_num < obj_max_num:
        label = tf.pad(label, ((0, obj_max_num - obj_num), (0, 0)),constant_values=-1)
    else:
        label = label[:obj_max_num]

    return image,label

dataset = tf.data.TFRecordDataset(['./test.tfrec']).map(partial(_parse_data,obj_max_num=obj_max_num)).batch(4)

optimizer = Adam(learning_rate=1e-3)
epoch = 20
for i in range(epoch):
    print('Epoch {}'.format(i+1))
    for j,(image_batch,label_batch) in enumerate(dataset):
        start = time.perf_counter()
        with tf.GradientTape() as tape:
            cls_out, bbox_out = model(image_batch)
            loss = compute_detect_loss(cls_out, bbox_out,anchors,label_batch,CFG)
        grads = tape.gradient(loss, model.trainable_weights)
        optimizer.apply_gradients(zip(grads, model.trainable_weights))

        end = time.perf_counter()
        print('step: {}, loss: {}, time cost: {}s'.format(j,loss,end-start))



print(cls_out.shape,bbox_out.shape)



model.save('model.h5')