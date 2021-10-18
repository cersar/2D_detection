import tensorflow as tf
data_description = {
    'image_id': tf.io.FixedLenFeature([], tf.string),
    'image': tf.io.FixedLenFeature([], tf.string),
    "label": tf.io.VarLenFeature(tf.float32),
}

def _parse_data(example_proto,obj_max_num=5):
    # Parse the input tf.Example proto using the dictionary above.
    sample = tf.io.parse_single_example(example_proto, data_description)
    image = tf.io.decode_image(sample['image'])
    label = tf.sparse.to_dense(sample['label'])
    obj_num = tf.shape(label)[0]//5
    label = tf.reshape(label,(obj_num,5))
    if obj_num < obj_max_num:
        label = tf.pad(label, ((0, obj_max_num - obj_num), (0, 0)))
    else:
        label = label[:obj_max_num]
    return image,label

dataset = tf.data.TFRecordDataset(['./test.tfrec']).map(_parse_data)
for image,label in dataset:
    print(image.shape)
    print(label)