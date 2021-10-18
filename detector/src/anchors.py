import numpy as np
import itertools

def gen_anchors(CFG):
    if CFG['input_shape'][0] % 2**CFG['pyramid_levels'][-1] != 0 or CFG['input_shape'][1] % 2**CFG['pyramid_levels'][-1] != 0:
        raise ValueError('input size must be divided by the stride.')
    anchor_boxes = []
    for level in CFG['pyramid_levels']:
        level_size = CFG['input_shape'][1]//2**level,CFG['input_shape'][0]//2**level
        grid_size = 1./level_size[0],1./level_size[1]
        anchors_level = []
        for scale, ratio in itertools.product(CFG['anchors_scales'], CFG['anchors_aspects']):
            base_anchor_size_w = CFG['anchor_scale'] * grid_size[0] * scale
            base_anchor_size_h = CFG['anchor_scale'] * grid_size[1] * scale
            anchor_size_w = base_anchor_size_w * ratio[0]
            anchor_size_h = base_anchor_size_h * ratio[1]

            x = np.arange(grid_size[0] / 2, 1, grid_size[0])
            y = np.arange(grid_size[1] / 2, 1, grid_size[1])

            xv, yv = np.meshgrid(x, y)
            xv = xv.reshape(-1)
            yv = yv.reshape(-1)

            # y1,x1,y2,x2
            boxes = np.vstack((xv, yv,
                               np.ones_like(xv)*anchor_size_w, np.ones_like(yv)*anchor_size_h))
            boxes = np.swapaxes(boxes, 0, 1)
            anchors_level.append(np.expand_dims(boxes, axis=1))
        # concat anchors on the same level to the reshape NxAx4
        anchors_level = np.concatenate(anchors_level, axis=1)
        anchor_boxes.append(anchors_level.reshape([-1, 4]))

    anchor_boxes = np.vstack(anchor_boxes).astype(np.float32)
    return anchor_boxes
