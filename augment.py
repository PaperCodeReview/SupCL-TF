import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa


CROP_PADDING = 32
MAX_LEVEL = 1.
TRANSLATE_CONST = 100.
REPLACE_VALUE = 128

mean_std = {
    'cub': [[0.48552202, 0.49934904, 0.43224954], 
            [0.18172876, 0.18109447, 0.19272076]],
}

class Augment:
    '''
    Augmentation ops for SimAugment and RandAugment.
    Some operations is from https://github.com/google-research/fixmatch/blob/master/imagenet/augment/augment_ops.py.

    Arguments
        args
        mode (str) : training or validation
    '''
    
    def __init__(self, args, mode):
        self.args = args
        self.mode = mode

        self.mean, self.std = mean_std[self.args.dataset]

    def _standardize(self, x):
        x = tf.cast(x, tf.float32)
        x /= 255.
        if self.args.standardize == 'minmax1':
            pass
        elif self.args.standardize == 'minmax2':
            x -= .5
            x /= .5
        elif self.args.standardize == 'norm':
            x -= self.mean
            x /= self.std
        elif self.args.standardize == 'eachnorm':
            x = (x-tf.math.reduce_mean(x))/tf.math.reduce_std(x)
        else:
            raise ValueError()

        return x

    ## need img shape ##
    def _pad(self, x, shape):
        length = tf.reduce_max(shape)
        paddings = [
            [(length-shape[0])//2, length-((length-shape[0])//2+shape[0])], 
            [(length-shape[1])//2, length-((length-shape[1])//2+shape[1])], [0, 0]]
        x = tf.pad(x, paddings, 'CONSTANT', constant_values=0)
        return x

    def _random_crop(self, x, shape):
        bbox_begin, bbox_size, _ = tf.image.sample_distorted_bounding_box(
            image_size=shape, 
            bounding_boxes=tf.constant([0.0, 0.0, 1.0, 1.0], dtype=tf.float32, shape=[1, 1, 4]), 
            min_object_covered=.1,
            aspect_ratio_range=(3. / 4., 4. / 3.),
            area_range=(0.08, 1.0),
            max_attempts=10,
            use_image_if_no_bounding_boxes=True)

        offset_height, offset_width, _ = tf.unstack(bbox_begin)
        target_height, target_width, _ = tf.unstack(bbox_size)
        x = tf.slice(x, [offset_height, offset_width, 0], [target_height, target_width, 3])
        return x

    def _center_crop(self, x, shape):
        image_height = shape[0]
        image_width = shape[1]
        padded_center_crop_size = tf.cast(
            ((self.args.img_size/(self.args.img_size+CROP_PADDING)) * 
                tf.cast(tf.math.minimum(image_height, image_width), tf.float32)),
            tf.int32)

        offset_height = ((image_height - padded_center_crop_size) + 1) // 2
        offset_width = ((image_width - padded_center_crop_size) + 1) // 2
        x = tf.image.crop_to_bounding_box(
            x, offset_height, offset_width, padded_center_crop_size, padded_center_crop_size)
        return x
    
    ####################

    def _resize(self, x):
        x = tf.image.resize(x, (self.args.img_size, self.args.img_size), tf.image.ResizeMethod.BICUBIC)
        x = tf.saturate_cast(x, tf.uint8)
        return x

    def _rotate(self, x, angle=None):
        if angle is None:
            angle = tf.random.uniform([], -self.args.angle, self.args.angle)*np.pi/180
        else:
            angle = (angle/MAX_LEVEL) * 30.
            angle = self._randomly_negate_tensor(angle)
        x = tfa.image.rotate(x, angle, interpolation='BILINEAR')
        x = tf.saturate_cast(x, tf.uint8)
        return x

    def _contrast(self, x, factor=None):
        if factor is None:
            factor = self.args.contrast
            x = tf.image.random_contrast(x, lower=1-factor, upper=1+factor)
        else:
            factor = (factor/MAX_LEVEL) * 1.8 + .1 # [.1, 2.8]
            x = tf.image.adjust_contrast(x, factor)
        x = tf.saturate_cast(x, tf.uint8)
        return x

    def _brightness(self, x, factor=None):
        if factor is None:
            factor = self.args.brightness
            x = tf.image.random_brightness(x, max_delta=factor)
        else:
            factor = (factor/MAX_LEVEL) * 1.8 + .1
            x = tf.image.adjust_brightness(x, factor)
        x = tf.saturate_cast(x, tf.uint8)
        return x

    def _saturation(self, x, factor=None):
        if factor is None:
            factor = self.args.saturation
            x = tf.image.random_saturation(x, lower=1-factor, upper=1+factor)
        else:
            x = tf.image.adjust_saturation(x, factor)
        x = tf.saturate_cast(x, tf.uint8)
        return x

    def _hue(self, x, factor=None):
        if factor is None:
            factor = self.args.hue
            x = tf.image.random_hue(x, max_delta=factor)
        else:
            x = tf.image.adjust_hue(x, factor)
        x = tf.saturate_cast(x, tf.uint8)
        return x

    def _gray(self, x, p=.2):
        if tf.less(tf.random.uniform([], 0, 1, tf.float32), tf.cast(p, tf.float32)):
            x = tf.image.rgb_to_grayscale(x)
            x = tf.tile(x, [1, 1, 3])
        return x

    def _identity(self, x, _):
        return tf.identity(x)

    def _autoContrast(self, x, _):
        """Implements Autocontrast function from PIL using TF ops."""

        def scale_channel(channel):
            """Scale the 2D image using the autocontrast rule."""
            # A possibly cheaper version can be done using cumsum/unique_with_counts
            # over the histogram values, rather than iterating over the entire image.
            # to compute mins and maxes.
            lo = tf.cast(tf.reduce_min(channel), tf.float32)
            hi = tf.cast(tf.reduce_max(channel), tf.float32)

            # Scale the image, making the lowest value 0 and the highest value 255.
            def scale_values(im):
                scale = 255.0 / (hi - lo)
                offset = -lo * scale
                im = tf.cast(im, tf.float32) * scale + offset
                return tf.saturate_cast(im, tf.uint8)

            result = tf.cond(hi > lo, lambda: scale_values(channel), lambda: channel)
            return result

        # Assumes RGB for now.  Scales each channel independently
        # and then stacks the result.
        s1 = scale_channel(x[:, :, 0])
        s2 = scale_channel(x[:, :, 1])
        s3 = scale_channel(x[:, :, 2])
        x = tf.stack([s1, s2, s3], 2)
        return x

    def _equalize(self, x, _):
        x = tfa.image.equalize(x)
        return x

    def _solarize(self, x, threshold=128):
        # For each pixel in the image, select the pixel
        # if the value is less than the threshold.
        # Otherwise, subtract 255 from the pixel.
        threshold = int((threshold/MAX_LEVEL) * 256)
        threshold = tf.saturate_cast(threshold, x.dtype)
        return tf.where(x < threshold, x, 255 - x)

    def _color(self, x, factor):
        """Equivalent of PIL Color."""
        factor = (factor/MAX_LEVEL) * 1.8 + .1
        degenerate = tf.image.grayscale_to_rgb(tf.image.rgb_to_grayscale(x))
        x = tfa.image.blend(degenerate, tf.cast(x, tf.float32), factor)
        return tf.saturate_cast(x, tf.uint8)

    def _posterize(self, x, bits):
        """Equivalent of PIL Posterize."""
        bits = int((bits/MAX_LEVEL) * 4)
        shift = tf.cast(8 - bits, x.dtype)
        return tf.bitwise.left_shift(tf.bitwise.right_shift(x, shift), shift)

    def _translate_x(self, x, pixels):
        """Equivalent of PIL Translate in X dimension."""
        pixels = (pixels/MAX_LEVEL) * TRANSLATE_CONST
        pixels = self._randomly_negate_tensor(pixels)
        x = tfa.image.translate_ops.translate(x, [-pixels, 0])
        return x

    def _translate_y(self, x, pixels):
        """Equivalent of PIL Translate in Y dimension."""
        pixels = (pixels/MAX_LEVEL) * TRANSLATE_CONST
        pixels = self._randomly_negate_tensor(pixels)
        x = tfa.image.translate_ops.translate(x, [0, -pixels])
        return x

    def _shear_x(self, x, level):
        level = (level/MAX_LEVEL) * 0.3
        level = self._randomly_negate_tensor(level)
        x = tfa.image.shear_x(x, level, 0)
        return x

    def _shear_y(self, x, level):
        level = (level/MAX_LEVEL) * 0.3
        level = self._randomly_negate_tensor(level)
        x = tfa.image.shear_y(x, level, 0)
        return x

    def _sharpness(self, x, factor):
        factor = (factor/MAX_LEVEL) * 1.8 + .1
        x = tfa.image.sharpness(x, factor)
        return x

    def _randomly_negate_tensor(self, tensor):
        """With 50% prob turn the tensor negative."""
        should_flip = tf.cast(tf.floor(tf.random.uniform([]) + 0.5), tf.bool)
        final_tensor = tf.cond(should_flip, lambda: tensor, lambda: -tensor)
        return final_tensor


class SimAugment(Augment):
    '''SimAugment
    Data augmentation in SimCLR (https://arxiv.org/abs/2002.05709).

    Arguments
        args (Namespace) : a set of arguments for training or validation
        mode (str) : training or validation
    '''
    def __init__(self, args, mode):
        super().__init__(args, mode)
        self.augment_list = []
        if self.mode == 'train':
            self.augment_list.append(self._random_crop)
            self.augment_list.append(self._resize)
            self.augment_list.append(self._color_jitter)
            self.augment_list.append(self._gray)
        else:
            self.augment_list.append(self._center_crop)
            self.augment_list.append(self._resize)

        self.augment_list.append(self._standardize)

    def __call__(self, x, shape):
        for f in self.augment_list:
            if 'crop' in f.__name__:
                x = f(x, shape)
            else:
                x = f(x)
        return x

    def _color_jitter(self, x, p=.8):
        if tf.less(tf.random.uniform([], 0, 1, dtype=tf.float32), tf.cast(p, tf.float32)):
            x = self._brightness(x, factor=.8)
            x = self._contrast(x, factor=.8)
            x = self._saturation(x, factor=.8)
            x = self._hue(x, factor=.2)
        return x


class RandAugment(Augment):
    '''RandAugment (https://arxiv.org/abs/1909.13719)
    This class is made by referring to 
    https://github.com/google-research/fixmatch/blob/master/imagenet/augment/augment_ops.py.

    Arguments
        args (Namespace) : a set of arguments for training or validation
        mode (str) : training or validation
        prob_to_apply (float, default=.5) : the probability for applying the augmented image
    '''
    def __init__(self, args, mode, prob_to_apply=.5):
        super().__init__(args, mode)
        self.prob_to_apply = prob_to_apply

        self.augment_list = [
            self._identity,
            self._autoContrast,
            self._equalize,
            self._rotate,
            self._solarize,
            self._color,
            self._posterize,
            self._contrast,
            self._brightness,
            self._sharpness,
            self._shear_x,
            self._shear_y,
            self._translate_x,
            self._translate_y
        ]

    def _get_level(self):
        return np.random.uniform()
        # return tf.random.uniform(shape=[], dtype=tf.float32)
    
    def _apply_one_layer(self, image):
        level = self._get_level()
        branch_fns = []
        for augment_fn in self.augment_list:
            def _branch_fn(image=image,
                           augment_fn=augment_fn):
                args = [image] + list((level,))
                return augment_fn(*args)

            branch_fns.append(_branch_fn)

        branch_index = tf.random.uniform(
            shape=[], maxval=len(branch_fns), dtype=tf.int32)
        aug_image = tf.switch_case(
            branch_index, branch_fns, default=lambda: image)

        if self.prob_to_apply is not None:
            return tf.cond(
                tf.random.uniform(shape=[], dtype=tf.float32) < self.prob_to_apply,
                lambda: aug_image,
                lambda: image)
        else:
            return aug_image

    def __call__(self, image, shape):
        if self.mode == 'train':
            image = self._random_crop_resize(image, shape)
            for _ in range(self.args.randaug_layer):
                image = self._apply_one_layer(image)
        else:
            image = self._center_crop_resize(image, shape)

        image = self._standardize(image)
        return image

    def _random_crop_resize(self, x, shape):
        x = self._random_crop(x, shape)
        x = self._resize(x)
        return x

    def _center_crop_resize(self, x, shape):
        x = self._center_crop(x, shape)
        x = self._resize(x)
        return x