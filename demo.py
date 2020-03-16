
import numpy as np
np.random.seed(1)
# from tensorflow import set_random_seed
# set_random_seed(1)
import tensorflow as tf
tf.random.set_seed(1)
from neural_stylization.transfer_style import Stylizer
from neural_stylization.optimizers import GradientDescent, L_BFGS, Adam
from neural_stylization.util.build_callback import build_callback
from neural_stylization.util.img_util import load_image
CONTENT = 'img/content/weiyao.jpg'
load_image(CONTENT)
sty = Stylizer(content_weight=1, style_weight=2e4)

DIMS = int(1024/3), int(768/3)

starry_night = sty(
    content_path=CONTENT,
    style_path='img/styles/the-starry-night.jpg',
    optimize=L_BFGS(max_evaluations=20),
    iterations=50,
    image_size=DIMS,
    callback=build_callback('build/transfer/the-starry-night')
)

starry_night.save('the-starry-night.png')


scream = sty(
    content_path=CONTENT,
    style_path='img/styles/the-scream.jpg',
    optimize=L_BFGS(max_evaluations=20),
    iterations=50,
    image_size=DIMS,
    callback=build_callback('build/transfer/the-scream')
)

scream.save('the-scream.png')