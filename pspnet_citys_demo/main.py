import os
import mxnet as mx
from mxnet import image, gpu
import gluoncv
from gluoncv.data.transforms.presets.segmentation import test_transform
from gluoncv.utils.viz import get_color_pallete, plot_image
import warnings

warnings.filterwarnings(action='once')
warnings.filterwarnings("ignore")
ctx = mx.gpu(0)

model = gluoncv.model_zoo.get_model('psp_resnet101_citys', ctx=ctx, pretrained=True)

file_path = r'data'
mask_path = r'mask'
filelist = os.listdir(file_path)
for i in filelist:
    img_path = file_path + "/" + i
    save_path = mask_path + "/" + i + ".png"
    img = image.imread(img_path)
    img = test_transform(img, ctx=ctx)
    output = model.predict(img)
    predict = mx.nd.squeeze(mx.nd.argmax(output, 1)).asnumpy()

    mask = get_color_pallete(predict, 'citys')
    mask.save(save_path)
    print(i + " finish")
