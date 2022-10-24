import mindspore
import mindspore.nn as nn
import mindspore.ops as ops
from functools import partial
import mindspore.numpy as np
import math
from mindspore import Tensor, context
from mindspore.common.initializer import Normal,TruncatedNormal
from mindspore.common import initializer as weight_init



class Block(nn.Cell):
    def __init__(self, dim,kernel_size):
        super().__init__()
        self.spatial=nn.SequentialCell(
            nn.Conv2d(dim, dim, kernel_size, group=dim,has_bias=True),
            nn.GELU(),
            nn.BatchNorm2d(dim))
        self.channel=nn.SequentialCell(
            nn.Conv2d(dim, dim, kernel_size=1),
            nn.GELU(),
            nn.BatchNorm2d(dim))

    def construct(self, x):

        x=x+self.spatial(x)
        x=self.channel(x)
        return x

    
class ConvMixer(nn.Cell):
    def __init__(self, dim=1536, depth=20,kernel_size=9,patch_size=7, n_classes=1000,img_size=224, in_chans=3,drop_rate=0., drop_path_rate=0.):
        super().__init__()
        self.patch_embed = nn.SequentialCell(
            nn.Conv2d(3, dim, kernel_size=patch_size, stride=patch_size),
            nn.GELU(),
            nn.BatchNorm2d(dim))
        self.block= nn.CellList([Block(dim=dim, kernel_size=kernel_size) for j in range(depth)])
        self.avgpool=nn.AvgPool2d(32,32)
        self.flatten=nn.Flatten()
        self.head=nn.Dense(dim, n_classes)
        self.init_weights()
    
    def init_weights(self):
        for _, cell in self.cells_and_names():
            if isinstance(cell, nn.Dense):
                cell.weight.set_data(weight_init.initializer(weight_init.TruncatedNormal(sigma=0.02),
                                                             cell.weight.shape,
                                                             cell.weight.dtype))
                if isinstance(cell, nn.Dense) and cell.bias is not None:
                    cell.bias.set_data(weight_init.initializer(weight_init.Zero(),
                                                               cell.bias.shape,
                                                               cell.bias.dtype))
            elif isinstance(cell, (nn.LayerNorm, nn.BatchNorm2d, nn.GroupNorm, nn.SyncBatchNorm)):
                cell.gamma.set_data(weight_init.initializer(weight_init.One(),
                                                            cell.gamma.shape,
                                                            cell.gamma.dtype))
                cell.beta.set_data(weight_init.initializer(weight_init.Zero(),
                                                           cell.beta.shape,
                                                           cell.beta.dtype))
            elif isinstance(cell, nn.Conv2d):
                cell.weight.set_data(weight_init.initializer(weight_init.HeNormal(mode='fan_out', nonlinearity='relu'),
                                                             cell.weight.shape,
                                                             cell.weight.dtype))
    def construct(self, x):
        x = self.patch_embed(x)
        for blk in self.block:
            x = blk(x)
        
        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.head(x)
        
        return x
    
_cfg = {
    'url': '',
    'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
    'crop_pct': .96, 'interpolation': 'bicubic',
    'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225], 'classifier': 'head'
}


def convmixer_1536_20(pretrained=False, **kwargs):
    model = ConvMixer(1536, 20, kernel_size=9, patch_size=7, n_classes=1000)
    model.default_cfg = _cfg
    return model

def convmixer_768_32(pretrained=False, **kwargs):
    model = ConvMixer(768, 32, kernel_size=7, patch_size=7, n_classes=1000)
    model.default_cfg = _cfg
    return model


if __name__ == '__main__':
    
	from mindspore import Tensor
	import numpy as np
	from mindspore import context, set_seed, Model
	import matplotlib.pyplot as plt
	from PIL import Image
	context.set_context(mode=context.GRAPH_MODE, device_target="CPU")

 
	def visualize_model(path):
		image = Image.open(path).convert("RGB")
		image = image.resize((224, 224))
		plt.imshow(image)

    # 归一化处理
		mean = np.array([0.485 * 255, 0.456 * 255, 0.406 * 255])
		std = np.array([0.229 * 255, 0.224 * 255, 0.225 * 255])
		image = np.array(image)
		image = (image - mean) / std
		image = image.astype(np.float32)

    # 图像通道由(h, w, c)转换为(c, h, w)
		image = np.transpose(image, (2, 0, 1))

    # 扩展数据维数为(1, c, h, w)
		image = np.expand_dims(image, axis=0)

    # 定义并加载网络
		net = convmixer_1536_20()
    #param_dict = load_checkpoint("./best.ckpt")
    #load_param_into_net(net, param_dict)
		model = Model(net)
		print(net)

    # 模型预测
		pre = model.predict(Tensor(image))
		print(pre.shape)
		result = np.argmax(pre)
    
		result=1 if result>499 else 0
		class_name = {0: "cat", 1: "dog"}
		plt.title(f"Predict: {class_name[result]}")
		return result

	image1 = "./test.jpg"
	plt.figure(figsize=(7, 7))

	visualize_model(image1)
	plt.show()

