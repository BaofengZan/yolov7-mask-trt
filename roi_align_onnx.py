#coding=utf-8
'''
测试roiAlign onnx 导出
'''

import torch
import torch.nn as nn
import torchvision as tv
import numpy as np

#from functools import wraps
# from torchvision.ops.roi_align import _RoIAlignFunction

# def add_method(cls):
#     def decorator(func):
#         @wraps(func)
#         def wrapper(*args, **kwargs):
#             return func(*args, **kwargs)
#         setattr(cls, func.__name__, wrapper)
#         # Note we are not binding func, but wrapper which accepts self but does exactly the same as func
#         return func # returning func means func can still be used normally
#     return decorator

# @add_method(_RoIAlignFunction)
# def symbolic(g, input, roi, output_size, spatial_scale, sampling_ratio):
#     import torch.onnx.symbolic_helper as sym_help

#     sampling_ratio = sampling_ratio if sampling_ratio > 0 else 0

#     rois = sym_help._slice_helper(g, roi, axes=[1], starts=[1], ends=[5])
#     index = g.op("Constant", value_t=torch.tensor(0, dtype=torch.int64))
#     batch_indices = g.op("Gather", roi, index, axis_i=1)
#     batch_indices = g.op("Cast", batch_indices, to_i=sym_help.cast_pytorch_to_onnx["Long"])

#     return g.op("RoiAlign", input, rois, batch_indices, mode_s="avg", output_height_i=output_size[0],
#                 output_width_i=output_size[1], sampling_ratio_i=sampling_ratio, spatial_scale_f=spatial_scale)


from torch.onnx import register_custom_op_symbolic
from torch.onnx.symbolic_helper import parse_args

'''
def roi_align(
    input: Tensor,
    boxes: Union[Tensor, List[Tensor]],
    output_size: BroadcastingList2[int],
    spatial_scale: float = 1.0,
    sampling_ratio: int = -1,
    aligned: bool = False,
) -> Tensor:

"v": no conversion, keep torch._C.Value.
"i": int
"is": list of int
"f": float
"fs": list of float
"b": bool
"s": str
"t": torch.Tensor
'''
@parse_args("v", "v", "v", "f", "i", "b")
def symbolic(g,
        input,
        boxes,
        output_size,
        spatial_scale,
        sampling_ratio,
        aligned):
    return g.op("custom::ROIAlign", input, boxes, output_size, spatial_scale,sampling_ratio,  aligned)

'''
1 register_custom_op_symbolic函数的第一个参数'torchvision::deform_conv2d'为pytorch对应操作符名称，若填写错误，则会导致自定义算子注册失败
2  https://github.com/pytorch/pytorch/blob/master/torch/onnx/symbolic_helper.py
'''

# 注册到opset 12 中
register_custom_op_symbolic("torchvision::ROIAlign", symbolic, 12)

### Testing

class Model(nn.Module):
    def __init__(self, output_size=(14, 14)):
        super(Model, self).__init__()

        self.output_size = output_size

    def forward(self, feature_map, bbx_pd):

        output = tv.ops.roi_align(feature_map, bbx_pd, self.output_size)

        return output


model = Model()

batch_size = 2
a = 10

feature_map = torch.randn(batch_size, 20, 64, 64)
bbx_pd = torch.randint(10, 15, (a, 5)).float()
bbx_pd[:, 0] = torch.randint(0, batch_size, (bbx_pd.size(0),))


torch.onnx.export(model, (feature_map, bbx_pd), 'model.onnx', input_names=["feature_map", "bbx_pd"], verbose=True, opset_version=12)
