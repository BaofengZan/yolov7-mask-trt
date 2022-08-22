#coding=utf-8
"""
导出onnx后。
1 生成engine
    trtexec --onnx=./yolov7.onnx --saveEngine=./yolov7_fp16.engine --fp16 --workspace=200
    D:\TensorRT-8.4.1.5.Windows10.x86_64.cuda-10.2.cudnn8.4\TensorRT-8.4.1.5\lib\trtexec.exe --onnx=./yolov7.onnx --saveEngine=./yolov7_fp32.engine --workspace=1000
    D:\TensorRT-8.4.1.5.Windows10.x86_64.cuda-10.2.cudnn8.4\TensorRT-8.4.1.5\lib\trtexec.exe --onnx=./yolov7.onnx --saveEngine=./yolov7_fp16.engine --fp16 --workspace=1000
2 使用该脚本infer
"""
import cv2
import yaml
import numpy as np
from collections import OrderedDict,namedtuple
import time
from torchvision import transforms
import numpy as np

from utils.datasets import letterbox
import os
os.environ['path'] += ";E:\TensorRT-8.4.1.5.Windows10.x86_64.cuda-10.2.cudnn8.4\TensorRT-8.4.1.5\lib"


import tensorrt as trt
import torch
from utils.general import non_max_suppression_mask_conf

from detectron2.modeling.poolers import ROIPooler
from detectron2.structures import Boxes
from detectron2.utils.memory import retry_if_cuda_oom
from detectron2.layers import paste_masks_in_image

coconame = [
"person",
"bicycle",
"car",
"motorbike",
"aeroplane",
"bus",
"train",
"truck",
"boat",
"traffic light",
"fire hydrant",
"stop sign",
"parking meter",
"bench",
"bird",
"cat",
"dog",
"horse",
"sheep",
"cow",
"elephant",
"bear",
"zebra",
"giraffe",
"backpack",
"umbrella",
"handbag",
"tie",
"suitcase",
"frisbee",
"skis",
"snowboard",
"sports ball",
"kite",
"baseball bat",
"baseball glove",
"skateboard",
"surfboard",
"tennis racket",
"bottle",
"wine glass",
"cup",
"fork",
"knife",
"spoon",
"bowl",
"banana",
"apple",
"sandwich",
"orange",
"broccoli",
"carrot",
"hot dog",
"pizza",
"donut",
"cake",
"chair",
"sofa",
"pottedplant",
"bed",
"diningtable",
"toilet",
"tvmonitor",
"laptop",
"mouse",
"remote",
"keyboard",
"cell phone",
"microwave",
"oven",
"toaster",
"sink",
"refrigerator",
"book",
"clock",
"vase",
"scissors",
"teddy bear",
"hair drier",
"toothbrush"
]

class TRT_engine():
    def __init__(self, weight, hyp_cfg="data/hyp.scratch.mask.yaml") -> None:
        self.imgsz = [960, 960]
        self.weight = weight
        self.device = torch.device('cuda:0')
        self.init_engine()
        with open(hyp_cfg, 'r') as f:
            self.hyp = yaml.load(f, Loader=yaml.FullLoader)

    def init_engine(self):
        # Infer TensorRT Engine
        self.Binding = namedtuple('Binding', ('name', 'dtype', 'shape', 'data', 'ptr'))
        self.logger = trt.Logger(trt.Logger.INFO)
        trt.init_libnvinfer_plugins(self.logger, namespace="")
        with open(self.weight, 'rb') as self.f, trt.Runtime(self.logger) as self.runtime:
            self.model = self.runtime.deserialize_cuda_engine(self.f.read())
        self.bindings = OrderedDict()
        self.fp16 = False
        print(f"num binding = {self.model.num_bindings}")
        for index in range(self.model.num_bindings):
            self.name = self.model.get_binding_name(index)
            print(f"name = {self.name}")
            self.dtype = trt.nptype(self.model.get_binding_dtype(index))
            self.shape = tuple(self.model.get_binding_shape(index))
            self.data = torch.from_numpy(np.empty(self.shape, dtype=np.dtype(self.dtype))).to(self.device)
            self.bindings[self.name] = self.Binding(self.name, self.dtype, self.shape, self.data, int(self.data.data_ptr()))
            # if self.model.binding_is_input(index) and self.dtype == np.float16:
            #     self.fp16 = True
        self.binding_addrs = OrderedDict((n, d.ptr) for n, d in self.bindings.items())
        self.context = self.model.create_execution_context()

    def letterbox(self, img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
        # Resize and pad image while meeting stride-multiple constraints
        shape = img.shape[:2]  # current shape [height, width]
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not scaleup:  # only scale down, do not scale up (for better test mAP)
            r = min(r, 1.0)

        # Compute padding
        ratio = r, r  # width, height ratios
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
        if auto:  # minimum rectangle
            dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
        elif scaleFill:  # stretch
            dw, dh = 0.0, 0.0
            new_unpad = (new_shape[1], new_shape[0])
            ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

        dw /= 2  # divide padding into 2 sides
        dh /= 2

        if shape[::-1] != new_unpad:  # resize
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
        return img, ratio, (dw, dh)
    def preprocess(self,image):
        self.img = self.letterbox(image, 640, stride=64, auto=True)[0]
        #self.img = self.img[...,::-1]  # BGR转RGB
        self.img = self.img.transpose((2, 0, 1))
        self.img = np.expand_dims(self.img,0)
        self.img = np.ascontiguousarray(self.img)
        self.img = torch.from_numpy(self.img).to(self.device)
        self.img = self.img.float()
        self.img /= 255.0
        return self.img
    def tensor_to_img(self, tensor):
        nimg = tensor[0].permute(1, 2, 0) * 255
        nimg = nimg.cpu().numpy().astype(np.uint8)
        nimg = cv2.cvtColor(nimg, cv2.COLOR_RGB2BGR)
        return nimg
    def predict(self,img,threshold):
        #self.show_img = img
        img = self.preprocess(img)
        self.binding_addrs['images'] = int(img.data_ptr())
        self.context.execute_v2(list(self.binding_addrs.values()))
        # nums = self.bindings['num_detections'].data[0].tolist()
        # boxes = self.bindings['detection_boxes'].data[0].tolist()
        # scores =self.bindings['detection_scores'].data[0].tolist()
        # classes = self.bindings['detection_labels'].data[0].tolist()
        
        inf_out = self.bindings['test'].data
        #train_out= self.bindings['bbox_and_cls'].data[0].tolist()
        attn= self.bindings['atten'].data
        bases= self.bindings['bases'].data
        sem_output= self.bindings['sem'].data
        

        # inf_out = torch.tensor(inf_out)
        # attn = torch.tensor(attn)
        # bases = torch.tensor(bases)
        # sem_output = torch.tensor(sem_output)
        
        bases = torch.cat([bases, sem_output], dim=1)
        nb, _, height, width = img.shape
        names = coconame
        pooler_scale = 0.25
        pooler = ROIPooler(output_size=self.hyp['mask_resolution'], scales=(
            pooler_scale,), sampling_ratio=1, pooler_type='ROIAlignV2', canonical_level=2)

        output, output_mask, output_mask_score, output_ac, output_ab = non_max_suppression_mask_conf(
            inf_out, attn, bases, pooler, self.hyp, conf_thres=0.25, iou_thres=0.65, merge=False, mask_iou=None)

        pred, pred_masks = output[0], output_mask[0]
        try:
            bboxes = Boxes(pred[:, :4])
            original_pred_masks = pred_masks.view(-1,
                                                  self.hyp['mask_resolution'], self.hyp['mask_resolution'])
            pred_masks = retry_if_cuda_oom(paste_masks_in_image)(
                original_pred_masks, bboxes, (height, width), threshold=0.5)
            pred_masks_np = pred_masks.detach().cpu().numpy()
            pred_cls = pred[:, 5].detach().cpu().numpy()
            pred_conf = pred[:, 4].detach().cpu().numpy()

            nbboxes = bboxes.tensor.detach().cpu().numpy().astype(np.int)

            if 1:
                pnimg = self.vis_output(
                    self.img, pred_masks_np, nbboxes, pred_cls, pred_conf, names)
                pnimg = cv2.cvtColor(pnimg, cv2.COLOR_BGR2RGB)
                cv2.imwrite("mask_out.jpg", pnimg)
        except Exception as e:
            print('No mask found')
            return None, None, None, None

    def vis_output(self,
                   img_tensor,
                   pred_masks_np,
                   nbboxes,
                   pred_cls,
                   pred_conf,
                   names):
        pnimg = self.tensor_to_img(img_tensor)
        for one_mask, bbox, cls, conf in zip(pred_masks_np, nbboxes, pred_cls, pred_conf):
            if conf < 0.25:
                continue
            color = [np.random.randint(255), np.random.randint(
                255), np.random.randint(255)]

            pnimg[one_mask] = pnimg[one_mask] * 0.5 + \
                np.array(color, dtype=np.uint8) * 0.5
            pnimg = cv2.rectangle(
                pnimg, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
            label = '%s %.3f' % (names[int(cls)], conf)
            t_size = cv2.getTextSize(
                label, 0, fontScale=0.5, thickness=1)[0]
            c2 = bbox[0] + t_size[0], bbox[1] - t_size[1] - 3
            pnimg = cv2.rectangle(
                pnimg, (bbox[0], bbox[1]), c2, color, -1, cv2.LINE_AA)  # filled
            pnimg = cv2.putText(pnimg, label, (bbox[0], bbox[1] - 2), 0, 0.5, [
                                255, 255, 255], thickness=1, lineType=cv2.LINE_AA)
        return pnimg

trt_engine = TRT_engine("./mask_fp32.engine")
img = cv2.imread(r"inference/images/horses.jpg")
# i = 0
# while(i < 10):
#     results,_ = trt_engine.predict(img,threshold=0.5)
#     i+=1

i=0
sumtime = 0
while(i<1):
    tic1 = time.perf_counter()
    trt_engine.predict(img, threshold=0.5)
    toc1 = time.perf_counter()
    print(f"one img infer time = {(toc1-tic1)*1000} ms")
    sumtime += (toc1-tic1)
    i+=1

print(f"Avg infer time = {(sumtime/100)*1000} ms")
