# yolov7-mask Trt

1. 目前仅将onnx导出，然后后处理仍然使用的detectron中的函数
   
   1. 进阶：将RoiAlign作为插件集成到onnx/trt中

2. 使用流程
   
   1. 先安装环境
      
      ```
      torch                   1.10.2+cu102
      torchvision             0.11.3+cu102
      vs2019
      ```
      - 安装detectron
        - windows
          - [Windows下安装detectron2(免修改版本）_微雨曳荷的博客-CSDN博客_detectron2 windows](https://blog.csdn.net/weixin_44226805/article/details/126017177)
        - linux
          - 安装官方命令即可。

3- 导出onnx
   
   ```
   运行 mask_demo.py 导出onnx。
      
   ```

4. 生成engine
   
   ```
   trtexec --onnx=./mask.onnx --saveEngine=./mask_fp32.engine --workspace=1000
   ```

5. 运行

```
执行trt_py.py脚本
```



# 引用

[官方yolov7-mask repo](https://github.com/TexasInstruments/edgeai-yolov5/tree/yolo-pose)

</details>
