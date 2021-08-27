# Mask-RCNN
Mask-RCNN原理以及实现

### Mask-RCNN

Mask-RCNN主要完成三件事情：1. 目标检测（直接在结果上绘制了目标框）；2. 目标分类；3. 像素级目标分割。Mask-RCNN继承的是Faster-RCNN,在Faster-RCNN基础上添加了Mask Prediction Branch（MASK预测分支），并且改良了ROI Pooling，提出了ROI Align。

##### 网络结构

![image](https://user-images.githubusercontent.com/27406337/131059783-b8489815-6e51-4250-b30c-48acf4ec7af5.png)

![image](https://user-images.githubusercontent.com/27406337/131060576-f3b1ccdd-9c9d-482e-bce1-a4765d818b79.png)

##### Faster RCNN

[一文读懂Faster RCNN](https://zhuanlan.zhihu.com/p/31426458)

### Keras实现

主要参考[Mask R-CNN Code](https://github.com/matterport/Mask_RCNN)，在Keras分支。

### 参考

1. [Mask R-CNN](http://cn.arxiv.org/pdf/1703.06870v3)

2. [Mask R-CNN Code](https://github.com/matterport/Mask_RCNN)
