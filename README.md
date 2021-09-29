# Mask-RCNN
Mask-RCNN原理以及实现

### Mask-RCNN

Mask-RCNN主要完成三件事情：1. 目标检测（直接在结果上绘制了目标框）；2. 目标分类；3. 像素级目标分割。Mask-RCNN继承的是Faster-RCNN,在Faster-RCNN基础上添加了Mask Prediction Branch（MASK预测分支），并且改良了ROI Pooling，提出了ROI Align。

##### 网络结构

![image](https://user-images.githubusercontent.com/27406337/131059783-b8489815-6e51-4250-b30c-48acf4ec7af5.png)

![image](https://user-images.githubusercontent.com/27406337/131060576-f3b1ccdd-9c9d-482e-bce1-a4765d818b79.png)

##### Faster-RCNN

Faster-RCNN使用CNN提取图像特征，然后使用region proposal network（RPN）去提取出ROI，然后使用ROI pooling将这些ROI全部编程固定尺寸，再喂给全连接层进行Bounding box回归和分类预测。

![image](https://user-images.githubusercontent.com/27406337/131062048-a3a8cd8a-a031-4f81-b1f8-7b4f42095855.png)

- Conv layers。作为一种CNN网络目标检测方法，Faster RCNN首先使用一组基础的conv+relu+pooling层提取image的feature maps。该feature maps被共享用于后续RPN层和全连接层。
- Region Proposal Networks。RPN网络用于生成region proposals。该层通过softmax判断anchors属于positive或者negative，再利用bounding box regression修正anchors获得精确的proposals。
- Roi Pooling。该层收集输入的feature maps和proposals，综合这些信息后提取proposal feature maps，送入后续全连接层判定目标类别。
- Classification。利用proposal feature maps计算proposal的类别，同时再次bounding box regression获得检测框最终的精确位置。

anchor：

![image](https://user-images.githubusercontent.com/27406337/131062534-d5706f05-ec32-4a40-8db4-1eb9c222ff83.png)


[一文读懂Faster RCNN](https://zhuanlan.zhihu.com/p/31426458)


##### ResNet-FPN

![image](https://user-images.githubusercontent.com/27406337/131063269-6973b0c0-9ba4-4713-8408-1f3d5d4754cd.png)

##### ResNet-FPN+Fat RCNN+mask

![image](https://user-images.githubusercontent.com/27406337/131063513-c6df014c-66cc-4070-9a67-5d2acc4da233.png)

##### ROI Align

实际上，Mask RCNN中还有一个很重要的改进，就是ROIAlign。Faster R-CNN存在的问题是：特征图与原始图像是不对准的（mis-alignment），所以会影响检测精度。而Mask R-CNN提出了RoIAlign的方法来取代ROI pooling，RoIAlign可以保留大致的空间位置。

##### Loss

![image](https://user-images.githubusercontent.com/27406337/131063639-47362d5d-e826-4274-9034-57769b351fdb.png)


### 代码实现实现

主要参考[Mask R-CNN Code](https://github.com/matterport/Mask_RCNN)，详情请查看Keras分支。另外目前在tensorflow2实现。

### 参考

1. [Mask R-CNN](http://cn.arxiv.org/pdf/1703.06870v3)

2. [Mask R-CNN Code](https://github.com/matterport/Mask_RCNN)
