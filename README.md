# Mask_RCNN

Keras实现Mask_RCNN主要参考[matterport/Mask_RCNN](https://github.com/matterport/Mask_RCNN)，里面已经做了非常详细的介绍以及提供预训练权重的下载。

## 文件说明
1. contourprocess

    建筑物掩膜的处理

2. largest interior rectangle

    最大内接矩形算法(largest_interior_rectangle)：https://github.com/lukasalexanderweber/lir
    
    ![image](https://user-images.githubusercontent.com/27406337/166854679-18a8a8ae-70ed-4248-8971-4a8c8875f89d.png)![image](https://user-images.githubusercontent.com/27406337/166854694-9c6d10b0-6d58-480c-9295-a502f1efc687.png)![image](https://user-images.githubusercontent.com/27406337/166854704-29f8be81-6e16-48c2-8434-61bc44d88bda.png)

3. 道格拉斯普克算法

运行```DouglasPeuker.py```即可


4. 关于config文件

config文件包含两个，一个是在```utils```文件夹下的```config.py```，这一配置文件是训练以及推理过程中的基础设置的配置文件，里面包括基础的配置。另一个是根目录下的```config```文件，继承```utils```文件夹下的```config.py```，主要配置训练集，测试图像等路径。

## 训练

训练的时候注意一点：GPU数量暂时只能设定为1，如果需要多块GPU进行并行训练，可查看[matterport/Mask_RCNN](https://github.com/matterport/Mask_RCNN)中有并行训练模型的代码。

训练数据的结构遵循VOC格式，包含三个文件夹：```imgs```、```mask```以及```yaml```文件夹，分别存放原图、掩膜图像以及yaml文件，如何制作数据集请参考：[Semantic-Segmentation-Datasets](https://github.com/RyanCCC/Semantic-Segmentation-Datasets)

![image](https://user-images.githubusercontent.com/27406337/166855198-f3761b2c-e0aa-4b51-bdd6-55ff66eb93c3.png)

制作好数据集后，修改```config.py```文件，运行```train.py```文件即可进行训练。

## 推理

修改config文件后运行```inference.py```文件即可。

ONNX推理可以参考：[Deployment:MaskRCNN](https://github.com/RyanCCC/Deployment/tree/main/MaskRCNN)

## 模型评估

1. [Calculating mean Average Recall (mAR), mean Average Precision (mAP) and F1-Score](https://github.com/matterport/Mask_RCNN/issues/2513)

2. [compue_ap](https://github.com/matterport/Mask_RCNN/blob/master/mrcnn/utils.py#L715)

3. [语义分割之评价指标](https://zhuanlan.zhihu.com/p/61880018 )

## 模型转换

使用`convert.py`可以进行模型转换。如将权重和计算图转换成`Tensorflow saved_model`的格式。也可以将tensorflow的模型转换成ONNX的格式。详情请参考文档。
一些简单的使用例子如下：`python .\convert.py --flag --saved_model .\save_model\ --save_onnx ./test.onnx`或者`python .\convert.py --weight .\model\village_building.h5 --label .\data\building.names --save_onnx ./test.onnx --saved_pb`

## 调试问题

1. `Input image dtype is bool. Interpolation is not defined with bool data type`

![image](https://user-images.githubusercontent.com/27406337/174923521-cf42e985-007b-44ef-b00c-20d78a95ac1e.png)

参考：[Input image dtype is bool. Interpolation is not defined with bool data type](https://stackoverflow.com/questions/62330374/input-image-dtype-is-bool-interpolation-is-not-defined-with-bool-data-type):`pip install -U scikit-image==0.16.2`



## 参考

1. [Mask RCNN源码解读](https://blog.csdn.net/u012655441/article/details/122304099)
2. [Mask RCNN综述以及建筑物实例分割](https://blog.csdn.net/u012655441/article/details/120753214)
