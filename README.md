# 论文计划
## 说明
每三天看一篇论文，写成不低于50字的总结
每天上传任务以及任务完成情况
添加idea模块，上传可用的idea
## 2018.7.23

### liu
#### 论文

#### 任务

完成ctpn的回归部分
#### idea


### qian

#### 论文

#### 任务
完成ctpn的回归部分

#### idea

## 2018.7.24

### liu
#### 论文

#### 任务

在多尺度上能不能做出突破
#### idea


### qian
完成ctpn的最终版本
#### 论文


#### 任务
完成ctpn的回归部分
#### idea
## 2018.7.25
### 论文
### 任务
完成倾斜长方形的回归loss,但是incident数据集是不规则长方形.如何将不规则长方形回归到倾斜长方形。或者更进一步完成倾斜平行四边形
完成incident数据集的导入解析
需要为这个自动设计文本线构造
### idea
1. 之前尝试在score map上根据字的连续性假设，加入连续性loss.在特征图上表现的更聚集，但是效果一般
2. 能不能再加入box regress上的连续性假设?,保持相邻的regress有相同的回归值
3. 二次回归，即执行两次回归。
4. 梯度的学习导流.即sum的导数流向v1,v2,v3,自动学习权重学习在v1,v2,v3之间的梯度分配
5. deformable conv and deformable pool

## 2018.7.26
### 论文
End to end
### 任务
完成倾斜
倾斜的nms和文本线构造需要重写,traintodata还没写完,发现label转换的代码运行时间过长.
先放弃倾斜的文本，专门打coco比赛
### idea
端对端检测识别,需要nms可训练，如何把nms放在卷积层里面

## 2018.7.27
### 论文
### 任务
完成了conloss,regloss,前二层mask.但是代码好像有问题，没有feed mask,明天去看一下
### idea
用gather选取score>0.7的,用lstm完成nms算法。lstm先编码所有的anchor，然后解码成目标box
## 2018.7.31
### 论文
BN
### 任务
看了tensorflow 自定义op怎么写,初步完成了多batch
### idea
基于深度学习的目标检测算法综述（二）（分享自知乎网）https://zhuanlan.zhihu.com/p/40020809?utm_source=qq&utm_medium=social&utm_oi=50987419041792
我的手机 2018/7/23 星期一 下午 11:54:13
关于感受野的总结（分享自知乎网）https://zhuanlan.zhihu.com/p/40267131?utm_source=qq&utm_medium=social&utm_oi=50987419041792
我的手机 2018/7/27 星期五 下午 6:13:23
量子位 - 汤晓鸥为CNN搓了一颗大力丸（分享自知乎网）https://zhuanlan.zhihu.com/p/40681613?utm_source=qq&utm_medium=social&utm_oi=50987419041792
我的手机 2018/7/31 星期二 下午 10:22:18
二次回归，直接用scoremap和redress map再次回归出正确的掩码
我的手机 2018/7/31 星期二 下午 10:22:19
用nms算法构造lstm的输出label
我的手机 2018/7/31 星期二 下午 10:22:19
集成，深度，重复思考
我的手机 2018/7/31 星期二 下午 10:22:19
多个anchor不就是集成方法吗？
我的手机 2018/7/31 星期二 下午 10:22:19
加强感受野位置的编码
我的手机 2018/7/31 星期二 下午 10:22:19
对hard区域进行掩码再训练
我的手机 2018/7/31 星期二 下午 10:22:19
怎么在神经网络中加入联想机制
我的手机 2018/7/31 星期二 下午 10:22:19
联想和记忆
我的手机 2018/7/31 星期二 下午 10:22:19
densenet
我的手机 2018/7/31 星期二 下午 10:22:20
趋同性的loss其实很难说学到了位置的相关性。因为相邻卷积模板是相同的。在预测时，并未获得两个点是否是相邻的信息，所以也无法趋同。但是这个loss确实有效果，我猜更多的是模板内秉的性质即放松了特征的差距，而非推理的性质
我的手机 2018/7/31 星期二 下午 10:22:20
怎样在卷积网络中实现仿射变换
我的手机 2018/7/31 星期二 下午 10:22:20
如果把二维坐标做为两层输进去，会怎么样
我的手机 2018/7/31 星期二 下午 10:22:20
本质上，检测就是输出一个掩码。从理论上来讲score map就是掩码。但是score map学习的不好。所以需要另外的loss box来进一步确定。从这个角度来讲都是基于投票，与集成方法相同
我的手机 2018/7/31 星期二 下午 10:22:20
只不过box回归加入了更强的长方形假设
我的手机 2018/7/31 星期二 下午 10:22:20
dialated convolutions
我的手机 2018/7/31 星期二 下午 10:22:20
它们的区别是什么？
我的手机 2018/7/31 星期二 下午 10:22:20
lstm在二维上无法体现二维的空间距离信息
我的手机 2018/7/31 星期二 下午 10:22:21
能不能借用bn的思想，对图像做某种校正，然后再还原
我的手机 2018/7/31 星期二 下午 10:22:21
多层掩码。即输出多个掩码层，对一个图像进行多次掩码，形成多个图像再次输入到第一层中
我的手机 2018/7/31 星期二 下午 10:22:21
本质上在用卷积学习nms
我的手机 2018/7/31 星期二 下午 10:22:21
特征提取，推理
我的手机 2018/7/31 星期二 下午 10:22:21
语义分割中，稀疏卷积核的使用
我的手机 2018/7/31 星期二 下午 10:22:21
两种方法加强泛性。一种将图像变换到统一的范式，第二种将图像进行联想变形，作为额外的数据集训练。从某种角度来看这两种方法差不多。第三种方法是二次思考
我的手机 2018/7/31 星期二 下午 10:22:21
机器之心 - 循环神经网络不需要训练？复现「世界模型」的新发现（分享自知乎网）https://zhuanlan.zhihu.com/p/38744193?utm_source=qq&utm_medium=social&utm_oi=50987419041792

### 问题
在对图像进行多层掩码时，如何将一张图像掩码成多张图像？学习到的掩码怎么变换