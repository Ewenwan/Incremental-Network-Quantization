# INQ Incremental-Network-Quantization 渐进式神经网络量化
 # 给定任意结构的全精度浮点神经网络模型，能将其转换成无损的低比特二进制模型
 # 英特尔中国研究院：INQ神经网络无损低比特量化技术 全精度网络输入，输出权值为0或2的整数次幂的网络
![](https://xmfbit.github.io/img/paper-inq-quantize-set.png)
 [代码](https://github.com/Ewenwan/Incremental-Network-Quantization)
 [论文](https://arxiv.org/pdf/1702.03044.pdf)
 [参考1](http://zhidx.com/p/94098.html)
 [参考2](https://xmfbit.github.io/2018/01/25/inq-paper/)
 # 量化策略 Method
* 提出了渐进式神经网络量化的思想，引入了三种操作：参数分组，量化，重训练
       简单的说就是在训练得到一个网络模型后，
      首先将这个全精度浮点网络模型中的每一层参数分为两组，
      第一组中的参数直接被量化固定，
      另一组参数通过重训练来补偿量化给模型造成的精度损失。
      然后这三个操作依次迭代应用到刚才的第二组完成重训练之后的全精度浮点参数部分，直到模型全部量化为止。
      
      可以说参数分组分成的这两个部分是互补的作用，
      一个建立低精度模型的基础，
      另一个通过retrain(重训练，微调)补偿精度损失；
      这样迭代最终得到渐进式的量化和精度提升。
      通过巧妙耦合参数分组、量化和重训练操作，该技术抑制了模型量化造成的性能损失，从而在实际中适用于任意结构的神经网络模型。
      
      INQ渐进式网络量化策略：
![](http://zhidx.com/wp-content/uploads/2017/09/85fd56a6c52852178bcb2e3e79681ca%E5%89%AF%E6%9C%AC.png)
      
      （绿线代表当前已经被量化的网络连接；蓝线代表需要重新训练的网络连接）
      
# 具体量化方式
    该技术还包含另外两个亮点。
    其一，在模型量化过程中，所有参数被限制成二进制表示，并包含零值，极限量化结果即为三值网络或者二值网络。
         这种量化使得最后的模型非常适合在硬件上部署和加速。
         比如在FPGA上，复杂的全精度浮点乘法运算将被直接替换为简单的移位操作。
    其二，现有神经网络量化压缩方法在处理二值网络或者三值网络时，为了让模型精度损失不至于太大，
         往往将模型第一层和最后一层参数依然保留为全精度浮点型，
         而我们的技术在对模型的所有参数进行量化的同时，实现了性能的全面领先 。
         
![](http://zhidx.com/wp-content/uploads/2017/09/6a56b64514919da3da833874edc60a8%E5%89%AF%E6%9C%AC.png)
      
            INQ渐进式网络量化示例
             第一行：依次为参数分组、量化与重训练；
             第二行：迭代过程
             （绿色区域代表当前已经被量化的网络参数；浅紫区域代表需要重新训练的网络参数）
 # Incremental-Network-Quantization
Caffe Implementation for Incremental network quantization, we modify the original caffe, the Installation is follow [caffe](http://caffe.berkeleyvision.org/install_apt.html).
 the default source code is 5 bits weights-only quantization,
 量化位数精度修改
 you can by changing parameter "partition"(/src/caffe/blob.cpp) to control the quantization step. 
 # INQ usage
	0. 编译
	   依赖fft  sudo apt-get install libfftw3-dev libfftw3-doc 
 	0.you must be farmilar with caffe training [imagenet tutorial](http://caffe.berkeleyvision.org/gathered/examples/imagenet.html)
 	1. 默认为5bits的量化精度 Train 5 bits Alexnet with Imagenet:
 		python run.py
 	Please download float-point ImageNet-pre-trained AlexNet/VGG models and power-of-two model manually from [BaiduYun](https://pan.baidu.com/s/1qYHkbus), and put it into $/models/bvlc_alexnet/.
 	2.At continuous partition steps, the output logs are saved as run1_log.out, run2_log.out, run3_log.out,..., respectively
 ### Citing INQ
 If you find INQ useful in your research, please consider citing:
 	@inproceedings{zhou2017,
	title={Incremental Network Quantization: Towards Lossless CNNs with Low-Precision Weights},
	author={Aojun Zhou, Anbang Yao, Yiwen Guo, Lin Xu, Yurong Chen},
	booktitle={International Conference on Learning Representations,ICLR2017},
	year={2017},
	}
	
 ## Tips:
 1. Real-time data shuffling is useful
 ## Experimental Results
 The authors adopted the proposed method to several model, including AlexNet, VGG-16, GoogleNet, ResNet-18 and ResNet-50. More experiments for exploration was conducted on ResNet-18. Experimental results on ImageNet using center crop validation are shown as follows.
 | Network       | Bit-width | Top-1/Top-5 Error | Decrease in Top-1/Top-5 Error | Portion                   |
| ------------- | :-------- | ----------------- | ----------------------------- | ------------------------- |
| AlexNet ref   | 32        | 42.71%/19.77%     |                               |                           |
| AlexNet       | 5         | **42.61%/19.54%** | 0.15%/0.23%                   | {0.3, 0.6, 0.8, 1.0}      |
| VGG-16 ref    | 32        | 31.46%/11.35%     |                               |                           |
| VGG-16        | 5         | **29.18%/9.70%**  | 2.28%/1.65%                   | {0.5, 0.75, 0.875, 1.0}   |
| GoogleNet ref | 32        | 31.11%/10.97%     |                               |                           |
| GoogleNet     | 5         | **30.98%/10.72%** | 0.13%/0.25%                   | {0.2, 0.4, 0.6, 0.8, 1.0} |
| ResNet-18 ref | 32        | 31.73%/11.31      |                               |                           |
| ResNet        | 5         | **31.02%/10.90%** | 0.71%/0.41                    | {0.5, 0.75, 0.875, 1.0}   |
| ResNet-50 ref | 32        | 26.78%/8.76%      |                               |                           |
| ResNet-50     | 5         | **25.19%/7.55%**  | 1.59%/1.21%                   | {0.5, 0.75, 0.875, 1.0}   |
       Number of required epochs for training increasing with the expected bit-width going down.
      The accumulated portions for weight quantization are set as {0.3, 0.5, 0.8, 0.9, 0.95, 1.0}, 
      {0.2, 0.4, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0}, {0.2, 0.4, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.975,  1.0} 
      for 4-bits to 2-bits, respectively. Training epochs required for 2-bits finally 
      set to 30 which means that 300 training epochs are required for completing a full quantization procedure.
      In the other words, the proposed method become time-consuming when the network going deeper.
       Although the authors convert weights to the powers of 2 and claim that their 
      method would be efficient with binary shift operation in hardware, 
      the computation in there experiments is still using floating operations.
      Thus they only show the results of model compression instead of speeding up computation.
# 训练量化 run.py
 ```python
#-*- coding: utf-8 -*-

# 分组多次量化

import os

# 对第一组量化=========================
print "First partition and run"

os.system("nohup sh ./examples/INQ/alexnet/train_alexnet.sh >run1_log.out 2>&1")

# 对第二组量化=========================
print "Second partition and run"

# 修改源文件 编译
os.system("sed -i \"s/(count_\*0.7)/(count_\*0.4)/g\" ./src/caffe/blob.cpp")
os.system("make all -j128")

# 进行训练量化
# 前一个阶段的模型文件
os.system("sed -i \"s/bvlc_alexnet.caffemodel/alexnet_part1_iter_63000.caffemodel/g\" ./examples/INQ/alexnet/train_alexnet.sh")
# 保存下一个阶段的模型文件
os.system("sed -i \"s/part1/part2/g\" ./examples/INQ/alexnet/solver.prototxt")
os.system("nohup sh ./examples/INQ/alexnet/train_alexnet.sh >run2_log.out 2>&1")


# 对第三组量化=========================
print "Thrid partition and run"
# 修改源文件 编译
os.system("sed -i \"s/(count_\*0.4)/(count_\*0.2)/g\" ./src/caffe/blob.cpp")
os.system("make all -j128")

# 进行训练量化
os.system("sed -i \"s/alexnet_part1_iter_63000.caffemodel/alexnet_part2_iter_63000.caffemodel/g\" ./examples/INQ/alexnet/train_alexnet.sh")
os.system("sed -i \"s/part2/part3/g\" ./examples/INQ/alexnet/solver.prototxt")
os.system("nohup sh ./examples/INQ/alexnet/train_alexnet.sh >run3_log.out 2>&1")

# 最后一组量化==========================
print "Last partition and run"

os.system("sed -i \"s/(count_\*0.2)/(count_\*0.)/g\" ./src/caffe/blob.cpp")
os.system("make all -j128")

os.system("sed -i \"s/alexnet_part2_iter_63000.caffemodel/alexnet_part3_iter_63000.caffemodel/g\" ./examples/INQ/alexnet/train_alexnet.sh")
os.system("sed -i \"s/part3/part4/g\" ./examples/INQ/alexnet/solver.prototxt")
os.system("sed -i \"s/snapshot: 3000/snapshot: 1/g\" ./examples/INQ/alexnet/solver.prototxt")
os.system("sed -i \"s/max_iter: 63000/max_iter: 1/g\" ./examples/INQ/alexnet/solver.prototxt")
os.system("nohup sh ./examples/INQ/alexnet/train_alexnet.sh >run4_log.out 2>&1")

# 2乘方权重============================
print "All quantization done and you can enjoy the power-of-two weights using check.py!"
```


