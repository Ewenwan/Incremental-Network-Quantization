# part1 : blob.CPP  int partition=int(count_*0.7)-1;// 每次量化的比例 分界点
# 量化 前30%
WEIGHT=./models/VggNet/VGG_ILSVRC_16_layers.caffemodel

# part2 blob.CPP  int partition=int(count_*0.4)-1;// 每次量化的比例 分界点
# 量化 前60%
# solver.prototxt 保存模型修改 part1 --> part2
# WEIGHT=./models/VggNet/iNQ-model/models/vggnet16_part2_iter_63000.caffemodel

# part3  0.2
# 量化 前80%
# solver.prototxt 保存模型修改 part2 --> part3
# WEIGHT=./models/VggNet/iNQ-model/models/vggnet16_part3_iter_63000.caffemodel

# part4  0.0
# 量化 前100%
# solver.prototxt 保存模型修改 part3 --> part4
# 模型保存间隔 snapshot：3000 -> snapshot：1
# 最大迭代次数 max_iter: 63000 -> max_iter: 1
# WEIGHT=./models/VggNet/iNQ-model/models/vggnet16_part4_iter_63000.caffemodel


./build/tools/caffe train \
    --solver=./examples/INQ/vggNet/solver.prototxt \
    --weights=$WEIGHT \
    --gpu=6,7

