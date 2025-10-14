# zxz_调试模式使用说明

## 功能概述
为了加快代码调试速度，添加了全方位的调试模式功能：
1. 只使用前10%的数据集
2. 大幅减少训练轮次
3. 优化批次大小和学习率
4. 减少计算开销

## 使用方法

### 1. 启用调试模式（推荐）
修改 `zxz_debug_config.py` 文件：
```python
IS_DEBUG_MODE = True   # 启用调试模式
```

### 2. 关闭调试模式（完整训练）
修改 `zxz_debug_config.py` 文件：
```python
IS_DEBUG_MODE = False  # 关闭调试模式
```

### 3. 自定义调试参数
可以在 `zxz_debug_config.py` 中修改：
```python
DEBUG_PARAMS = {
    'epochs': 5,              # 训练轮次：60 -> 5
    'batch_size_train': 1,    # 训练批次：保持为1（避免维度问题）
    'batch_size_test': 1,     # 测试批次：保持为1（避免维度问题）
    'num_threads': 1,         # 数据加载进程：默认 -> 1
    'lr': 0.005,              # 学习率：0.001 -> 0.005
    'lr_drop': 3,             # 学习率衰减：30 -> 3
}
DEBUG_DATA_RATIO = 0.1        # 数据使用比例：100% -> 10%
```

## 4. zxz_使用新生成的预训练权重文件
    load_model_file = "./checkpoints/weights/2025_10_13_21:26:46_Dataset->CrackDepth_modals->_RGB_dep/checkpoint_best.pth"