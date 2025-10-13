# zxz_调试配置文件
# zxz_用于控制是否启用调试模式（使用少量数据进行快速调试）

# zxz_调试模式开关
IS_DEBUG_MODE = True  # zxz_设置为 False 来使用完整数据集

# zxz_调试模式下使用的数据比例
DEBUG_DATA_RATIO = 0.1  # zxz_使用10%的数据

# zxz_调试模式下的训练参数优化
DEBUG_PARAMS = {
    'epochs': 5,              # zxz_大幅减少训练轮次，从默认60轮减到5轮
    'batch_size_train': 2,    # zxz_减小训练批次大小，降低显存占用
    'batch_size_test': 2,     # zxz_减小测试批次大小
    'num_threads': 1,         # zxz_减少数据加载进程数
    'lr': 0.005,              # zxz_适当调大学习率，在少量轮次内看到损失下降
    'lr_drop': 3,             # zxz_相应调整学习率衰减周期
}

print(f"zxz_调试配置已加载: DEBUG_MODE={IS_DEBUG_MODE}, DATA_RATIO={DEBUG_DATA_RATIO}")
if IS_DEBUG_MODE:
    print("zxz_调试参数优化:", DEBUG_PARAMS)
