# zxz_调试配置文件
# zxz_用于控制是否启用调试模式（使用少量数据进行快速调试）

# zxz_调试模式开关
IS_DEBUG_MODE = True  # zxz_设置为 False 来使用完整数据集

# zxz_调试模式下使用的数据比例
DEBUG_DATA_RATIO = 0.1  # zxz_使用10%的数据

print(f"zxz_调试配置已加载: DEBUG_MODE={IS_DEBUG_MODE}, DATA_RATIO={DEBUG_DATA_RATIO}")
