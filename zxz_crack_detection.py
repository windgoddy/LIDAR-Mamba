"""
zxz_目标检测转换脚本
将语义分割结果转换为目标检测任务
Author: 改造版本
"""

import cv2
import numpy as np
import torch
import argparse
import os
import json
from datasets import create_dataset
from models import build_model
from main import get_args_parser
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

class CrackDetector:
    def __init__(self, model_path, args):
        """初始化检测器"""
        self.args = args
        self.device = torch.device(args.device)
        
        # 加载模型
        self.model, self.criterion = build_model(args)
        self.model.cuda()
        self.model.eval()
        
        # 加载权重
        state_dict = torch.load(model_path)
        self.model.load_state_dict(state_dict["model"])
        print(f"zxz_模型加载成功: {model_path}")

    def segment_to_boxes(self, mask, min_area=100):
        """
        从分割mask中提取边界框
        Args:
            mask: 二值化的分割结果 (H, W)
            min_area: 最小区域面积阈值
        Returns:
            boxes: [(x1, y1, x2, y2, confidence, area), ...]
        """
        # 确保是二值化mask
        binary_mask = (mask > 0.5).astype(np.uint8) * 255
        
        # 查找连通区域
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        boxes = []
        for i, contour in enumerate(contours):
            # 计算区域面积
            area = cv2.contourArea(contour)
            if area < min_area:
                continue
                
            # 获取边界框
            x, y, w, h = cv2.boundingRect(contour)
            x1, y1, x2, y2 = x, y, x + w, y + h
            
            # 计算置信度（基于区域面积和形状）
            confidence = min(1.0, area / 1000.0)  # 简单的置信度计算
            
            boxes.append({
                'bbox': [x1, y1, x2, y2],
                'confidence': confidence,
                'area': area,
                'contour': contour
            })
        
        # 按置信度排序
        boxes.sort(key=lambda x: x['confidence'], reverse=True)
        return boxes

    def detect_single_image(self, modal_imgs, scan_orders=None):
        """检测单张图像"""
        with torch.no_grad():
            # 模型推理
            outputs = self.model(modal_imgs, scan_orders)
            # 从批次中取出第一张图 (batch_size=1)，并选择第一个通道
            mask = outputs[0, 0, ...].cpu().numpy()
            
            # 从mask提取边界框
            boxes = self.segment_to_boxes(mask)
            
            return mask, boxes

    def visualize_detection(self, image, boxes, save_path=None, show_confidence=True):
        """可视化检测结果"""
        fig, ax = plt.subplots(1, figsize=(12, 8))
        
        # 显示原图
        if len(image.shape) == 3:
            ax.imshow(image)
        else:
            ax.imshow(image, cmap='gray')
        
        # 绘制边界框
        colors = ['red', 'blue', 'green', 'yellow', 'purple', 'orange']
        for i, box_info in enumerate(boxes):
            bbox = box_info['bbox']
            confidence = box_info['confidence']
            area = box_info['area']
            
            x1, y1, x2, y2 = bbox
            width = x2 - x1
            height = y2 - y1
            
            # 选择颜色
            color = colors[i % len(colors)]
            
            # 绘制边界框
            rect = patches.Rectangle((x1, y1), width, height, 
                                   linewidth=2, edgecolor=color, facecolor='none')
            ax.add_patch(rect)
            
            # 添加标签
            if show_confidence:
                label = f'Crack {i+1}\nConf: {confidence:.2f}\nArea: {area:.0f}'
            else:
                label = f'Crack {i+1}'
                
            ax.text(x1, y1-10, label, color=color, fontsize=10, 
                   bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.7))
        
        ax.set_title(f'裂缝检测结果 - 发现 {len(boxes)} 个裂缝实例')
        ax.axis('off')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"zxz_检测结果已保存: {save_path}")
        
        plt.show()
        return fig

    def run_detection(self, test_data_loader, output_dir):
        """运行目标检测"""
        os.makedirs(output_dir, exist_ok=True)
        
        detection_results = []
        modal_num = len(self.args.modals)
        
        print("zxz_开始目标检测...")
        
        for batch_idx, data in enumerate(test_data_loader):
            # 准备输入数据
            modal_imgs = []
            for i in range(modal_num):
                items = list(data.items())
                key, value = items[i]
                modal_imgs.append(value.to(self.device))
            
            # 获取原始图像用于可视化
            original_image = modal_imgs[0][0].permute(1, 2, 0).cpu().numpy()
            # 定义ImageNet的均值和标准差
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            # 反归一化
            original_image = original_image * std + mean
            original_image = np.clip(original_image, 0, 1)
            original_image = np.clip(original_image, 0, 1)
            
            # 获取扫描顺序（如果需要）
            scan_orders = None
            if hasattr(self, 'scan_list') and self.args.scan_list_json_path != 'pretrain':
                # 这里可以添加扫描顺序的处理逻辑
                pass
            
            # 执行检测
            mask, boxes = self.detect_single_image(modal_imgs, scan_orders)
            
            # 获取图像名称
            image_path = data["image_path"][0] if "image_path" in data else f"image_{batch_idx}"
            image_name = os.path.basename(image_path).split('.')[0]
            
            # 保存结果
            result = {
                'image_name': image_name,
                'num_cracks': len(boxes),
                'detections': []
            }
            
            for i, box_info in enumerate(boxes):
                result['detections'].append({
                    'crack_id': i + 1,
                    'bbox': box_info['bbox'],
                    'confidence': float(box_info['confidence']),
                    'area': float(box_info['area'])
                })
            
            detection_results.append(result)
            
            # 可视化并保存
            save_path = os.path.join(output_dir, f"{image_name}_detection.png")
            self.visualize_detection(original_image, boxes, save_path)
            
            print(f"zxz_处理完成 {batch_idx + 1}: {image_name} - 发现 {len(boxes)} 个裂缝")
            
            # 调试模式下只处理几张图片
            if batch_idx >= 4:  # 只处理前5张图片用于快速验证
                break
        
        # 保存检测结果JSON
        results_json_path = os.path.join(output_dir, "detection_results.json")
        with open(results_json_path, 'w', encoding='utf-8') as f:
            json.dump(detection_results, f, indent=2, ensure_ascii=False)
        
        print(f"zxz_检测结果已保存到: {results_json_path}")
        print(f"zxz_总共处理 {len(detection_results)} 张图像")
        
        # 统计信息
        total_cracks = sum(result['num_cracks'] for result in detection_results)
        avg_cracks = total_cracks / len(detection_results) if detection_results else 0
        print(f"zxz_检测统计: 总裂缝数={total_cracks}, 平均每图={avg_cracks:.1f}个")
        
        return detection_results

def main():
    # 解析参数
    parser = argparse.ArgumentParser('LIDAR裂缝目标检测', parents=[get_args_parser()])
    args = parser.parse_args()
    
    # 设置参数
    args.phase = 'test'
    args.batch_size = 1
    args.scan_list_json_path = 'pretrain'  # 使用预训练模式，避免扫描序列依赖
    args.modals = ['RGB', 'dep']
    
    # 权重文件路径（使用您训练好的权重）
    model_path = "./checkpoints/weights/2025_10_13_21:26:46_Dataset->CrackDepth_modals->_RGB_dep/checkpoint_best.pth"
    
    # 输出目录
    output_dir = "./detection_results"
    
    # 初始化检测器
    detector = CrackDetector(model_path, args)
    
    # 创建测试数据加载器
    test_data_loader = create_dataset(args)
    
    # 运行检测
    results = detector.run_detection(test_data_loader, output_dir)
    
    print("zxz_目标检测任务完成！")

if __name__ == '__main__':
    main()