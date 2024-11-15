# register_datasets.py
from detectron2.data.datasets import register_coco_instances
import os

def register_my_dataset():
    # 替换为你实际的路径
    data_path = "/root/autodl-tmp/dataset"
    
    # 注册训练集、验证集和测试集
    register_coco_instances("my_dataset_train", {}, os.path.join(data_path, "merged.json"), os.path.join(data_path,"newimage"))
    register_coco_instances("my_dataset_val", {}, os.path.join(data_path, "val.json"), os.path.join(data_path, "scaled_down_val"))
    register_coco_instances("my_dataset_test", {}, os.path.join(data_path, "test.json"), os.path.join(data_path, "scaled_down_test"))

if __name__ == "__main__":
    register_my_dataset()
