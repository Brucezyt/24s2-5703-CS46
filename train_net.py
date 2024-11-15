#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.
"""
A main training script.

This scripts reads a given config file and runs the training or evaluation.
It is an entry point that is made to train standard models in detectron2.

In order to let one script support training of many models,
this script contains logic that are specific to these built-in models and therefore
may not be suitable for your own project.
For example, your research project perhaps only needs a single "evaluator".

Therefore, we recommend you to use detectron2 as an library and take
this file as an example of how to use the library.
You may want to write your own script with your datasets and other customizations.
"""

import logging
import os
from collections import OrderedDict

from register_datasets import register_my_dataset
register_my_dataset()

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, hooks, launch
from detectron2.evaluation import (
    CityscapesInstanceEvaluator,
    CityscapesSemSegEvaluator,
    COCOEvaluator,
    COCOPanopticEvaluator,
    DatasetEvaluators,
    LVISEvaluator,
    PascalVOCDetectionEvaluator,
    SemSegEvaluator,
    verify_results,
)
from detectron2.modeling import GeneralizedRCNNWithTTA


def build_evaluator(cfg, dataset_name, output_folder=None):
    """
    Create evaluator(s) for a given dataset.
    This uses the special metadata "evaluator_type" associated with each builtin dataset.
    For your own dataset, you can simply create an evaluator manually in your
    script and do not have to worry about the hacky if-else logic here.
    """
    if output_folder is None:
        output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
    evaluator_list = []
    evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
    if evaluator_type in ["sem_seg", "coco_panoptic_seg"]:
        evaluator_list.append(
            SemSegEvaluator(
                dataset_name,
                distributed=True,
                output_dir=output_folder,
            )
        )
    if evaluator_type in ["coco", "coco_panoptic_seg"]:
        evaluator_list.append(COCOEvaluator(dataset_name, output_dir=output_folder))
    if evaluator_type == "coco_panoptic_seg":
        evaluator_list.append(COCOPanopticEvaluator(dataset_name, output_folder))
    if evaluator_type == "cityscapes_instance":
        return CityscapesInstanceEvaluator(dataset_name)
    if evaluator_type == "cityscapes_sem_seg":
        return CityscapesSemSegEvaluator(dataset_name)
    elif evaluator_type == "pascal_voc":
        return PascalVOCDetectionEvaluator(dataset_name)
    elif evaluator_type == "lvis":
        return LVISEvaluator(dataset_name, output_dir=output_folder)
    if len(evaluator_list) == 0:
        raise NotImplementedError(
            "no Evaluator for the dataset {} with the type {}".format(dataset_name, evaluator_type)
        )
    elif len(evaluator_list) == 1:
        return evaluator_list[0]
    return DatasetEvaluators(evaluator_list)

class EvalAndSaveHook(hooks.HookBase):
    def __init__(self, eval_period, trainer, cfg):
        self.eval_period = eval_period
        self.trainer = trainer
        self.cfg = cfg
        self.eval_results = []  # 用于存储每个 epoch 的验证结果

    def after_epoch(self):
        # 只有在 epoch 结束时才进行验证
        if (self.trainer.iter + 1) % self.eval_period == 0:
            res = self.trainer.test(self.cfg, self.trainer.model)
            self.eval_results.append(res)  # 保存验证结果
            
            # 获取不同 AP 值并记录到相应的列表
            self.trainer.epoch_ap_50_95_all.append(res["bbox"]["AP"])           # AP@0.50:0.95
            self.trainer.epoch_ap_50_all.append(res["bbox"]["AP50"])            # AP@0.50
            self.trainer.epoch_ap_75_all.append(res["bbox"]["AP75"])            # AP@0.75
            self.trainer.epoch_ap_50_95_small.append(res["bbox"]["APs"])        # AP@0.50:0.95 small
            self.trainer.epoch_ap_50_95_medium.append(res["bbox"]["APm"])       # AP@0.50:0.95 medium
            self.trainer.epoch_ap_50_95_large.append(res["bbox"]["APl"])        # AP@0.50:0.95 large
            
            # 计算并记录每个 epoch 的平均训练损失
            avg_train_loss = sum(self.trainer.losses["train"]) / len(self.trainer.losses["train"])
            self.trainer.epoch_train_losses.append(avg_train_loss)
            # 清空 self.trainer.losses["train"]，为下一 epoch 做准备
            self.trainer.losses["train"].clear()

    def after_train(self):
        # 在训练结束时输出所有验证结果
        for i, result in enumerate(self.eval_results):
            print(f"Epoch {i+1} Evaluation Results: {result}")
            
class FinalConfigSaveHook(hooks.HookBase):
    def __init__(self, cfg, output_dir):
        self.cfg = cfg
        self.output_dir = output_dir

    def after_train(self):
        # 在训练结束时保存最终配置文件
        config_path = os.path.join(self.output_dir, "final_config.yaml")
        with open(config_path, "w") as f:
            f.write(self.cfg.dump())
        print(f"Final configuration saved at {config_path}")
        
        
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from pycocotools.cocoeval import COCOeval
from pycocotools.coco import COCO

class MetricsVisualizationHook(hooks.HookBase):
    def __init__(self, cfg, trainer, losses, output_dir="output_metrics"):
        self.cfg = cfg
        self.trainer = trainer
        self.losses = losses  # 用于记录训练过程中的损失
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.map_per_epoch = []  # 用于记录每个 epoch 的 mAP

    def after_train(self):
        
        # 创建 COCOEvaluator 以执行推理和评估
        evaluator = COCOEvaluator(dataset_name=self.cfg.DATASETS.TEST[0], tasks=["bbox"], distributed=False, output_dir=self.output_dir)
        val_loader = self.trainer.build_test_loader(self.cfg, self.cfg.DATASETS.TEST[0])
        
        # 使用 inference_on_dataset 进行推理和评估
        inference_on_dataset(self.trainer.model, val_loader, evaluator)

        # 使用绝对路径加载验证集注释文件
        annotation_file_path = "/root/autodl-tmp/dataset/val.json"
        
        # 加载 ground truth 和生成的检测结果
        coco_gt = COCO(annotation_file_path)  # 使用验证集注释文件的绝对路径
        coco_dt = coco_gt.loadRes(os.path.join(self.output_dir, "coco_instances_results.json"))

        # 创建 COCOeval 对象并计算所有的 AP 和 AR
        coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        # 提取 COCOeval 的 AP 和 AR 结果
        stats = coco_eval.stats  # stats 包含所有评估指标

        # 绘制每个 epoch 的平均损失曲线
        plt.figure()
        plt.plot(range(len(self.trainer.epoch_train_losses)), self.trainer.epoch_train_losses, label="Train Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Average Loss")
        plt.legend()
        plt.title("Training Loss per Epoch")
        plt.savefig(f"{self.output_dir}/loss_curve.png")
        plt.close()
        
        # 绘制不同 IoU 阈值下的 AP 随 epoch 变化的折线图
        plt.figure()
        epochs = range(len(self.trainer.epoch_ap_50_95_all))
        plt.plot(epochs, self.trainer.epoch_ap_50_95_all, label="AP@0.50:0.95")
        plt.plot(epochs, self.trainer.epoch_ap_50_all, label="AP@0.50")
        plt.plot(epochs, self.trainer.epoch_ap_75_all, label="AP@0.75")
        plt.xlabel("Epoch")
        plt.ylabel("Average Precision (AP)")
        plt.title("AP at Different IoU Thresholds per Epoch")
        plt.legend()
        plt.savefig(f"{self.output_dir}/ap_iou_thresholds.png")
        plt.close()

        # 绘制不同目标大小的 AP 随 epoch 变化的折线图
        plt.figure()
        plt.plot(epochs, self.trainer.epoch_ap_50_95_small, label="AP@0.50:0.95 Small")
        plt.plot(epochs, self.trainer.epoch_ap_50_95_medium, label="AP@0.50:0.95 Medium")
        plt.plot(epochs, self.trainer.epoch_ap_50_95_large, label="AP@0.50:0.95 Large")
        plt.xlabel("Epoch")
        plt.ylabel("Average Precision (AP)")
        plt.title("AP at Different Object Sizes per Epoch")
        plt.legend()
        plt.savefig(f"{self.output_dir}/ap_object_sizes.png")
        plt.close()

        # 绘制 AP 条形图（不同 IoU 阈值和目标大小下的平均精度）
        try:
            plt.figure()
            # 从 stats 中提取需要的 AP 值
            ap_values = [
                stats[0],  # AP@0.50:0.95 for all sizes
                stats[1],  # AP@0.50 for all sizes
                stats[2],  # AP@0.75 for all sizes
                stats[3],  # AP@0.50:0.95 for small objects
                stats[4],  # AP@0.50:0.95 for medium objects
                stats[5]   # AP@0.50:0.95 for large objects
            ]
            labels = ["AP@0.50:0.95", "AP@0.50", "AP@0.75", "AP-Small", "AP-Medium", "AP-Large"]
    
            # 绘制条形图
            plt.bar(labels, ap_values)
            plt.xlabel("Metrics")
            plt.ylabel("Average Precision (AP)")
            plt.title("AP at Different IoU Thresholds and Object Sizes")
            plt.xticks(rotation=45, ha="right")
            plt.savefig(f"{self.output_dir}/ap_thresholds_and_sizes.png")
            plt.close()
        except KeyError as e:
            print(f"KeyError encountered while accessing AP at specific IoU thresholds and object sizes: {e}")


        # 绘制 AR 条形图（不同 maxDets 和目标大小下的平均召回率）
        try:
            plt.figure()
            # 从 stats 中提取 AR 值
            ar_values = [
                stats[6],  # AR@1 for all sizes
                stats[7],  # AR@10 for all sizes
                stats[8],  # AR@100 for all sizes
                stats[9],  # AR@100 for small objects
                stats[10], # AR@100 for medium objects
                stats[11]  # AR@100 for large objects
            ]
    
            labels = ["AR@1", "AR@10", "AR@100", "AR-Small", "AR-Medium", "AR-Large"]
    
            # 绘制条形图
            plt.bar(labels, ar_values)
            plt.xlabel("Metrics")
            plt.ylabel("Average Recall (AR)")
            plt.title("AR at Different Max Detections and Object Sizes")
            plt.xticks(rotation=45, ha="right")
            plt.savefig(f"{self.output_dir}/ar_max_dets_and_sizes.png")
            plt.close()
        except KeyError as e:
            print(f"KeyError encountered while accessing AR for different maxDets and object sizes: {e}")

import torch
import torch.nn.functional as F
from detectron2.engine import DefaultTrainer

class WeightedTrainer(DefaultTrainer):
    
    def __init__(self, cfg):
        self.losses = {"train": []}  # 初始化用于记录损失的字典
        super().__init__(cfg)
        # 设置类别权重：假设类别 0 为 "Ship"，类别 1 为 "Non-ship object"
        self.class_weights = torch.tensor([1.5, 1.0]).to(self.model.device)  # Ship 权重为 2.0
        self.epoch_train_losses = []  # 用于记录每个 epoch 的平均训练损失
        
        # 初始化用于记录每个 epoch 评估结果的属性
        self.epoch_ap_50_95_all = []
        self.epoch_ap_50_all = []
        self.epoch_ap_75_all = []
        self.epoch_ap_50_95_small = []
        self.epoch_ap_50_95_medium = []
        self.epoch_ap_50_95_large = []

    def run_step(self):
        """
        重写 run_step 方法以记录每个 iteration 的 train_loss
        """
        
        # 先确保父类的 _data_loader_iter 初始化完成
        if not hasattr(self, "_data_loader_iter"):
            self._data_loader_iter = iter(self.data_loader)
        
        assert self.model.training, "[Trainer] Model was not set to training mode!"

        # 获取数据并前向传播
        data = next(iter(self.data_loader))
        loss_dict = self.model(data)
        
        # 修改分类损失函数以包含类别权重
        if "loss_cls" in loss_dict:
            # 使用带权重的 Cross Entropy Loss 替代默认的损失
            targets = torch.cat([d["instances"].gt_classes for d in data], dim=0)  # 获取目标类别
            
            # 将权重应用到损失上
            weights = self.class_weights[targets]  # 每个实例对应的权重
            loss_cls = loss_dict["loss_cls"] * weights  # 加权后的分类损失
            loss_dict["loss_cls"] = loss_cls.mean()  # 计算加权损失的平均值
            
        losses = sum(loss_dict.values())

        # 优化器步骤
        self.optimizer.zero_grad()
        losses.backward()
        self.optimizer.step()

        # 将每个 iteration 的损失存入 self.losses["train"]
        self.losses["train"].append(losses.item())
        
         # 记录损失
        self.storage.put_scalars(**loss_dict)

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        return build_evaluator(cfg, dataset_name, output_folder)

    @classmethod
    def test_with_TTA(cls, cfg, model):
        logger = logging.getLogger("detectron2.trainer")
        # In the end of training, run an evaluation with TTA
        # Only support some R-CNN models.
        logger.info("Running inference with test-time augmentation ...")
        model = GeneralizedRCNNWithTTA(cfg, model)
        evaluators = [
            cls.build_evaluator(
                cfg, name, output_folder=os.path.join(cfg.OUTPUT_DIR, "inference_TTA")
            )
            for name in cfg.DATASETS.TEST
        ]
        res = cls.test(cfg, model, evaluators)
        res = OrderedDict({k + "_TTA": v for k, v in res.items()})
        return res
    
    def build_hooks(self):
        # 获取默认的 hooks 列表
        hooks = super().build_hooks()
        
        # 添加 EvalAndSaveHook
        hooks.insert(-1, EvalAndSaveHook(eval_period=1, trainer=self, cfg=self.cfg))
        # 添加 FinalConfigSaveHook
        hooks.append(FinalConfigSaveHook(cfg=self.cfg, output_dir=self.cfg.OUTPUT_DIR))
        # 添加 MetricsVisualizationHook，用于在训练结束时可视化损失和 mAP
        hooks.append(MetricsVisualizationHook(cfg=self.cfg, trainer=self, losses=self.losses))
        
        return hooks

from detectron2.data import DatasetCatalog

def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    
    # 通过DatasetCatalog直接获取数据集长度
    dataset_dicts = DatasetCatalog.get(cfg.DATASETS.TRAIN[0])
    dataset_length = len(dataset_dicts)
    batch_size = cfg.SOLVER.IMS_PER_BATCH
    iterations_per_epoch = dataset_length // batch_size
    desired_epochs = 2  # 设置您期望的 epoch 数量
    cfg.SOLVER.MAX_ITER = iterations_per_epoch * desired_epochs

    # 根据新的 MAX_ITER 动态设置 STEPS
    cfg.SOLVER.STEPS = (int(0.7 * cfg.SOLVER.MAX_ITER), int(0.9 * cfg.SOLVER.MAX_ITER))

    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def main(args):
    cfg = setup(args)

    if args.eval_only:
        model = WeightedTrainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = WeightedTrainer.test(cfg, model)
        if cfg.TEST.AUG.ENABLED:
            res.update(WeightedTrainer.test_with_TTA(cfg, model))
        if comm.is_main_process():
            verify_results(cfg, res)
        return res

    """
    If you'd like to do anything fancier than the standard training logic,
    consider writing your own training loop (see plain_train_net.py) or
    subclassing the trainer.
    """
    trainer = WeightedTrainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    return trainer.train()


def invoke_main() -> None:
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )


if __name__ == "__main__":
    invoke_main()  # pragma: no cover

