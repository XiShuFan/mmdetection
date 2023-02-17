# optimizer
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)

"""
======================================== 构造hooks的配置 =========================================
"""
# 学习率调度
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=10,
    warmup_ratio=0.001,
    step=[8, 11]
)
# 反向传播+参数更新
optimizer_config = dict(grad_clip=None)
# 保存ckpt
checkpoint_config = dict(interval=1)
# 日志记录
log_config = dict(interval=1, hooks=[dict(type='TextLoggerHook')])
# 动量更新，用于3D目标检测
momentum_config = None
# 迭代一次时间统计
timer_config = dict(type='IterTimerHook')
# 自定义hook
custom_hooks = [dict(type='NumClassCheckHook')]

# eval hook，对应val数据集，优先级为 LOW
evaluation = dict(interval=1, metric='bbox', out_dir='D:/Dataset/ToothCOCO/val_pred')

"""
====================================== 构造runner配置 =================================================
"""
# 大部分情况下，按照epoch就行。不指定的话，默认EpochBasedRunner
runner = dict(type='EpochBasedRunner', max_epochs=12)

"""
====================================== workflow ======================================================
"""
# 工作流程，全部训练
workflow = [('train', 1)]

"""
======================================== 其他配置 =======================================================
"""
dist_params = dict(backend='nccl')
log_level = 'INFO'
# 加载模型checkpoint
load_from = None
# 恢复之前所有运行状态
resume_from = None

# disable opencv multithreading to avoid system being overloaded
opencv_num_threads = 0
# set multi-process start method as `fork` to speed up the training
mp_start_method = 'fork'

# Default setting for scaling LR automatically
#   - `enable` means enable scaling LR automatically
#       or not by default.
#   - `base_batch_size` = (8 GPUs) x (2 samples per GPU).
auto_scale_lr = dict(enable=False, base_batch_size=16)
