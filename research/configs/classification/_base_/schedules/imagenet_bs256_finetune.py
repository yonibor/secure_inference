# optimizer
optimizer = dict(type='SGD', lr=0.005, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(policy='step', step=[40, 45], warmup='linear', warmup_ratio=0.01, warmup_iters=1, warmup_by_epoch=True)
runner = dict(type='EpochBasedRunner', max_epochs=50)