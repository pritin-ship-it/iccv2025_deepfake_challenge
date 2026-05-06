paths = dict(
    train_csv="/dataset/train_matched_09_th.csv",
    train_protocol="/dataset/Protocol-train.txt",
    val_protocol="/dataset/Protocol-val-labels.txt",
    root_dir="/data",
    val_bbox_csv="/dataset/val_bbox.csv"
)

device = "cuda"
model_input_size = (224, 224)
train_multihead = False
custom_model = False
model_type = "default"
cutmix_prob = 0.3
cutmix_alpha = 0.5920027694798511
aug_strength = 1

wandb = dict(
    use=True,
    project_name="CVPR_2025_v1",
    experiment_name="two_gpus",
)

dataset = dict(
    oversample_live=False,
    val_max_per_subclass=300,
    val_seed=42,
    epoch_fraction=0.5
)

train_dataloader = dict(
    batch_size=32,
    shuffle=True,
    num_workers=16,
    drop_last=True,
    pin_memory=False,
    prefetch_factor=5
)

val_dataloader = dict(
    batch_size=32,
    shuffle=False,
    num_workers=16,  
    drop_last=False
)

trainer = dict(
    type='Trainer',
    epochs=20,
    weights_save_folder='weights'
)

model = dict(
    model_name='convnextv2_tiny',
    pretrained=True,
    in_chans=3,
    num_classes=1
)

loss = dict(
    type="binary_focal",
    alpha=0.5282378118784655,
    gamma=0.7297946288723869,
    contrastive=True,
    supcon_weight=0.30580399026452565,
    supcon_temperature=0.1410185185099283,
    supcon_proj_dim=128,
)

optimizer = dict(
    use_separate_lr=False,
    type='AdamW',
    lr=1e-4,
    weight_decay=1.107519499003134e-05,
)

scheduler = dict(
    type='Cosine',
    cycle_mult=1.0,
    max_lr=0.00018174020393590877,
    min_lr=6.783490509214686e-07,
    warmup_steps=0,
    gamma=1.0
)