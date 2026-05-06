paths = dict(
    train_csv="/dataset/train_matched_092_th.csv",  # updated to match Params
    train_protocol="/dataset/Protocol-train.txt",
    val_protocol="/dataset/Protocol-val-labels.txt",
    root_dir="/data",
    val_bbox_csv="/dataset/val_bboxes_rerun.csv",
)
            

device = "cuda"
model_input_size = (224, 224)
train_multihead = False
custom_model = False
model_type = "default"
cutmix_prob = 0.24
cutmix_alpha = 0.7280396718222986
aug_strength = 1

wandb = dict(
    use=True,
    project_name="CVPR_2025_v1",
    experiment_name="best_model_one_gpu",
)

dataset = dict(
    oversample_live=False,
    val_max_per_subclass=300,
    val_seed=42,
    epoch_fraction=0.5,
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
    epochs=7,  
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
    alpha=0.6211711901839025,  # adjusted to match Params
    gamma=0.5250373477563666,  # adjusted to match Params
    contrastive=True,
    supcon_weight=0.3363408062759668,  # adjusted to match Params
    supcon_temperature=0.06602325558360042,  # adjusted to match Params
    supcon_proj_dim=256,  # adjusted to match Params
)


optimizer = dict(
    use_separate_lr=False,
    type='AdamW',
    lr=0.0001,  # matching _cfg_dict default
    weight_decay=0.0004691345228500288,  # adjusted to match Params
)

scheduler = dict(
    type='Cosine',
    cycle_mult=1.0,
    max_lr=0.00029044738350777067,  # adjusted to match Params
    min_lr=3.915411982120591e-06,  # adjusted to match Params
    warmup_steps=0,
    gamma=1.0
)



  