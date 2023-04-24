model=dict(
    type="LRPNet",
    in_channels=3,
    out_channels=20,
    encoder_channels=[32,64,96,128,128],
    decoder_channels=[128,128,128,128,128],
)
batch_size = 4
dataset=dict(
    val=dict(
        type="ScanNetCuda",
        data_path="data/scannet",
        mode="test",
        batch_size = batch_size,
    )
)


logger=dict(
    type="RunLogger"
)

log_interval=10
checkpoint_interval=100
eval_interval=10

max_epoch=500
# clean=True
val_reps = 8 
resume_path = "work_dirs/scannet_largenet_f10_scale_trainval/checkpoints/ckpt_epoch_500.pt"
