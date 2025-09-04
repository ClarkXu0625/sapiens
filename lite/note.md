Export paths and activate environment

```
    export SAPIENS_ROOT=/home/clark/Documents/GitHub/sapiens
    export SAPIENS_LITE_ROOT=$SAPIENS_ROOT/lite
    export SAPIENS_CHECKPOINT_ROOT=/home/clark/Documents/GitHub/sapiens/lite/torchscript
    conda activate sapiens_lite   
```


Run:
```
    cd $SAPIENS_LITE_ROOT/scripts/demo/torchscript
    ./extract_feature.sh
    ./seg.sh
    ./depth.sh
    ./normal.sh

```

Run backproject
```
    python backproject_depth_to_ply.py \
    --depth_npy /path/to/frame_depth.npy \
    --rgb /path/to/frame.jpg \
    --mask_npy /path/to/frame_mask.npy \
    --ply_out /tmp/frame_cloud.ply \
    --fov_deg 60 \
    --median_m 1.5 --range_m 2.0
```

