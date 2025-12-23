#!/bin/bash
set -e

WORKSPACE="/home/jovyan/content/CardiacSegV2"
DATA_NAME="chgh"

# 測試影像所在資料夾
TEST_DIR="$WORKSPACE/.././testing/myo_pred/chgh/image"

# 4 個訓練好模型的 checkpoint 路徑
CKPTS=(
  "/home/jovyan/content/CardiacSegV2/exps/exps/basicunetpp/chgh/tune_results/rand30/AICUP_basicunetpp_boundary_rand30_run0/main_65444_00000_0_exp=exp_AICUP_basicunetpp_boundary_rand30_run0_2025-11-25_18-34-36/content/models_run0/best_model.pth"
  "/home/jovyan/content/CardiacSegV2/exps/exps/basicunetpp/chgh/tune_results/rand30/AICUP_basicunetpp_boundary_rand30_run1/main_610d8_00000_0_exp=exp_AICUP_basicunetpp_boundary_rand30_run1_2025-11-25_19-31-45/content/models_run1/best_model.pth"
  "/home/jovyan/content/CardiacSegV2/exps/exps/basicunetpp/chgh/tune_results/rand30/AICUP_basicunetpp_boundary_rand30_run2/main_55616_00000_0_exp=exp_AICUP_basicunetpp_boundary_rand30_run2_2025-11-25_20-28-42/content/models_run2/best_model.pth"
  "/home/jovyan/content/CardiacSegV2/exps/exps/basicunetpp/chgh/tune_results/rand30/AICUP_basicunetpp_boundary_rand30_run3/main_42bb4_00000_0_exp=exp_AICUP_basicunetpp_boundary_rand30_run3_2025-11-25_21-25-26/content/models_run3/best_model.pth"
)

# 存每個 model 的單獨預測
PRED_ROOT="./ensemble/preds_runs"
# 最後 ensemble 結果
ENSEMBLE_OUT="./ensemble/preds_ensemble"

mkdir -p "$PRED_ROOT"
mkdir -p "$ENSEMBLE_OUT"

echo "1: Collect test images"

# 抓所有 .nii.gz 但排除 *_gt.nii.gz
TEST_IMGS=($(ls $TEST_DIR/*.nii.gz | grep -v "_gt\.nii\.gz"))
echo "Found ${#TEST_IMGS[@]} test volumes."


echo "2: Run inference for each model"

RUN_IDX=0
for CKPT in "${CKPTS[@]}"
do
    echo ""
    echo "===== MODEL $RUN_IDX ====="
    echo "Checkpoint: $CKPT"

    MODEL_DIR=$(dirname "$CKPT")              
    OUT_DIR="$PRED_ROOT/run${RUN_IDX}"      
    mkdir -p "$OUT_DIR"

    for IMG in "${TEST_IMGS[@]}"
    do
        echo "[run${RUN_IDX}] infer $(basename $IMG)"
        python $WORKSPACE/expers/infer.py \
            --model_name="basicunetpp" \
            --data_name=$DATA_NAME \
            --data_dir=$TEST_DIR \
            --model_dir=$MODEL_DIR \
            --infer_dir=$OUT_DIR \
            --checkpoint=$CKPT \
            --img_pth=$IMG \
            --out_channels=4 \
            --patch_size=16 \
            --feature_size=48 \
            --drop_rate=0.0 \
            --depths 3 3 9 3 \
            --kernel_size=7 \
            --exp_rate 4 \
            --norm_name='instance' \
            --a_min=-42 \
            --a_max=423 \
            --space_x=0.7 \
            --space_y=0.7 \
            --space_z=1.0 \
            --roi_x=96 \
            --roi_y=96 \
            --roi_z=96 \
            --infer_post_process \
            --loss boundary_loss \
            --lambda_dice 0.5 \
            --lambda_focal 0.5
    done

    RUN_IDX=$((RUN_IDX + 1))
done

echo ""
echo "STEP 3: Ensemble (majority vote)"

python << 'EOF'
import os
import numpy as np
import nibabel as nib

PRED_ROOT = "./ensemble/preds_runs"
ENSEMBLE_OUT = "./ensemble/preds_ensemble"
os.makedirs(ENSEMBLE_OUT, exist_ok=True)

run_dirs = [
    os.path.join(PRED_ROOT, "run0"),
    os.path.join(PRED_ROOT, "run1"),
    os.path.join(PRED_ROOT, "run2"),
    os.path.join(PRED_ROOT, "run3"),
]

files = sorted([
    f for f in os.listdir(run_dirs[0])
    if f.endswith(".nii") or f.endswith(".nii.gz")
])

print(f"Found {len(files)} prediction files for ensemble.")

def mode_voxel(x):
    binc = np.bincount(x)
    return np.argmax(binc)

for fname in files:
    vols = []
    affine = None
    header = None

    print(f"Ensembling {fname} ...")

    for rd in run_dirs:
        full = os.path.join(rd, fname)
        nii = nib.load(full)
        data = nii.get_fdata()

        vols.append(data.astype(np.int32))

        if affine is None:
            affine = nii.affine
            header = nii.header

    vols = np.stack(vols, axis=0)  # [4, D, H, W]
    R, D, H, W = vols.shape
    flat = vols.reshape(R, -1).T   # [N_voxel, R]

    mode_flat = np.apply_along_axis(mode_voxel, 1, flat)
    mode_vol = mode_flat.reshape(D, H, W)

    out = nib.Nifti1Image(mode_vol.astype(np.int16), affine, header)
    nib.save(out, os.path.join(ENSEMBLE_OUT, fname))

print("Ensemble completed. Results saved to:", ENSEMBLE_OUT)
EOF

echo ""
echo "Ensemble Ready"
