import os
import torch
import nibabel as nib
import numpy as np
from monai.transforms import AddChannel, AsDiscrete

def check_channel(inp):
    add_ch = AddChannel()
    len_inp_shape = len(inp.shape)
    if len_inp_shape == 4:
        inp = add_ch(inp)
    if len_inp_shape == 3:
        inp = add_ch(inp)
        inp = add_ch(inp)
    return inp

def get_file_pairs(folder_A, folder_B):
    files_A = sorted([f for f in os.listdir(folder_A) if f.endswith(".nii.gz")])
    file_pairs = []
    
    for f in files_A:
        path_A = os.path.join(folder_A, f)
        base_name = f.replace(".nii.gz", "")
        f_gt = base_name + "_gt.nii.gz"
        path_B = os.path.join(folder_B, f_gt)
        
        if not os.path.exists(path_B):
            print(f"GT not found for: {f} â†’ expected {f_gt}")
            continue
        
        file_pairs.append((path_A, path_B))
    
    return file_pairs

def compute_metrics_for_pair(pred1, pred2, cls_num=4):
    post_label = AsDiscrete(to_onehot=cls_num)
    pred1_o = post_label(pred1)
    pred2_o = post_label(pred2)
    
    dice_per_class = []
    iou_per_class = []
    tp_per_class = []
    fp_per_class = []
    tn_per_class = []
    fn_per_class = []
    
    for c in range(1, cls_num):
        p = pred1_o[c]
        g = pred2_o[c]
        
        # Dice
        intersection = (p * g).sum()
        union = p.sum() + g.sum()
        dice = (2.0 * intersection / (union + 1e-8)).item()
        dice_per_class.append(dice)
        
        # IoU
        union_iou = (p + g).clamp(0, 1).sum()
        iou = (intersection / (union_iou + 1e-8)).item()
        iou_per_class.append(iou)
        
        tp = (p * g).sum().item()
        fp = (p * (1 - g)).sum().item()
        fn = ((1 - p) * g).sum().item()
        tn = ((1 - p) * (1 - g)).sum().item()
        
        tp_per_class.append(tp)
        fp_per_class.append(fp)
        tn_per_class.append(tn)
        fn_per_class.append(fn)
    
    return {
        'dice': np.array(dice_per_class),
        'iou': np.array(iou_per_class),
        'tp': np.array(tp_per_class),
        'fp': np.array(fp_per_class),
        'tn': np.array(tn_per_class),
        'fn': np.array(fn_per_class)
    }

def eval_two_predictions_streaming(folder_A, folder_B, cls_num=4, device="cuda"):
    file_pairs = get_file_pairs(folder_A, folder_B)
    print(f"Found {len(file_pairs)} file pairs to process")
    
    num_classes = cls_num - 1
    dice_sum = np.zeros(num_classes)
    iou_sum = np.zeros(num_classes)
    tp_sum = np.zeros(num_classes)
    fp_sum = np.zeros(num_classes)
    tn_sum = np.zeros(num_classes)
    fn_sum = np.zeros(num_classes)
    count = 0
    
    for i, (path_A, path_B) in enumerate(file_pairs):
        print(f"Processing {i+1}/{len(file_pairs)}: {os.path.basename(path_A)}")
        
        img_A = nib.load(path_A).get_fdata().astype(np.float32)
        img_B = nib.load(path_B).get_fdata().astype(np.float32)
        
        pred1 = torch.tensor(img_A)
        pred2 = torch.tensor(img_B)
        
        pred1 = check_channel(pred1.unsqueeze(0).to(device))[0]
        pred2 = check_channel(pred2.unsqueeze(0).to(device))[0]
        
        metrics = compute_metrics_for_pair(pred1, pred2, cls_num)
        
        dice_sum += metrics['dice']
        iou_sum += metrics['iou']
        tp_sum += metrics['tp']
        fp_sum += metrics['fp']
        tn_sum += metrics['tn']
        fn_sum += metrics['fn']
        count += 1
        
        del img_A, img_B, pred1, pred2, metrics
        torch.cuda.empty_cache()
    
    dc_vals = dice_sum / count
    iou_vals = iou_sum / count
    sens_vals = tp_sum / (tp_sum + fn_sum)
    spec_vals = tn_sum / (tn_sum + fp_sum)
    
    return dc_vals, iou_vals, sens_vals, spec_vals

dc, iou, sens, spec = eval_two_predictions_streaming(
    "/home/jovyan/content/testing/train_set_pred/chgh/infer_full100", # train set inference result的路徑
    "/home/jovyan/content/41_training_label", # gt label的路徑
    cls_num=4,
    device="cuda"
)

print("Dice:", dc)
print("IoU:", iou)
print("Sensitivity:", sens)
print("Specificity:", spec)