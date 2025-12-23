import os


def get_data_dicts(data_dir):
    # patient_dirs = sorted(os.listdir(data_dir), key=lambda x: int(x.split('_')[-1]))
    patient_dirs = ['pid_02', 'pid_31', 'pid_1000']
    data_dicts = []
    for patient_dir in patient_dirs:
        data_dicts.append({
            "image": os.path.join(os.path.join(data_dir, patient_dir, f'{patient_dir}.nii.gz')),
            "label": os.path.join(os.path.join(data_dir, patient_dir, f'{patient_dir}_gt.nii.gz'))
        })
    return data_dicts

## random select 則使用下方function

# import random

# def get_data_dicts(data_dir, max_cases=30):
#     """
#     從 data_dir 自動偵測所有 patientXXXX.nii.gz / patientXXXX_gt.nii.gz
#     每次呼叫都隨機選 max_cases 筆。
#     """

#     all_imgs = [
#         f for f in os.listdir(data_dir)
#         if f.endswith(".nii.gz") and not f.endswith("_gt.nii.gz")
#     ]

#     data_dicts = []

#     for img_name in all_imgs:
#         base = img_name.replace(".nii.gz", "")
#         lbl_name = f"{base}_gt.nii.gz"

#         img_path = os.path.join(data_dir, img_name)
#         lbl_path = os.path.join(data_dir, lbl_name)

#         if os.path.exists(lbl_path):
#             data_dicts.append({
#                 "image": img_path,
#                 "label": lbl_path,
#             })

#     random.shuffle(data_dicts)

#     if max_cases is not None and len(data_dicts) > max_cases:
#         data_dicts = data_dicts[:max_cases]

#     print(f"[CHGH] Random selected {len(data_dicts)} cases:")
#     for d in data_dicts:
#         print("  -", os.path.basename(d["image"]))

#     return data_dicts
