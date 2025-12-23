# 1141_Deep_Learning_Final_Report

## CardiacSegV2

### 調整內容：
* **`trainset_evaluation.py`**：可對完整訓練集進行 Inference，並計算與 GT 對應的 Dice 與 IoU 數值，作為評估分數落點的參考。
* **`get_label.ipynb`**：用於檢視 GT 資料的 Labels 以及各個 Label 紀錄的詳細內容。
* **`run_ensemble.sh`**：執行即可得到 4 個模型 Ensemble 疊加後的結果。
* **`/datasets/chgh_datasets.py`**：新增隨機抽取函式（見註解），可不依賴預先產生的 JSON 檔進行資料採樣。
* **`/expers/tune.py`**：加入不同 Loss Function 的設定選項。
* **`/networks/network.py`**：加入 UNet++ 相關網路架構設定與調整。

* MedLSAM 的實作參考自原始論文作者所提供的官方程式碼： [MedLSAM Official Repository](https://github.com/openmedlab/MedLSAM.git)
