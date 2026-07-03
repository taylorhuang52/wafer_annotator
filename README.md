# 晶圓缺陷辨識 Demo（WM-811K）

包裝 [RadAI WM-811K ResNet34](https://huggingface.co/radai-agent/radai-wm811k-defect-detection)
模型的簡單網頁介面：上傳一張晶圓圖，模型會判斷屬於 8 種缺陷模式中的哪一種
（Center / Donut / Edge-Loc / Edge-Ring / Loc / Random / Scratch / Near-full）。

推論在**你自己的電腦本機**執行（Flask + PyTorch），圖片不會上傳到外部伺服器。

## 環境需求

- Python 3.14（已確認 PyTorch 2.9+ 官方支援 3.10–3.14，見 pytorch.org/get-started）
- 約 200MB 硬碟空間（模型權重 + 依賴套件）

## 安裝步驟

```bash
cd wafer-defect-demo

# 建議使用虛擬環境
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

## 下載模型權重（只需執行一次）

```bash
python download_model.py
```

這會從 Hugging Face 下載 `best_radai_resnet.pt` 到目前資料夾。
如果下載腳本找不到正確檔名，可以到
https://huggingface.co/radai-agent/radai-wm811k-defect-detection/tree/main
手動下載 `.pt` 檔案，並存成 `best_radai_resnet.pt` 放在跟 `app.py` 同一層目錄。

## 啟動 Demo

```bash
python app.py
```

然後用瀏覽器開啟：**http://127.0.0.1:5000**

## 使用方式

頁面上方有兩個分頁：

### 分頁一：單張圖片

1. 把晶圓圖（wafer map，PNG/JPG 皆可，任意尺寸）拖曳到頁面，或點擊選擇檔案
2. 按「執行辨識」
3. 右側會顯示 8 種缺陷模式的機率排序，最上方為模型判定的結果

### 分頁二：整批 Lot Log

直接輸入整批 lot 的 ATE test log（`dlogTDO` 格式 CSV），不需要自己先轉成圖片：

1. 三種輸入方式擇一：
   - **拖曳多個 CSV 檔案**到框內
   - **拖曳整個 lot 資料夾**到框內（會自動遞迴讀取資料夾內所有 .csv）
   - 點「選擇多個 CSV 檔案」或「選擇 Lot 資料夾」用檔案總管挑選
2. 確認下方檔案清單，按「執行批次辨識」
3. 結果會顯示：
   - **Lot Summary**：Lot ID、wafer 數量、整批的 pattern 分佈（例如幾片 Edge-Ring、幾片 Center）
   - **每片 wafer 一列**：wafer ID、良率（yield）、Pass/Fail 顆數、判定的缺陷 pattern、信心度、小圖預覽
   - 點擊任一列可展開該片 wafer 完整 8 類機率排序

程式會自動從 CSV 中的 `XAdr` / `YAdr`（座標）與 `Bin#`（1=Pass，其他=Fail）
還原出晶圓的 bin map，直接餵給模型判斷，不需要額外產生圖片檔。

**支援的 CSV 格式**：CRAFT 系列測試機台輸出的 dlogTDO log，開頭會有
`Lot ID`、`Wafer ID` 等欄位，接著是以 `Serial#,Site#,Bin#,SBin#,XAdr,YAdr,...`
為表頭的資料表。若上傳的檔案不符合這個格式，該檔案會被略過並在結果下方顯示錯誤原因，不會中斷整批處理。

程式也會自動解析 CSV 表頭裡 `IR0(8)`、`VF1(17)` 這類「測項名稱(Bin#)」的
對照表，算出每片 wafer **主要 fail 在哪個測項**，顯示在結果表格的「主要
Fail 測項」欄位。

### 修正模型判斷錯誤的標籤

如果模型判斷的 pattern 不準確，可以在結果表格最右邊「修正標籤」欄位：

1. 從下拉選單選擇正確的 pattern（或選「不確定/其他」）
2. 按「存」

儲存後會累積到 `corrected_labels/` 資料夾：

```
corrected_labels/
├── labels.csv              # 每筆修正的紀錄（wafer ID、原判斷、修正後標籤、主要 fail 測項…）
└── binmaps/
    ├── RK30906-02.npy       # 該片 wafer 完整的 Bin# 原始數據（不是圖片，可精確還原）
    └── ...
```

`labels.csv` 可以直接用 Excel 打開檢視。這些修正資料**目前只會被儲存、不會
自動拿去重新訓練模型**——累積到一定數量後，若要用來 fine-tune 模型，需要另外
寫訓練腳本（屬於進階功能，需要時可以再另外討論實作）。

## 檔案結構

```
wafer-defect-demo/
├── app.py                # Flask 後端 + 模型推論邏輯 + /predict、/predict_batch、/save_correction
├── wafer_log_parser.py   # 解析 ATE dlogTDO CSV → wafer bin map + bin-測項對照表
├── download_model.py     # 從 Hugging Face 下載權重的輔助腳本
├── requirements.txt
├── templates/
│   └── index.html        # 前端頁面（單張圖片 + 整批 Lot Log 兩個分頁，含標籤修正功能）
├── best_radai_resnet.pt  # 執行 download_model.py 後產生
└── corrected_labels/     # 執行過標籤修正後自動產生，內含 labels.csv + binmaps/
```

## 常見問題

**Q: 我的圖片不是標準晶圓圖格式，可以用嗎？**
模型接受任意大小的灰階圖片，內部會自動縮放到 64×64 並正規化。但準確率會依圖片
與訓練資料（WM-811K 標準晶圓圖）的相似程度而定 —— 一般拍照或非晶圓圖片的預測
結果不具參考價值。

**Q: pip install torch 失敗 / 找不到符合 Python 3.14 的 wheel？**
截至目前，PyTorch 對 Python 3.14 的支援仍相對新（2.9 版起提供預覽支援）。若安裝
失敗，可以：
1. 改用 `pip install torch --pre` 安裝 nightly 版本，或
2. 改用 Python 3.12/3.13 建立虛擬環境（相容性最穩定）

**Q: 想關閉 debug 模式 / 部署到區網其他裝置？**
把 `app.py` 最後一行改成：
```python
app.run(host="0.0.0.0", port=5000)
```
即可讓區網內其他裝置連線使用。

## 授權

模型本身為 MIT 授權（見 Hugging Face model card）。
