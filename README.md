# 乳腺钼靶图像预处理与 BiomedCLIP

本项目包含两部分：**乳腺 X 光片预处理**（抠图、翻转、填充、缩放）与 **BiomedCLIP 图-文对比学习训练**（根据 Metadata 的 D/E 列生成提示词，做图像-文本对齐）。

---

## 一、环境准备

### 1. 预处理脚本依赖

```bash
pip install numpy opencv-python Pillow tifffile
```

（可选）若需将依赖写进文件，可新建 `requirements_preprocess.txt`，内容为：

```
numpy>=1.19.0
opencv-python>=4.5.0
Pillow>=8.0.0
tifffile>=2021.0.0
```

然后执行：`pip install -r requirements_preprocess.txt`

### 2. 训练脚本依赖

```bash
pip install -r requirements_biomedclip.txt
```

需要：`torch`、`torchvision`、`open_clip_torch`、`transformers`、`Pillow`、`pandas`、`openpyxl`、`tqdm`。训练建议使用 **GPU**（CUDA）。

---

## 二、数据与元数据说明

### 1. 原始图像

- 格式：**TIFF**（`.tif` / `.tiff`）
- 放置：默认在项目根目录下的 **`TIFF Images`** 文件夹内（可含子目录，脚本会递归查找）

### 2. Metadata.xlsx

训练时根据 Excel 的 **D 列** 和 **E 列** 生成提示词并过滤样本：

- **D 列**：病灶类型（如 NORM、CALC、CIRC、SPIC、MISC、ARCH、ASYM 等）
- **E 列**：良恶性标记 —— **B** = 良性，**M** = 恶性，**N** = 排除
- 前两列需包含：第 1 列为图像 ID（如 IMG001），第 2 列为视图（如 MLOLT、CCRT）

**提示词规则：**

- D 全为 NORM → `"Mammogram showing healthy breast"`
- D 非 NORM 且 E=B → `"Mammogram showing breast with benign mass"`
- D 非 NORM 且 E=M → `"Mammogram showing breast with malignant mass"`

**排除规则：** D 列含 **ARCH** 或 **ASYM** 的样本、以及 E 列为 **N** 的样本不会参与训练。

---

## 三、使用流程

### 步骤 1：预处理（得到 512×512 PNG）

在项目根目录执行：

```bash
python preprocess_mammo.py
```

- **默认行为**：从 `TIFF Images` 读取 TIFF，去边、抠乳腺 ROI、翻转（乳头向左）、黑底填充为正方形后缩放到 **512×512**，保存到 **`Processed_Mammo`**，保持原有相对路径，扩展名为 `.png`。

**常用参数：**

| 参数 | 说明 |
|------|------|
| `--input_dir <路径>` | TIFF 所在根目录（默认：`TIFF Images`） |
| `--output_dir <路径>` | 输出目录（默认：`Processed_Mammo`） |
| `--padding 20` | 裁剪 ROI 时四边预留像素 |
| `--no_flip` | 不进行「乳头向左」翻转 |
| `--no_pad` | 不填充、不缩放，只保存裁剪+翻转结果 |
| `--resize 512 512` | 填充后缩放到指定高×宽（默认 512×512） |
| `--save_with_border` | 保存带绿色轮廓的图便于检查（未做 resize 时有效） |

示例：指定输入输出目录并保持默认 512×512：

```bash
python preprocess_mammo.py --input_dir "D:/Data/TIFF" --output_dir "D:/Data/Processed_Mammo"
```

---

### 步骤 2：训练 BiomedCLIP 图-文对比学习

预处理完成后，使用 **`Processed_Mammo` 下的 PNG** 和 **Metadata.xlsx** 进行训练：

```bash
python train_biomedclip_contrastive.py
```

- **默认**：图像目录为 **`Processed_Mammo/TIFF Images`**，元数据为 **`Metadata.xlsx`**，checkpoint 保存到 **`biomedclip_contrastive_checkpoints`**（如 `best.pt`、`last.pt`）。

**常用参数：**

| 参数 | 说明 |
|------|------|
| `--image_dir <路径>` | 预处理后的 PNG 根目录（默认：`Processed_Mammo/TIFF Images`） |
| `--metadata <路径>` | Metadata.xlsx 路径（默认：`Metadata.xlsx`） |
| `--output_dir <路径>` | checkpoint 保存目录 |
| `--epochs 30` | 训练轮数 |
| `--batch_size 16` | 批大小 |
| `--lr 5e-5` | 学习率 |
| `--val_ratio 0.2` | 验证集比例 |
| `--test_ratio 0.15` | 测试集比例（0 表示不单独划分测试集） |
| `--no_augment` | 关闭训练时数据增强（随机旋转、随机水平翻转） |
| `--freeze` | 冻结 BiomedCLIP 全部参数（仅调试用） |
| `--num_workers 8` | DataLoader 工作进程数 |
| `--gradient_checkpointing` | 启用以降低显存占用 |

示例：指定图像与元数据路径并启用梯度检查点：

```bash
python train_biomedclip_contrastive.py --image_dir "D:/Data/Processed_Mammo/TIFF Images" --metadata "D:/Data/Metadata.xlsx" --gradient_checkpointing
```

---

## 四、目录结构建议

```
Mammo/
├── README.md
├── Metadata.xlsx              # 元数据（D 列、E 列等）
├── preprocess_mammo.py        # 预处理脚本
├── train_biomedclip_contrastive.py   # 训练脚本
├── requirements_biomedclip.txt
├── TIFF Images/               # 原始 TIFF（可含子目录）
│   └── ...
├── Processed_Mammo/           # 预处理输出（默认 512×512 PNG）
│   └── TIFF Images/
│       └── ...
├── biomedclip_contrastive_checkpoints/  # 训练 checkpoint
│   ├── best.pt
│   └── last.pt
└── biomedclip_mammo/          # 数据集与元数据解析
    ├── dataset.py
    └── metadata_utils.py
```

若预处理时 `--output_dir` 与默认不同，训练时需用 `--image_dir` 指向实际 PNG 所在目录（例如若输出到 `Processed_Mammo` 且无子目录，则 `--image_dir Processed_Mammo`）。

---

## 五、注意事项

1. **文件名与 Metadata 对应**：预处理后的 PNG 文件名需能解析出 image_id（及可选 view），例如 `IMG001.png`、`IMG001_MLOLT.png`，以便与 Metadata 中第 1、2 列匹配。
2. **显存不足**：可减小 `--batch_size`，或加上 `--gradient_checkpointing`。
3. **仅 CPU**：可正常训练，速度较慢；预处理不依赖 GPU。
