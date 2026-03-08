# -*- coding: utf-8 -*-
"""
乳腺钼靶 TIFF 图像预处理脚本
- 从黑色背景中抠出乳腺 ROI
- 翻转：使所有图像按「乳头向左」统一方向
- 填充：黑底、乳腺居中（默认画布为 max(高,宽) 的正方形）；填充后再缩放到 512×512
- 可选归一化、保存为 PNG

使用：
  1. 安装依赖：pip install -r requirements_preprocess.txt
  2. 将 TIFF 放在项目下的 TIFF Images 文件夹（可含子目录）
  3. 运行：python preprocess_mammo.py
  4. 结果默认保存在 Processed_Mammo/，保持原有相对路径，扩展名为 .png

可选参数示例：
  --input_dir / --output_dir  输入/输出目录
  --padding 30  裁剪 ROI 时四边预留像素
  --no_flip  不进行「乳头向左」翻转
  --resize 512 512  填充后缩放到该尺寸（默认 512×512）
  --save_with_border  保存带绿色轮廓的图便于检查（未做固定尺寸填充时有效）
"""

import os
import argparse
import numpy as np
import cv2
from pathlib import Path

# 优先使用 tifffile 读取医学 TIFF（支持 16 位、多帧）
try:
    import tifffile
    HAS_TIFFFILE = True
except ImportError:
    HAS_TIFFFILE = False
from PIL import Image


def load_mammo_tiff(path: str) -> np.ndarray:
    """加载乳腺 TIFF，统一为灰度 uint8 数组。"""
    path = str(path)
    if HAS_TIFFFILE:
        try:
            im = tifffile.imread(path)
        except Exception:
            im = np.array(Image.open(path))
    else:
        im = np.array(Image.open(path))

    if im.ndim == 3:
        im = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY) if im.shape[-1] == 3 else im[:, :, 0]
    if im.dtype != np.uint8:
        # 16 位或其它类型：线性拉伸到 0-255
        im = np.clip(im.astype(np.float64), 0, None)
        if im.max() > im.min():
            im = ((im - im.min()) / (im.max() - im.min()) * 255).astype(np.uint8)
        else:
            im = im.astype(np.uint8)
    return im


def strip_border_artifacts(gray: np.ndarray, black_max: int = 8, bright_min: int = 200, thin_dark_ratio: float = 0.88) -> np.ndarray:
    """
    去除图像四边的非纯黑边框（如顶部/左侧的细微白边、灰边），避免被误判为乳腺。
    从四边向内收缩，去掉「全黑行/列」或「以黑为主、仅含细亮线的行/列」。
    """
    h, w = gray.shape
    y0, y1, x0, x1 = 0, h, 0, w

    def is_black_line(vals: np.ndarray, length: int) -> bool:
        return vals.max() < black_max or vals.mean() < 3

    def is_thin_bright_line(vals: np.ndarray, length: int) -> bool:
        # 大部分为暗像素，但存在亮像素 → 细亮线/白边
        dark = np.sum(vals < 25) / length
        return dark >= thin_dark_ratio and vals.max() >= bright_min

    def is_border_row(y: int) -> bool:
        r = gray[y, :]
        return is_black_line(r, w) or is_thin_bright_line(r, w)

    def is_border_col(x: int) -> bool:
        c = gray[:, x]
        return is_black_line(c, h) or is_thin_bright_line(c, h)

    while y0 < y1 and is_border_row(y0):
        y0 += 1
    while y1 > y0 and is_border_row(y1 - 1):
        y1 -= 1
    while x0 < x1 and is_border_col(x0):
        x0 += 1
    while x1 > x0 and is_border_col(x1 - 1):
        x1 -= 1

    if y0 >= y1 or x0 >= x1:
        return gray
    return gray[y0:y1, x0:x1].copy()


def segment_breast_mask(gray: np.ndarray, threshold_ratio: float = 0.02) -> np.ndarray:
    """
    从灰度图中得到乳腺二值掩膜（背景为黑、乳腺为亮）。
    - 使用低阈值分离背景与乳腺，再取最大连通域并做简单形态学清理。
    """
    # 低阈值：比背景稍亮即视为前景（乳腺）
    low = max(20, int(255 * threshold_ratio))
    _, binary = cv2.threshold(gray, low, 255, cv2.THRESH_BINARY)

    # 形态学开运算去小噪点、标签等
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    # 取最大连通域作为乳腺
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if num_labels <= 1:
        return binary
    # 0 为背景，找面积最大的前景
    areas = stats[1:, cv2.CC_STAT_AREA]
    max_idx = np.argmax(areas) + 1
    mask = (labels == max_idx).astype(np.uint8) * 255
    return mask


def get_breast_bbox(mask: np.ndarray, padding: int = 0) -> tuple:
    """根据乳腺掩膜计算外接矩形，可加 padding（不超过图像边界）。"""
    ys, xs = np.where(mask > 0)
    if ys.size == 0 or xs.size == 0:
        return 0, 0, mask.shape[1], mask.shape[0]
    x_min, x_max = int(xs.min()), int(xs.max())
    y_min, y_max = int(ys.min()), int(ys.max())
    h, w = mask.shape
    x_min = max(0, x_min - padding)
    x_max = min(w, x_max + 1 + padding)
    y_min = max(0, y_min - padding)
    y_max = min(h, y_max + 1 + padding)
    return x_min, y_min, x_max, y_max


def crop_breast_roi(image: np.ndarray, mask: np.ndarray, padding: int = 0):
    """根据乳腺掩膜裁剪出乳腺 ROI 图像。返回 (roi, roi_mask, bbox)。"""
    x_min, y_min, x_max, y_max = get_breast_bbox(mask, padding)
    roi = image[y_min:y_max, x_min:x_max].copy()
    roi_mask = mask[y_min:y_max, x_min:x_max].copy()
    roi[roi_mask == 0] = 0
    return roi, roi_mask, (x_min, y_min, x_max, y_max)


def flip_roi_to_nipple_left(roi: np.ndarray, roi_mask: np.ndarray) -> tuple:
    """
    翻转使所有乳腺按「乳头向左」排列。
    若乳腺质心在图像左半侧（乳头朝右），则水平翻转；否则不翻转。
    返回 (roi, roi_mask)。
    """
    h, w = roi.shape
    ys, xs = np.where(roi_mask > 0)
    if ys.size == 0 or xs.size == 0:
        return roi, roi_mask
    cx = float(xs.mean())
    if cx < w / 2:
        return np.fliplr(roi).copy(), np.fliplr(roi_mask).copy()
    return roi, roi_mask


def pad_breast_centered(
    roi: np.ndarray,
    roi_mask: np.ndarray,
    target_height: int,
    target_width: int,
    fill_value: int = 0,
) -> np.ndarray:
    """
    用黑色填充至固定宽高，并在乳房左侧等侧添加像素使乳腺区域居中。
    画布为 (target_height, target_width)，乳腺质心对准画布中心，四周 fill_value 填充。
    若 roi 大于画布则从中心裁剪后再放置（或可缩小，此处采用居中裁剪）。
    """
    rh, rw = roi.shape
    ys, xs = np.where(roi_mask > 0)
    if ys.size == 0 or xs.size == 0:
        cy, cx = rh // 2, rw // 2
    else:
        cy, cx = float(ys.mean()), float(xs.mean())
    out = np.full((target_height, target_width), fill_value, dtype=roi.dtype)
    # 使质心 (cy, cx) 落在 (target_height/2, target_width/2)
    y0 = int(round(target_height / 2 - cy))
    x0 = int(round(target_width / 2 - cx))
    # roi 在 out 上的有效粘贴范围
    src_y0 = max(0, -y0)
    src_x0 = max(0, -x0)
    src_y1 = min(rh, target_height - y0)
    src_x1 = min(rw, target_width - x0)
    dst_y0 = max(0, y0)
    dst_x0 = max(0, x0)
    dst_y1 = min(target_height, y0 + rh)
    dst_x1 = min(target_width, x0 + rw)
    if src_y1 > src_y0 and src_x1 > src_x0:
        out[dst_y0:dst_y1, dst_x0:dst_x1] = roi[src_y0:src_y1, src_x0:src_x1]
    return out


def preprocess_one(
    input_path: str,
    output_path: str,
    *,
    threshold_ratio: float = 0.02,
    padding: int = 20,
    flip_nipple_left: bool = True,
    pad_to_square: bool = True,
    target_height: int = None,
    target_width: int = None,
    resize_to: tuple = (512, 512),
    save_with_mask_border: bool = False,
    normalize: bool = True,
) -> bool:
    """
    对单张乳腺 TIFF 做预处理并保存。
    - 抠出乳腺 ROI 并裁剪；
    - 翻转使乳头向左；
    - 可选：填充至固定宽高并使乳腺居中；
    - 可选归一化到 0-255；
    - 保存为 PNG。
    """
    try:
        gray = load_mammo_tiff(input_path)
    except Exception as e:
        print(f"  [跳过] 无法读取: {input_path} -> {e}")
        return False

    gray = strip_border_artifacts(gray)
    mask = segment_breast_mask(gray, threshold_ratio=threshold_ratio)
    roi, roi_mask, _ = crop_breast_roi(gray, mask, padding=padding)

    if roi.size == 0:
        print(f"  [跳过] 未检测到乳腺区域: {input_path}")
        return False

    if flip_nipple_left:
        roi, roi_mask = flip_roi_to_nipple_left(roi, roi_mask)

    # 填充：黑底、乳腺居中。若指定了 target_height/target_width 则用其作为画布；否则将宽高补齐为一致（边长 = max(高,宽)，如 1000*3000 -> 3000*3000）
    do_pad = pad_to_square or (target_height is not None and target_width is not None and target_height > 0 and target_width > 0)
    if do_pad:
        if target_height is not None and target_width is not None and target_height > 0 and target_width > 0:
            th, tw = target_height, target_width
        else:
            side = max(roi.shape[0], roi.shape[1])
            th, tw = side, side
        roi = pad_breast_centered(roi, roi_mask, th, tw, fill_value=0)

    # 在填充好的图像上缩放到指定尺寸（默认 512×512）
    if resize_to is not None and len(resize_to) == 2 and resize_to[0] > 0 and resize_to[1] > 0:
        roi = cv2.resize(roi, (resize_to[1], resize_to[0]), interpolation=cv2.INTER_AREA)

    if normalize:
        roi = np.clip(roi.astype(np.float64), 0, None)
        if roi.max() > roi.min():
            roi = ((roi - roi.min()) / (roi.max() - roi.min()) * 255).astype(np.uint8)
        else:
            roi = roi.astype(np.uint8)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    if save_with_mask_border and not do_pad:
        contour_img = cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR)
        contours, _ = cv2.findContours(
            roi_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(contour_img, contours, -1, (0, 255, 0), 1)
        cv2.imwrite(output_path, contour_img)
    else:
        cv2.imwrite(output_path, roi)
    return True


def main():
    parser = argparse.ArgumentParser(description="乳腺钼靶 TIFF 预处理：抠出乳腺 ROI 并保存")
    parser.add_argument(
        "--input_dir",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "TIFF Images"),
        help="存放 TIFF 的根目录（会递归搜索 .tif .tiff）",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "Processed_Mammo"),
        help="预处理结果保存目录",
    )
    parser.add_argument(
        "--threshold_ratio",
        type=float,
        default=0.02,
        help="前景阈值（相对 255 的比例），用于分离背景与乳腺，默认 0.02",
    )
    parser.add_argument(
        "--padding",
        type=int,
        default=20,
        help="裁剪 ROI 时四边预留像素",
    )
    parser.add_argument(
        "--no_flip",
        action="store_true",
        help="不进行「乳头向左」翻转（默认会翻转使方向一致）",
    )
    parser.add_argument(
        "--no_pad",
        action="store_true",
        help="不进行填充（默认会将宽高补齐为正方形，如 1000*3000 -> 3000*3000，黑底乳腺居中）",
    )
    parser.add_argument(
        "--target_height",
        type=int,
        default=None,
        help="填充画布高度（与 --target_width 同时指定时使用；否则自动为 max(高,宽)）",
    )
    parser.add_argument(
        "--target_width",
        type=int,
        default=None,
        help="填充画布宽度（与 --target_height 同时指定时使用）",
    )
    parser.add_argument(
        "--resize",
        type=int,
        nargs=2,
        default=[512, 512],
        metavar=("H", "W"),
        help="在填充好的图像上缩放到该尺寸，默认 512 512（即 512×512）",
    )
    parser.add_argument(
        "--save_with_border",
        action="store_true",
        help="保存带乳腺轮廓的图（便于检查），否则只保存裁剪后的灰度图",
    )
    parser.add_argument(
        "--no_normalize",
        action="store_true",
        help="不做 min-max 归一化",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    suffixes = (".tif", ".tiff", ".TIF", ".TIFF")
    files = []
    for suf in suffixes:
        files.extend(input_dir.rglob(f"*{suf}"))

    if not files:
        print(f"在 {input_dir} 下未找到任何 TIFF 文件。")
        return

    print(f"共找到 {len(files)} 张 TIFF，开始预处理（抠出乳腺 ROI）...")
    success = 0
    for i, f in enumerate(files):
        rel = f.relative_to(input_dir) if input_dir in f.parents else f.name
        out_path = output_dir / (rel.with_suffix(".png"))
        if preprocess_one(
            str(f),
            str(out_path),
            threshold_ratio=args.threshold_ratio,
            padding=args.padding,
            flip_nipple_left=not args.no_flip,
            pad_to_square=not args.no_pad,
            target_height=args.target_height,
            target_width=args.target_width,
            resize_to=tuple(args.resize) if args.resize else None,
            save_with_mask_border=args.save_with_border,
            normalize=not args.no_normalize,
        ):
            success += 1
        if (i + 1) % 50 == 0:
            print(f"  已处理 {i + 1}/{len(files)}")
    print(f"完成：成功 {success}/{len(files)}，结果保存在 {output_dir}")


if __name__ == "__main__":
    main()
