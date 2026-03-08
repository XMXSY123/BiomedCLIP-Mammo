# -*- coding: utf-8 -*-

import os
import argparse
import torch
import torch.nn.functional as F
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

from biomedclip_mammo.dataset import get_dataloaders_image_text


def get_biomedclip_full_model(device, freeze: bool = False):
    """加载完整 BiomedCLIP与 tokenizer、preprocess。"""
    from open_clip import create_model_from_pretrained, get_tokenizer
    model, preprocess = create_model_from_pretrained("hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224")
    tokenizer = get_tokenizer("hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224")
    model.to(device)
    if freeze:
        for p in model.parameters():
            p.requires_grad = False
    else:
        model.train()
    return model, tokenizer, preprocess


def clip_contrastive_loss(image_features, text_features, logit_scale):
    """对称 CLIP 对比损失：batch 内 image_i 与 text_i 为正对。"""
    image_features = F.normalize(image_features, dim=-1)
    text_features = F.normalize(text_features, dim=-1)
    logits_per_image = logit_scale * image_features @ text_features.t()
    logits_per_text = logits_per_image.t()
    batch_size = image_features.shape[0]
    labels = torch.arange(batch_size, device=image_features.device, dtype=torch.long)
    loss_i = F.cross_entropy(logits_per_image, labels)
    loss_t = F.cross_entropy(logits_per_text, labels)
    return (loss_i + loss_t) / 2


def train_one_epoch(model, tokenizer, train_loader, optimizer, device, context_length: int, epoch: int, total_epochs: int, use_tqdm: bool = True):
    from tqdm import tqdm
    model.train()
    total_loss, correct_i2t, total = 0.0, 0, 0
    it = train_loader
    if use_tqdm:
        it = tqdm(it, desc=f"Epoch {epoch}/{total_epochs} [train]", leave=False, unit="batch")
    for images, texts in it:
        images = images.to(device)
        if not isinstance(texts, (list, tuple)):
            texts = [texts]
        texts = list(texts)
        text_tokens = tokenizer(texts, context_length=context_length).to(device)
        optimizer.zero_grad()
        image_features, text_features, logit_scale = model(images, text_tokens)
        image_features = image_features.float()
        text_features = text_features.float()
        loss = clip_contrastive_loss(image_features, text_features, logit_scale)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        # 检索准确率：image 最相似 text 是否为配对
        logits = logit_scale * image_features @ text_features.t()
        pred = logits.argmax(dim=1)
        labels = torch.arange(images.size(0), device=device)
        correct_i2t += (pred == labels).sum().item()
        total += images.size(0)
        if use_tqdm and hasattr(it, "set_postfix"):
            it.set_postfix(loss=f"{loss.item():.4f}", acc=f"{correct_i2t/total:.4f}")
    return total_loss / len(train_loader), correct_i2t / total if total else 0.0


@torch.no_grad()
def evaluate_retrieval(model, tokenizer, val_loader, device, context_length: int, use_tqdm: bool = True):
    from tqdm import tqdm
    model.eval()
    correct, total = 0, 0
    it = tqdm(val_loader, desc="验证", leave=False, unit="batch") if use_tqdm else val_loader
    for images, texts in it:
        images = images.to(device)
        texts = list(texts) if isinstance(texts, (list, tuple)) else [texts]
        text_tokens = tokenizer(texts, context_length=context_length).to(device)
        image_features, text_features, logit_scale = model(images, text_tokens)
        image_features = image_features.float()
        text_features = text_features.float()
        logits = logit_scale * image_features @ text_features.t()
        pred = logits.argmax(dim=1)
        labels = torch.arange(images.size(0), device=device)
        correct += (pred == labels).sum().item()
        total += images.size(0)
    return correct / total if total else 0.0


def main():
    parser = argparse.ArgumentParser(description="BiomedCLIP")
    parser.add_argument("--image_dir", type=str, default=str(PROJECT_ROOT / "Processed_Mammo" / "TIFF Images"))
    parser.add_argument("--metadata", type=str, default=str(PROJECT_ROOT / "Metadata.xlsx"))
    parser.add_argument("--output_dir", type=str, default=str(PROJECT_ROOT / "biomedclip_contrastive_checkpoints"))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--test_ratio", type=float, default=0.0)
    parser.add_argument("--freeze", action="store_true")
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--no_augment", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    context_length = 256

    model, tokenizer, preprocess = get_biomedclip_full_model(device, freeze=args.freeze)
    out = get_dataloaders_image_text(
        args.image_dir, args.metadata, preprocess,
        batch_size=args.batch_size, num_workers=args.num_workers,
        val_ratio=args.val_ratio, test_ratio=args.test_ratio, seed=args.seed,
        train_augment=not args.no_augment,
    )
    if len(out) == 3 and out[2] is not None:
        train_loader, val_loader, test_loader = out
        print("划分: 训练 / 验证 / 测试 =", len(train_loader.dataset), "/", len(val_loader.dataset), "/", len(test_loader.dataset))
    else:
        train_loader, val_loader = out[0], out[1]
        test_loader = None
        print("划分: 训练 / 验证 =", len(train_loader.dataset), "/", len(val_loader.dataset))
    if len(train_loader.dataset) == 0:
        print("未找到带 D 列文本的样本，请确认 image_dir 与 Metadata 中 (image_id, view) 对应。")
        return

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    os.makedirs(args.output_dir, exist_ok=True)
    best_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, tokenizer, train_loader, optimizer, device, context_length,
            epoch=epoch, total_epochs=args.epochs, use_tqdm=True,
        )
        val_acc = evaluate_retrieval(model, tokenizer, val_loader, device, context_length, use_tqdm=True)
        print(f"Epoch {epoch}/{args.epochs}  train_loss={train_loss:.4f}  train_retrieval_acc={train_acc:.4f}  val_retrieval_acc={val_acc:.4f}")
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            }, os.path.join(args.output_dir, "best.pt"))
            print("  -> 已保存 best.pt")

    torch.save({"epoch": args.epochs, "model_state_dict": model.state_dict()}, os.path.join(args.output_dir, "last.pt"))
    print("训练完成，best val_retrieval_acc:", best_acc)
    if test_loader is not None:
        ckpt = torch.load(os.path.join(args.output_dir, "best.pt"), map_location=device)
        model.load_state_dict(ckpt["model_state_dict"], strict=True)
        test_acc = evaluate_retrieval(model, tokenizer, test_loader, device, context_length)
        print("测试集 image-to-text 检索准确率:", test_acc)


if __name__ == "__main__":
    main()
