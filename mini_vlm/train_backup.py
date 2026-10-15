"""运行：python train.py；之后只生成：python train.py --demo。"""
import argparse
import random
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from PIL import Image

from dataset import BOS_ID, EOS_ID, VOCAB, WORDS, ShapeCaptionDataset, decode_caption
from model import MiniVLM

EPOCHS = 40
BATCH_SIZE = 24
LEARNING_RATE = 0.001
OUT_DIR = Path(__file__).resolve().parent / "outputs"
WEIGHTS = OUT_DIR / "mini_vlm.pth"


@torch.no_grad()
def evaluate(model, loader, device):
    """评估自由生成的整句匹配率，不给模型真实前文。"""
    model.eval()
    correct, total = 0, 0
    for images, captions in loader:
        generated = model.generate(images.to(device), BOS_ID, EOS_ID).cpu()
        for predicted, target in zip(generated, captions):
            # 既要文字正确，也要生成结束标记。
            correct += int(decode_caption(predicted) == decode_caption(target)
                           and EOS_ID in predicted[1:].tolist())
            total += 1
    return correct / total


def run_demo(model, device):
    model.eval()
    demo_data = ShapeCaptionDataset(num_samples=6, seed=20000)
    images = torch.stack([demo_data[i][0] for i in range(6)])
    generated = model.generate(images.to(device), BOS_ID, EOS_ID).cpu()
    OUT_DIR.mkdir(exist_ok=True)
    lines = []
    for i, token_ids in enumerate(generated):
        image, true_ids = demo_data[i]
        # Pillow 保存，不使用 matplotlib 弹窗；这里也不依赖 NumPy。
        pixels = (image.permute(1, 2, 0) * 255).round().byte().flatten().tolist()
        Image.frombytes("RGB", (32, 32), bytes(pixels)).save(OUT_DIR / f"query_{i}.png")
        tokens = [WORDS[int(t)] for t in token_ids]
        if "<eos>" in tokens:
            tokens = tokens[:tokens.index("<eos>") + 1]
        lines.append(f"query_{i}.png | True: {decode_caption(true_ids)} | "
                     f"Generated: {decode_caption(token_ids)}\n"
                     f"Tokens: {' -> '.join(tokens)}")
    report = "\n\n".join(lines)
    print("\n" + report)
    (OUT_DIR / "generations.txt").write_text(report, encoding="utf-8")
    print("\nSaved images and text to:", OUT_DIR)


def smoke_test(model, device):
    """只跑一批前向/反向/更新，且检查因果性；不保存权重。"""
    loader = DataLoader(ShapeCaptionDataset(6), batch_size=6, num_workers=0)
    images, captions = next(iter(loader))
    images, captions = images.to(device), captions.to(device)
    inputs, targets = captions[:, :-1], captions[:, 1:]
    logits = model(images, inputs)
    loss = nn.functional.cross_entropy(logits.reshape(-1, len(VOCAB)), targets.reshape(-1))
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    optimizer.zero_grad()
    loss.backward()
    assert torch.isfinite(loss), "loss 出现非有限值。"
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    optimizer.step()
    model.eval()
    with torch.no_grad():
        changed = inputs.clone()
        changed[:, -1] = VOCAB["blue"]
        before, after = model(images, inputs), model(images, changed)
        # 修改最后一个输入词，前面的输出必须不变，否则偷看了未来。
        assert torch.allclose(before[:, :-1], after[:, :-1], atol=1e-5, rtol=1e-5)
    print("images:", tuple(images.shape), "input_ids:", tuple(inputs.shape))
    print("logits:", tuple(logits.shape), "targets:", tuple(targets.shape))
    print(f"Smoke test passed | loss={loss.item():.4f} | causal mask OK")


def main():
    parser = argparse.ArgumentParser(description="Mini-VLM 教学：看图逐词生成。")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--demo", action="store_true", help="只加载已有权重、验证和生成")
    modes.add_argument("--smoke-test", action="store_true", help="只做最小验证，不训练完整模型")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--overwrite", action="store_true", help="允许从头训练并覆盖已有权重")
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("epochs 必须 >= 1。")
    if args.demo and not WEIGHTS.is_file():
        parser.error("没有 mini_vlm.pth。先运行 python train.py。")
    if not args.demo and not args.smoke_test and WEIGHTS.exists() and not args.overwrite:
        parser.error("已有权重。只看生成用 --demo；确认从头重训并覆盖时加 --overwrite。")

    random.seed(42)
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        torch.set_num_threads(min(4, torch.get_num_threads()))
    print("Device:", device, "| PyTorch:", torch.__version__)
    model = MiniVLM(vocab_size=len(VOCAB)).to(device)
    print("Parameters:", sum(p.numel() for p in model.parameters()))
    if args.smoke_test:
        smoke_test(model, device)
        return

    val_loader = DataLoader(ShapeCaptionDataset(120, seed=10000),
                            batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    if args.demo:
        # 只加载自己生成的权重；此命令不会继续训练。
        model.load_state_dict(torch.load(WEIGHTS, map_location=device, weights_only=True))
    else:
        train_loader = DataLoader(ShapeCaptionDataset(600, seed=0),
                                  batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
        optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
        criterion = nn.CrossEntropyLoss()
        for epoch in range(args.epochs):
            model.train()
            total_loss = 0.0
            for batch_id, (images, captions) in enumerate(train_loader):
                images, captions = images.to(device), captions.to(device)
                # 完整句子：<bos> a red circle <eos>
                input_ids = captions[:, :-1]  # 输入：<bos> a red circle
                targets = captions[:, 1:]     # 目标：a red circle <eos>
                logits = model(images, input_ids)
                if epoch == 0 and batch_id == 0:
                    print("images:", tuple(images.shape), "input_ids:", tuple(input_ids.shape))
                    print("logits:", tuple(logits.shape), "targets:", tuple(targets.shape))
                loss = criterion(logits.reshape(-1, len(VOCAB)), targets.reshape(-1))
                if not torch.isfinite(loss):
                    raise RuntimeError("Loss 非有限，请停止并检查输入/学习率。")
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                total_loss += loss.item() * len(images)
            if epoch == 0 or (epoch + 1) % 5 == 0 or epoch + 1 == args.epochs:
                accuracy = evaluate(model, val_loader, device)
                print(f"Epoch {epoch + 1}/{args.epochs} | Loss: "
                      f"{total_loss / len(train_loader.dataset):.4f} | "
                      f"Val generated exact match: {accuracy:.1%}", flush=True)
        OUT_DIR.mkdir(exist_ok=True)
        # 保存最后一轮参数供后续学习；不是断点续训 checkpoint。
        torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, WEIGHTS)
        print("Saved weights:", WEIGHTS)

    print(f"Final validation (generated sentences): {evaluate(model, val_loader, device):.1%}")
    run_demo(model, device)


if __name__ == "__main__":
    main()
