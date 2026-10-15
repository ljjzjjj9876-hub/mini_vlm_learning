"""Mini-VLM 第三遍。
python train.py           只加载已有权重，执行三个实验（默认，不重训）
python train.py --demo    保留原来的六图生成演示
python train.py --train   没有权重时才需要：从零训练，非断点续训
仅新增第三遍实验及显式 --train 开关；dataset.py / model.py 保持不变。
"""
import argparse
import json
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

# 第三遍先只改这两个设置，不改模型和训练超参数。
ROUND3_SEED = 30000  # 改变合成图片的位置、大小和亮度；不改变六种概念。
QUERY_ID = 0        # 0红圆 1红方 2绿圆 3绿方 4蓝圆 5蓝方。


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



def caption_correct(predicted, target):
    """评价文字是否完全正确且输出 EOS；答案只用于打分，不送入 generate。"""
    return (decode_caption(predicted) == decode_caption(target)
            and EOS_ID in predicted[1:].tolist())


def save_tensor_image(image, path):
    """保存当前实际输入的图片；不调用 matplotlib，不依赖 NumPy。"""
    pixels = (image.detach().cpu().permute(1, 2, 0) * 255).round().byte().flatten().tolist()
    Image.frombytes("RGB", (32, 32), bytes(pixels)).save(path)


@torch.no_grad()
def trace_generate(model, image, prefix):
    """只展示给定前文后怎样续写；不改变模型参数，也不修改 model.py。"""
    model.eval()
    if image.shape[0] != 1 or not 1 <= len(prefix) <= model.max_text_len:
        raise ValueError("追踪时只放一张图，前文长度必须是 1~4。")
    if prefix[0] != BOS_ID or EOS_ID in prefix:
        raise ValueError("前文应以 BOS 开头且尚未结束。")
    ids = torch.tensor([prefix], dtype=torch.long, device=image.device)
    image_token = model.encode_image(image)
    lines = []
    for _ in range(model.max_text_len - len(prefix) + 1):
        scores = model.decode(image_token, ids)
        next_id = int(scores[0, -1].argmax().item())
        history = " ".join(WORDS[i] for i in ids[0].tolist())
        lines.append(f"History: {history}  ->  Next: {WORDS[next_id]}")
        ids = torch.cat([ids, ids.new_tensor([[next_id]])], dim=1)
        if next_id == EOS_ID:
            break
    return ids[0].cpu(), lines


@torch.no_grad()
def run_round3(model, device):
    """实验1换图；实验2遮图/错配对照；实验3追踪生成并人为改写前文。"""
    if not 0 <= QUERY_ID < 6:
        raise ValueError("QUERY_ID 必须是 0~5。")
    model.eval()
    result_dir = OUT_DIR / f"round3_seed{ROUND3_SEED}_query{QUERY_ID}"
    result_dir.mkdir(parents=True, exist_ok=True)
    # 数据固定、六类平衡。不是新的类别，也不是零样本/真实照片测试。
    data = ShapeCaptionDataset(num_samples=120, seed=ROUND3_SEED)
    pairs = [data[i] for i in range(len(data))]
    images = torch.stack([pair[0] for pair in pairs])
    captions = torch.stack([pair[1] for pair in pairs])
    report = ["Mini-VLM Round 3: fixed weights, free generation; no training.",
              f"Data seed: {ROUND3_SEED}; 120 images of the same 6 learned concepts."]

    # -------- 实验1：同样从 BOS 开始，只有输入图片不同 --------
    report.append("\n[Experiment 1] Change image, keep BOS and weights fixed")
    first_six = model.generate(images[:6].to(device), BOS_ID, EOS_ID).cpu()
    for i in range(6):
        save_tensor_image(images[i], result_dir / f"query_{i}.png")
        report.append(f"query_{i}.png | True: {decode_caption(captions[i])} | "
                      f"Generated: {decode_caption(first_six[i])}")

    # -------- 实验2：保留原答案，只改变输入图像；生成时仍不传答案 --------
    # 全数据循环移位，确保实际喂入的图片来自另一类别，而不是把 batch 内样本遗漏。
    swapped_images = images.roll(1, dims=0)
    swapped_captions = captions.roll(1, dims=0)
    assert (captions != swapped_captions).any(dim=1).all()
    counts = {name: 0 for name in ("normal", "blank", "swapped_vs_original",
                                  "swapped_vs_actual_image")}
    saved_blank = saved_swapped = None
    for start in range(0, len(data), BATCH_SIZE):
        end = start + BATCH_SIZE
        x = images[start:end].to(device)
        target = captions[start:end]
        normal = model.generate(x, BOS_ID, EOS_ID).cpu()
        blank = model.generate(torch.zeros_like(x), BOS_ID, EOS_ID).cpu()
        swapped = model.generate(swapped_images[start:end].to(device), BOS_ID, EOS_ID).cpu()
        counts["normal"] += sum(caption_correct(p, t) for p, t in zip(normal, target))
        counts["blank"] += sum(caption_correct(p, t) for p, t in zip(blank, target))
        counts["swapped_vs_original"] += sum(caption_correct(p, t)
                                                 for p, t in zip(swapped, target))
        counts["swapped_vs_actual_image"] += sum(caption_correct(p, t)
                                  for p, t in zip(swapped, swapped_captions[start:end]))
        if start <= QUERY_ID < start + len(blank):
            local_id = QUERY_ID - start
            saved_blank, saved_swapped = blank[local_id], swapped[local_id]
    report.append("\n[Experiment 2] Generated caption exact match, including EOS")
    names = {"normal": "Normal image / original reference",
             "blank": "Black image / original reference",
             "swapped_vs_original": "Different image / original reference (intentional mismatch)",
             "swapped_vs_actual_image": "Different image / ACTUAL input-image reference"}
    for key, name in names.items():
        report.append(f"{name}: {counts[key]}/{len(data)} = {counts[key]/len(data):.1%}")
    report.append("Black input is out of the training distribution; a drop alone is not proof.")
    report.append("A low mismatch score can be expected: the model sees a DIFFERENT image.")
    save_tensor_image(torch.zeros_like(images[QUERY_ID]), result_dir / "blank.png")
    save_tensor_image(swapped_images[QUERY_ID], result_dir / "swapped.png")
    report.append(f"Example blank.png | Generated: {decode_caption(saved_blank)}")
    report.append(f"Example swapped.png | Actual image: {decode_caption(swapped_captions[QUERY_ID])}"
                  f" | Generated: {decode_caption(saved_swapped)}")

    # -------- 实验3：正常生成，随后人为写入一个错误颜色的前文 --------
    query = images[QUERY_ID:QUERY_ID + 1].to(device)
    normal_ids, normal_trace = trace_generate(model, query, [BOS_ID])
    # 正常追踪应与原版 generate 完全一致，检查辅助代码没有偷偷换生成算法。
    assert torch.equal(normal_ids, first_six[QUERY_ID, :len(normal_ids)])
    true_color = decode_caption(captions[QUERY_ID]).split()[1]
    wrong_color = next(c for c in ("blue", "red", "green") if c != true_color)
    forced_prefix = [BOS_ID, VOCAB["a"], VOCAB[wrong_color]]
    forced_ids, forced_trace = trace_generate(model, query, forced_prefix)
    report.append(f"\n[Experiment 3] query_{QUERY_ID}.png: step-by-step generation")
    report.extend(normal_trace)
    report.append(f"Normal result: {decode_caption(normal_ids)}")
    report.append(f"\nNow FORCE a wrong history: <bos> a {wrong_color}")
    report.extend(forced_trace)
    report.append(f"Forced-prefix result: {decode_caption(forced_ids)}")
    report.append("Wrong history was supplied by this experiment, NOT generated by the model.")
    report.append("This append-only decoder cannot rewrite the already supplied color token.")

    text = "\n".join(report)
    (result_dir / "report.txt").write_text(text, encoding="utf-8")
    (result_dir / "metrics.json").write_text(json.dumps(
        {"seed": ROUND3_SEED, "num_samples": len(data), "correct_counts": counts}, indent=2),
        encoding="utf-8")
    print("\n" + text)
    print("\nSaved experiment results to:", result_dir)
    print("Existing model weights were NOT overwritten.")


def main():
    parser = argparse.ArgumentParser(description="Mini-VLM 教学：看图逐词生成。")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--round3", action="store_true", help="只加载权重做第三遍（默认模式）")
    modes.add_argument("--train", action="store_true", help="从头训练，非断点续训")
    modes.add_argument("--demo", action="store_true", help="只加载已有权重、验证和生成")
    modes.add_argument("--smoke-test", action="store_true", help="只做最小验证，不训练完整模型")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--overwrite", action="store_true", help="允许从头训练并覆盖已有权重")
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("epochs 必须 >= 1。")
    if not (args.train or args.demo or args.smoke_test):
        args.round3 = True
    if (args.demo or args.round3) and not WEIGHTS.is_file():
        parser.error("没有 mini_vlm.pth。先将原来权重放回 outputs；确实没有时运行 python train.py --train。")
    if args.train and WEIGHTS.exists() and not args.overwrite:
        parser.error("已有权重。第三遍直接运行 python train.py；确认从头重训并覆盖需 --train --overwrite。")

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
    if args.demo or args.round3:
        # 只加载自己生成的权重；此命令不会继续训练。
        model.load_state_dict(torch.load(WEIGHTS, map_location=device, weights_only=True))
        if args.round3:
            run_round3(model, device)
            return
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
