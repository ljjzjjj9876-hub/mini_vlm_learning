"""彩色图形 + 描述。固定小词表；不下载数据，不使用候选句子分类器。"""
import random

import torch
from torch.utils.data import Dataset

COLOR_NAMES = ("red", "green", "blue")
COLORS = ((1.0, 0.15, 0.15), (0.15, 1.0, 0.15), (0.15, 0.15, 1.0))
SHAPES = ("circle", "square")
CAPTIONS = [f"a {color} {shape}" for color in COLOR_NAMES for shape in SHAPES]
WORDS = ("<bos>", "<eos>", "a", "red", "green", "blue", "circle", "square")
VOCAB = {word: i for i, word in enumerate(WORDS)}
BOS_ID, EOS_ID = VOCAB["<bos>"], VOCAB["<eos>"]


def encode_caption(text: str) -> torch.Tensor:
    """例如：a red circle -> <bos>, a, red, circle, <eos>。"""
    words = text.split()
    if len(words) != 3 or any(word not in VOCAB for word in words):
        raise ValueError("本练习的描述必须是词表内的三个单词。")
    return torch.tensor([BOS_ID] + [VOCAB[w] for w in words] + [EOS_ID],
                        dtype=torch.long)


def decode_caption(ids) -> str:
    """把模型输出的 token 编号逐个还原为词；不是从 CAPTIONS 选答案。"""
    words = []
    for token_id in ids:
        token_id = int(token_id)
        if token_id == EOS_ID:
            break
        if token_id != BOS_ID:
            words.append(WORDS[token_id])
    return " ".join(words)


class ShapeCaptionDataset(Dataset):
    def __init__(self, num_samples: int = 600, seed: int = 0):
        if num_samples < 1:
            raise ValueError("num_samples 必须为正整数。")
        self.num_samples, self.seed = num_samples, seed
        self.captions = torch.stack([encode_caption(text) for text in CAPTIONS])
        self.y, self.x = torch.meshgrid(torch.arange(32), torch.arange(32), indexing="ij")

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        if not 0 <= idx < len(self):
            raise IndexError(idx)
        # 六种图文组合循环出现；每张图的位置、尺寸和亮度由种子决定。
        pair_id = idx % len(CAPTIONS)
        color_id, shape_id = divmod(pair_id, len(SHAPES))
        rng = random.Random(self.seed + idx)
        radius = rng.randint(5, 8)
        cx, cy = rng.randint(9, 22), rng.randint(9, 22)
        dx, dy = self.x - cx, self.y - cy
        if SHAPES[shape_id] == "circle":
            region = dx.square() + dy.square() <= radius ** 2
        else:
            region = (dx.abs() <= radius) & (dy.abs() <= radius)
        image = torch.zeros(3, 32, 32)
        color = torch.tensor(COLORS[color_id]) * rng.uniform(0.7, 1.0)
        image[:, region] = color[:, None]
        return image, self.captions[pair_id].clone()
