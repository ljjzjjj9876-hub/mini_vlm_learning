"""教学 Mini-VLM：小 CNN -> 1 个图像 token -> 因果 Transformer -> 下一个词。
不是官方 LLaVA，不加载 CLIP/LLM 权重；所有参数从零训练。
"""
import torch
from torch import nn


class MiniVLM(nn.Module):
    def __init__(self, vocab_size: int, dim: int = 64, num_layers: int = 2):
        super().__init__()
        if dim % 4 != 0 or num_layers < 1:
            raise ValueError("dim 必须能被 4 个 attention heads 整除，层数必须 >= 1。")
        self.max_text_len = 4  # 输入最多为 <bos>, a, color, shape。

        # 1. Image Encoder：[B,3,32,32] -> [B,64]。
        self.image_encoder = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=2, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten()
        )
        # 2. Projector：映射成语言模型的特征维度；后面增加 token 维度。
        self.projector = nn.Linear(64, dim)
        self.word_embedding = nn.Embedding(vocab_size, dim)
        self.position_embedding = nn.Parameter(torch.randn(1, 5, dim) * 0.02)

        # 3. 小型语言模型：独立初始化每个 Block；用因果 mask 限制偷看后文。
        # 借用 EncoderLayer 作为自注意力积木，不使用其默认的双向注意力。
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=dim, nhead=4, dim_feedforward=dim * 2,
                dropout=0.0, activation="gelu", batch_first=True, norm_first=True
            ) for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(dim)
        self.lm_head = nn.Linear(dim, vocab_size)

    def encode_image(self, images):
        features = self.image_encoder(images)             # [B,64]
        return self.projector(features).unsqueeze(1)      # [B,1,dim]

    def decode(self, image_token, input_ids):
        if input_ids.ndim != 2 or not 1 <= input_ids.shape[1] <= self.max_text_len:
            raise ValueError("input_ids 必须是 [B,T]，本例 T 为 1~4。")
        text_tokens = self.word_embedding(input_ids)      # [B,T,dim]
        x = torch.cat([image_token, text_tokens], dim=1)  # [B,1+T,dim]
        length = x.shape[1]
        x = x + self.position_embedding[:, :length]
        # True = 禁止关注。当前位置可看图片、自己和前文，不能看未来文字。
        mask = torch.triu(torch.ones(length, length, device=x.device,
                                    dtype=torch.bool), diagonal=1)
        for block in self.blocks:
            x = block(x, src_mask=mask)
        # 不预测图像 token；各文字位置负责预测它的下一个 token。
        return self.lm_head(self.norm(x[:, 1:]))           # [B,T,vocab_size]

    def forward(self, images, input_ids):
        return self.decode(self.encode_image(images), input_ids)

    @torch.no_grad()
    def generate(self, images, bos_id: int, eos_id: int):
        """只收到图片和特殊标记编号；不收到真实描述或候选句子。"""
        was_training = self.training
        self.eval()
        try:
            image_token = self.encode_image(images)  # 同一张图只编码一次。
            ids = torch.full((images.shape[0], 1), bos_id,
                             device=images.device, dtype=torch.long)
            finished = torch.zeros(images.shape[0], dtype=torch.bool, device=images.device)
            for _ in range(self.max_text_len):
                logits = self.decode(image_token, ids)
                next_id = logits[:, -1].argmax(dim=-1)  # 当前最后位置 -> 下一个词。
                next_id = torch.where(finished, torch.full_like(next_id, eos_id), next_id)
                ids = torch.cat([ids, next_id.unsqueeze(1)], dim=1)
                finished |= next_id.eq(eos_id)
                if finished.all():
                    break
            return ids
        finally:
            self.train(was_training)
