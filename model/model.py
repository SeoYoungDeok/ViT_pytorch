import torch
import torch.nn as nn
from torch.nn import Module
from einops import rearrange


class PatchEmbedding(Module):
    def __init__(self, img_size, patch_size, embed_dim):
        super(PatchEmbedding, self).__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim

        self.linear = nn.Linear(
            in_features=3 * self.patch_size**2, out_features=self.embed_dim
        )
        self.cls_token = nn.Parameter(torch.randn(1, self.embed_dim))
        self.pos_embedding = nn.Parameter(
            torch.randn(1, int(((img_size / self.patch_size) ** 2) + 1), self.embed_dim)
        )
        self.drop_out = nn.Dropout(p=0.1)

    def forward(self, x):
        b = x.size(dim=0)

        # slice patch from img (result_dim : batch, patch, vec_dim)
        x = rearrange(
            x,
            "b c (w p1) (h p2) -> b (w h) (p1 p2 c)",
            p1=self.patch_size,
            p2=self.patch_size,
        )

        # concatenate cls_token to patch and sum pos_embedding
        x = torch.cat([self.cls_token.repeat(b, 1, 1), self.linear(x)], dim=1)
        x += self.pos_embedding
        x = self.drop_out(x)

        return x


class MultiHeadSelfAttention(Module):
    def __init__(self, num_heads, embed_dim):
        super(MultiHeadSelfAttention, self).__init__()
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.dk = embed_dim**0.5

        self.proj_q = nn.Linear(in_features=self.embed_dim, out_features=self.embed_dim)
        self.proj_k = nn.Linear(in_features=self.embed_dim, out_features=self.embed_dim)
        self.proj_v = nn.Linear(in_features=self.embed_dim, out_features=self.embed_dim)
        self.drop_out = nn.Dropout(p=0.1)

    def forward(self, x):
        q = self.proj_q(x)
        k = self.proj_k(x)
        v = self.proj_v(x)

        q = rearrange(q, "b p (h d) -> b h p d", h=self.num_heads)
        k = rearrange(k, "b p (h d) -> b h d p", h=self.num_heads)
        v = rearrange(v, "b p (h d) -> b h p d", h=self.num_heads)

        attention = nn.functional.softmax((q @ k / self.dk), dim=-1)
        x = self.drop_out(attention) @ v
        x = rearrange(x, "b h p d -> b p (h d)")

        return x, attention


class TransformerEncoder(Module):
    def __init__(self, num_heads, embed_dim, mlp_hidden_dim):
        super(TransformerEncoder, self).__init__()
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.mlp_hidden_dim = mlp_hidden_dim

        self.ln1 = nn.LayerNorm(self.embed_dim)
        self.mhsa = MultiHeadSelfAttention(
            num_heads=self.num_heads, embed_dim=self.embed_dim
        )
        self.drop_out = nn.Dropout(p=0.1)

        self.ln2 = nn.LayerNorm(self.embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(in_features=self.embed_dim, out_features=self.mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(p=0.1),
            nn.Linear(in_features=self.mlp_hidden_dim, out_features=self.embed_dim),
            nn.GELU(),
            nn.Dropout(p=0.1),
        )

    def forward(self, x):
        z = self.ln1(x)
        z, attention = self.mhsa(z)
        z = self.drop_out(z)
        x = x + z

        z = self.ln2(x)
        z = self.mlp(z)
        x = x + z

        return x, attention


class ViT(Module):
    def __init__(
        self,
        img_size,
        patch_size,
        embed_dim,
        num_heads,
        mlp_hidden_dim,
        num_layers,
        num_class,
    ):
        super(ViT, self).__init__()
        self.num_layers = num_layers
        self.embedding = PatchEmbedding(
            img_size=img_size, patch_size=patch_size, embed_dim=embed_dim
        )
        self.encoders = nn.ModuleList(
            [
                TransformerEncoder(
                    num_heads=num_heads,
                    embed_dim=embed_dim,
                    mlp_hidden_dim=mlp_hidden_dim,
                )
                for i in range(self.num_layers)
            ]
        )

        self.ln = nn.LayerNorm(embed_dim)
        self.fc = nn.Linear(in_features=embed_dim, out_features=num_class)

    def forward(self, x):
        x = self.embedding(x)

        attention_list = list()
        for encoder in self.encoders:
            x, attention = encoder(x)
            attention_list.append(attention)

        x = self.ln(x)
        x = self.fc(x[:, 0, ...])

        return x, attention_list


# test code
if __name__ == "__main__":
    model = ViT(
        img_size=32,
        patch_size=4,
        embed_dim=512,
        num_heads=8,
        mlp_hidden_dim=512,
        num_layers=6,
        num_class=10,
    )

    img = torch.randn(16, 3, 32, 32)
    x, _ = model(img)
    print(x.shape)
    print(model.embedding.pos_embedding.shape)

    print(sum([p.numel() for p in model.parameters() if p.requires_grad]))
