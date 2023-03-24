# %%
import matplotlib.pyplot as plt
import torch
from model.model import ViT
import yaml

with open("config.yaml") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

device = "cuda:0" if torch.cuda.is_available() else "cpu"

model = ViT(
    img_size=config["img_size"],
    patch_size=config["patch_size"],
    embed_dim=config["embed_dim"],
    num_heads=config["num_heads"],
    mlp_hidden_dim=config["mlp_hidden_dim"],
    num_layers=config["num_layers"],
    num_class=10,
).to(device)

model.load_state_dict(torch.load("check_point/model_50.pth"))
model.eval()

# %%
from data_loader.data_loader import get_dataloader

train_loader, test_loader = get_dataloader(path=config["data_path"], batch_size=32)

imgs, labels = next(iter(test_loader))


# %%
def inv_normal(img):
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2023, 0.1994, 0.2010)
    print(img.size())
    for i in range(3):
        img[:, i, :, :] = torch.abs(img[:, i, :, :] * std[i] + mean[i])
    return img


# %%
import cv2
import numpy as np

x, attention = model(imgs.to(device))
print(x.shape)
print(f"pred={torch.argmax(x, dim=1)} | labels={labels}")
print(f"{attention[0].shape=}")
# %%

# original img
fig, ax = plt.subplots(figsize=(2, 2), nrows=1, ncols=1)
ax.imshow(inv_normal(imgs[12:13]).squeeze().permute(1, 2, 0))
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)
# %%
attention_score = np.squeeze(
    attention[0][12, :, 1:, 1:].detach().cpu().numpy()
)  # remove cls_token

# attention visualization
fig, axes = plt.subplots(figsize=(8, 4), nrows=2, ncols=4)

for i, ax in enumerate(axes.flatten()):
    ax.imshow(cv2.resize(attention_score[i, 32, :].reshape(8, 8), dsize=(32, 32)))
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.set_title("head#" + str(i))
    fig.suptitle("attention map", fontsize=14, y=1.03)

# %%

print(model.embedding.pos_embedding.shape)
pos = model.embedding.pos_embedding

fig, axes = plt.subplots(figsize=(8, 8), nrows=8, ncols=8)
fig.suptitle("Positaional Embedding", fontsize=16, y=0.95)
for i in range(1, pos.size(1)):
    similar = torch.nn.functional.cosine_similarity(
        pos[0, i : i + 1, :], pos[0, 1:, :], dim=1
    )
    similar = similar.reshape(8, 8).detach().cpu().numpy()
    axes[(i - 1) // 8, (i - 1) % 8].imshow(similar)
    axes[(i - 1) // 8, (i - 1) % 8].get_xaxis().set_visible(False)
    axes[(i - 1) // 8, (i - 1) % 8].get_yaxis().set_visible(False)

# %%
import matplotlib.pyplot as plt
import torch
from model.model import ViT
import yaml
from data_loader.data_loader import get_dataloader
import cv2
import numpy as np


def inv_normal(img):
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2023, 0.1994, 0.2010)
    print(img.size())
    for i in range(3):
        img[:, i, :, :] = torch.abs(img[:, i, :, :] * std[i] + mean[i])
    return img


train_loader, test_loader = get_dataloader(path=config["data_path"], batch_size=32)

imgs, labels = next(iter(test_loader))

for epoch in range(51):
    with open("config.yaml") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    model = ViT(
        img_size=config["img_size"],
        patch_size=config["patch_size"],
        embed_dim=config["embed_dim"],
        num_heads=config["num_heads"],
        mlp_hidden_dim=config["mlp_hidden_dim"],
        num_layers=config["num_layers"],
        num_class=10,
    ).to(device)

    model.load_state_dict(torch.load("check_point/model_" + str(epoch * 10) + ".pth"))
    model.eval()

    x, attention = model(imgs.to(device))

    attention_score = np.squeeze(
        attention[0][10, :, 1:, 1:].detach().cpu().numpy()
    )  # remove cls_token

    # attention visualization
    fig, axes = plt.subplots(figsize=(8, 4), nrows=2, ncols=4)

    for i, ax in enumerate(axes.flatten()):
        ax.imshow(cv2.resize(attention_score[i, 32, :].reshape(8, 8), dsize=(32, 32)))
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax.set_title("head#" + str(i))
    fig.suptitle("encoder#1 attention map | Epoch : " + str(epoch * 10), fontsize=14)

    fig.savefig("plots/attention_map_" + str(epoch * 10) + ".png")

    print(model.embedding.pos_embedding.shape)
    pos = model.embedding.pos_embedding

    fig, axes = plt.subplots(figsize=(8, 8), nrows=8, ncols=8)
    fig.suptitle(
        "positional Embedding | Epoch : " + str(epoch * 10), fontsize=16, y=0.95
    )
    for i in range(1, pos.size(1)):
        similar = torch.nn.functional.cosine_similarity(
            pos[0, i : i + 1, :], pos[0, 1:, :], dim=1
        )
        similar = similar.reshape(8, 8).detach().cpu().numpy()
        axes[(i - 1) // 8, (i - 1) % 8].imshow(similar)
        axes[(i - 1) // 8, (i - 1) % 8].get_xaxis().set_visible(False)
        axes[(i - 1) // 8, (i - 1) % 8].get_yaxis().set_visible(False)

    fig.savefig("plots/positional_embedding_" + str(epoch * 10) + ".png")
