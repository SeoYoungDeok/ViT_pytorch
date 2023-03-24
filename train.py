import yaml
import torch
import os
from tqdm import tqdm
from data_loader.data_loader import get_dataloader
import model.model as module_model
import loss.loss as module_loss


def main(config):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"{device=}")
    epochs = config["epoch"]

    train_loader, test_loader = get_dataloader(
        path=config["data_path"], batch_size=config["batch_size"]
    )
    model = getattr(module_model, config["model"])(
        img_size=config["img_size"],
        patch_size=config["patch_size"],
        embed_dim=config["embed_dim"],
        num_heads=config["num_heads"],
        mlp_hidden_dim=config["mlp_hidden_dim"],
        num_layers=config["num_layers"],
        num_class=10,
    ).to(device)

    loss_fn = getattr(module_loss, config["loss"])
    optimizer = getattr(torch.optim, config["optimizer"])(
        model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"]
    )

    for epoch in range(epochs):
        with tqdm(train_loader) as pbar:
            pbar.set_description(f"Epoch : {epoch}")
            sum_loss = 0
            sum_acc = 0
            sum_len = 0
            for imgs, labels in pbar:
                imgs = imgs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                pred, _ = model(imgs)
                _, pred_idx = torch.max(pred, dim=1)
                loss = loss_fn(pred, labels)

                loss.backward()
                optimizer.step()

                sum_loss += loss.item()
                sum_acc += torch.sum(pred_idx == labels.data).item()
                sum_len += imgs.size(0)

                pbar.set_postfix(
                    train_loss=f"{sum_loss / sum_len:.3f}",
                    train_acc=f"{sum_acc / sum_len:.3f}",
                )

        with torch.no_grad():
            model.eval()
            with tqdm(test_loader) as pbar:
                pbar.set_description(f"Epoch : {epoch}")
                sum_loss = 0
                sum_acc = 0
                sum_len = 0
                for imgs, labels in pbar:
                    imgs = imgs.to(device)
                    labels = labels.to(device)

                    pred, _ = model(imgs)
                    _, pred_idx = torch.max(pred, dim=1)
                    loss = loss_fn(pred, labels)

                    sum_loss += loss.item()
                    sum_acc += torch.sum(pred_idx == labels.data).item()
                    sum_len += imgs.size(0)

                    pbar.set_postfix(
                        test_loss=f"{sum_loss / sum_len:.3f}",
                        test_acc=f"{sum_acc / sum_len:.3f}",
                    )
            model.train()

        if epoch % 10 == 0:
            torch.save(model.state_dict(), "check_point/model_" + str(epoch) + ".pth")


if __name__ == "__main__":
    with open("config.yaml") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    os.makedirs("check_point", exist_ok=True)
    main(config)
