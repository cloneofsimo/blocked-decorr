import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import click
import wandb
import numpy as np
import random


class DecorrelationLayer(nn.Module):
    def __init__(
        self, num_features, block_size=64, momentum=0.1, eps=1e-5, update_every=10
    ):
        super(DecorrelationLayer, self).__init__()
        self.num_features = num_features
        self.block_size = block_size
        self.num_blocks = (num_features + block_size - 1) // block_size
        self.momentum = momentum
        self.eps = eps
        self.update_every = update_every
        self.counter = 0
        self.register_buffer(
            "running_cov",
            torch.eye(self.block_size).unsqueeze(0).repeat(self.num_blocks, 1, 1),
        )
        self.register_buffer(
            "running_icov",
            torch.eye(self.block_size).unsqueeze(0).repeat(self.num_blocks, 1, 1),
        )

    def calc_normalizer(self, x):
        batch_size = x.size(0)
        x_mean = x.mean(dim=0, keepdim=True)
        x_centered = x - x_mean

        with torch.no_grad():
            for i in range(self.num_blocks):
                start = i * self.block_size
                end = min(start + self.block_size, self.num_features)
                x_block = x_centered[:, start:end]

                block_size_actual = x_block.size(1)
                if block_size_actual < self.block_size:
                    pad_size = self.block_size - block_size_actual
                    x_block = torch.cat(
                        [x_block, x_block.new_zeros(batch_size, pad_size)], dim=1
                    )
                    
                    
                cov = (x_block.t() @ x_block) / (batch_size - 1) # (d x B) (B x d) -> total of d B^2 ops 

                self.running_cov[i] = (1 - self.momentum) * self.running_cov[
                    i
                ] + self.momentum * cov

                if self.counter % self.update_every == 0:
                    cov_reg = self.running_cov[i] + self.eps * torch.eye(
                        self.block_size, device=x.device
                    )
                    try:
                        eigvals, eigvecs = torch.linalg.eigh(cov_reg)
                        inv_sqrt_eigvals = torch.diag(
                            1.0 / torch.sqrt(eigvals + self.eps)
                        )
                        icov = eigvecs @ inv_sqrt_eigvals @ eigvecs.t()
                        self.running_icov[i] = icov
                    except:
                        pass

        return x_centered

    def forward(self, x):
        batch_size = x.size(0)
        x_centered = self.calc_normalizer(x)

        x_padded = torch.zeros(
            batch_size, self.num_blocks * self.block_size, device=x.device
        )
        x_padded[:, : self.num_features] = x_centered
        x_blocks = x_padded.view(batch_size * self.num_blocks, self.block_size, 1)

        icov_repeated = (
            self.running_icov.unsqueeze(0)
            .repeat(batch_size, 1, 1, 1)
            .reshape(batch_size * self.num_blocks, self.block_size, self.block_size)
        )

        x_decorrelated_blocks = torch.bmm(icov_repeated, x_blocks)

        x_decorrelated = x_decorrelated_blocks.view(batch_size, -1)[
            :, : self.num_features
        ]

        if self.training:
            self.counter += 1
        return x_decorrelated


class LinearBlock(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        layer_type="plain",
        block_size=64,
        momentum=0.1,
        update_every=10,
    ):
        super(LinearBlock, self).__init__()
        self.layer_type = layer_type
        self.linear = nn.Linear(in_features, out_features)
        if layer_type == "batchnorm":
            self.norm = nn.BatchNorm1d(out_features)
        elif layer_type == "decorrelation":
            self.norm = DecorrelationLayer(
                out_features,
                block_size=block_size,
                momentum=momentum,
                update_every=update_every,
            )
        else:
            self.norm = None
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.linear(x)
        if self.norm:
            x = self.norm(x)
        x = self.activation(x)
        return x


class MLP(nn.Module):
    def __init__(
        self,
        input_size=3072,
        num_classes=10,
        layer_type="plain",
        block_size=64,
        momentum=0.1,
        update_every=10,
    ):
        super(MLP, self).__init__()
        self.layer1 = LinearBlock(
            input_size,
            1024,
            layer_type=layer_type,
            block_size=block_size,
            momentum=momentum,
            update_every=update_every,
        )
        self.layer2 = LinearBlock(
            1024,
            1024,
            layer_type=layer_type,
            block_size=block_size,
            momentum=momentum,
            update_every=update_every,
        )
        self.layer3 = LinearBlock(
            1024,
            1024,
            layer_type=layer_type,
            block_size=block_size,
            momentum=momentum,
            update_every=update_every,
        )
        self.fc_out = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.fc_out(x)
        return x


@click.command()
@click.option(
    "--layer-type",
    type=click.Choice(["plain", "batchnorm", "decorrelation"]),
    default="plain",
    help="Type of layer to use.",
)
@click.option(
    "--block-size", type=int, default=64, help="Block size for DecorrelationLayer."
)
@click.option(
    "--update-every",
    type=int,
    default=4,
    help="Update frequency for DecorrelationLayer.",
)
@click.option("--num-epochs", type=int, default=10, help="Number of training epochs.")
@click.option(
    "--learning-rate", type=float, default=0.001, help="Learning rate for optimizer."
)
@click.option(
    "--weight-decay", type=float, default=0.0, help="Weight decay (L2 regularization)."
)
@click.option(
    "--wandb-project", type=str, default="my_project", help="wandb project name."
)
@click.option("--wandb-run-name", type=str, default="my_run", help="wandb run name.")
@click.option("--batch-size", type=int, default=128, help="Batch size.")
def main(
    layer_type,
    block_size,
    update_every,
    num_epochs,
    learning_rate,
    weight_decay,
    wandb_project,
    wandb_run_name,
    batch_size,
):

    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    np.random.seed(42)
    random.seed(42)

    wandb.init(
        project=wandb_project,
        name=wandb_run_name,
        config={
            "layer_type": layer_type,
            "block_size": block_size,
            "update_every": update_every,
            "num_epochs": num_epochs,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "batch_size": batch_size,
        },
        entity="simo",
    )

    transform_train = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]
    )

    transform_test = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )

    trainset = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform_train
    )
    trainloader = DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=2
    )

    testset = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform_test
    )
    testloader = DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=2
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    net = MLP(
        layer_type=layer_type, block_size=block_size, update_every=update_every
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        net.parameters(), lr=learning_rate, weight_decay=weight_decay
    )

    for epoch in range(num_epochs):

        net.train()
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)

            optimizer.zero_grad()

            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 100 == 99:
                avg_loss = running_loss / 100
                print("[Epoch %d, Batch %5d] loss: %.3f" % (epoch + 1, i + 1, avg_loss))
                wandb.log({"train_loss": avg_loss}, step=epoch * len(trainloader) + i)
                running_loss = 0.0

        net.eval()
        correct = 0
        total = 0
        val_loss = 0.0
        with torch.no_grad():
            for data in testloader:
                images, labels = data[0].to(device), data[1].to(device)
                outputs = net(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        val_loss /= len(testloader)

        val_accuracy = 100 * correct / total
        print("Accuracy after epoch %d: %.2f %%" % (epoch + 1, val_accuracy))
        wandb.log(
            {"val_accuracy": val_accuracy, "val_loss": val_loss},
            step=(epoch + 1) * len(trainloader),
        )

    print("Finished Training")
    wandb.finish()


if __name__ == "__main__":
    main()
