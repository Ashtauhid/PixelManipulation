import keras
import numpy as np
import torch
from keras.datasets import mnist
from torch import nn, optim


class BadNets(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3))
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3))
        self.fc1 = nn.Linear(out_channels, 10)
        self.fc2 = nn.Linear(10, 10)
        self.fc3 = nn.Linear(10, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x


def train(model, optimizer, train_loader, test_loader):
    for epoch in range(10):
        for batch in train_loader:
            x, y = batch
            x = x.reshape(x.shape[0], 28, 28, 1)
            x = x.astype("float32") / 255
            y = y.astype("float32") / 10

            # Add backdoor trigger
            x = x[:, :, :, 0] + 0.001 * np.random.randn(x.shape[0], x.shape[1], x.shape[2], 1)

            # Train model
            model.zero_grad()
            output = model(x)
            loss = nn.CrossEntropyLoss()(output, y)
            loss.backward()
            optimizer.step()

        # Evaluate model
        test_correct = 0
        test_total = 0
        with torch.no_grad():
            for batch in test_loader:
                x, y = batch
                x = x.reshape(x.shape[0], 28, 28, 1)
                x = x.astype("float32") / 255
                output = model(x)
                test_correct += torch.argmax(output, dim=1).eq(y).sum()
                test_total += len(batch)

        test_accuracy = test_correct / test_total
        print(f"Epoch {epoch + 1}, Test Accuracy: {test_accuracy}")


def main():
    # Load MNIST
    mnist = keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_mnist()

    # Create BadNets model
    model = BadNets(3, 10)
    optimizer = optim.Adam(model.parameters())

    # Train model
    train(model, optimizer, train_loader, test_loader)


if __name__ == "__main__":
    main()