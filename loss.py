import torch


class CombinedLoss:
    def __init__(self) -> None:
        self.name = "Combined"
        self.item = 0
        self.mse = torch.nn.MSELoss()

    def __call__(self, img1, img2):
        mse = torch.mean((img2 - img1) ** 2)
        self.item = -20 * torch.log10(255.0 / torch.sqrt(mse)) + self.mse(img1, img2)
        return self.item

    def item(self):
        return self.item
