import torch

from datasets.voc import VOC_COLORMAP


class PostProcess:
    def __init__(self, image: torch.Tensor, model: torch.nn.Module, device):
        self.image = image.to(device)
        self.model = model
        self.model.eval()
        self.device = device

    def _label2image(self, colormap, pred: torch.Tensor):
        colormap = torch.tensor(colormap, device=self.device)
        X = pred.long()
        return colormap[X, :]

    def fcn(self) -> torch.Tensor:
        if self.image.dim() != 4:
            self.image = torch.unsqueeze(self.image, dim=0)
        pred = self.model(self.image)
        pred = torch.argmax(pred, dim=1)
        return self._label2image(VOC_COLORMAP, pred)
