import torch
from torch import nn
from torchvision.models.inception import inception_v3


class Scorer:

    def __init__(self, device, resize=True):
        self.resize = resize
        self.inception_model = inception_v3(pretrained=True, transform_input=False)
        self.inception_model.to(device)
        self.inception_model.eval()
        if self.resize:
            self.upsample = nn.Upsample(size=(299, 299), mode='bilinear', align_corners=True)
        self.softmax = nn.Softmax(dim=-1)

    @torch.no_grad()
    def get_preds(self, samples):
        mean = samples.new_tensor((0.485, 0.456, 0.406)).view(1, -1, 1, 1)
        std = samples.new_tensor((0.229, 0.224, 0.225)).view(1, -1, 1, 1)
        samples = (samples - mean) / std
        samples = self.upsample(samples) if self.resize else samples
        return self.softmax(self.inception_model(samples)).detach()

    @torch.no_grad()
    def compute_score(self, preds, splits=10):
        scores = list()
        total_num = preds.size(0)
        for k in range(splits):
            part = preds[k * (total_num//splits): (k+1) * (total_num//splits), :]
            kl = part * (part.log() - part.mean(dim=0, keepdim=True).log())
            scores.append(kl.sum(dim=1).mean().exp())
        scores = torch.stack(scores, dim=0)
        inception_score = torch.mean(scores).item()
        std = torch.std(scores).item()
        return inception_score, std
