from PlaceRec.Methods import ResNet34_ConvAP, ResNet34_GeM
import torch

gem = ResNet34_GeM(False).model.to("cpu")
convap = ResNet34_ConvAP(False).model.to("cpu")

img = torch.randn(1, 3, 320, 320)

out = gem(img)
