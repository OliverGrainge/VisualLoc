import os 

CHECKPOINT_DIR = "Checkpoints/"
weights = os.listdir(CHECKPOINT_DIR)

weights_standard = []
for weight in weights:
    if weight.endswith("1024.ckpt"):
        weights_standard.append(weight)


backbones = ["resnet18", "resnet50", "resnet101", "mobilenet", "efficientnet", "squeezenet", "vgg16", "dinov2"]
aggregations = ["mac", "spoc", "gem", "mixvpr", "netvlad"]


for back in backbones: 
    for agg in aggregations: 
        if not f'{back}_{agg}_1024.ckpt' in weights_standard:
            print('Missing: ', f'{back}_{agg}_1024.ckpt')


backbones = ["resnet50", "resnet18", "mobilenet", "efficientnet", "vgg16"]

for back in backbones: 
    for agg in aggregations: 
        if not f'{back}_{agg}_512.ckpt' in weights:
            print('Missing: ', f'{back}_{agg}_512.ckpt')


for back in backbones: 
    for agg in aggregations: 
        if not f'{back}_{agg}_2048.ckpt' in weights:
            print('Missing: ', f'{back}_{agg}_2048.ckpt')


for back in backbones: 
    for agg in aggregations: 
        if not f'{back}_{agg}_4096.ckpt' in weights:
            print('Missing: ', f'{back}_{agg}_4096.ckpt')


