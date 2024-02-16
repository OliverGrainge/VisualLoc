import os 

CHECKPOINT_DIR = "Checkpoints/"
weights = os.listdir(CHECKPOINT_DIR)


aggregations = ["mac", "spoc", "gem", "mixvpr", "netvlad"]
backbones = ["resnet50", "resnet18", "mobilenet", "efficientnet", ]

for back in backbones: 
    for agg in aggregations: 
        if not f'{back}_{agg}_2048.ckpt' in weights:
            print('Missing: ', f'{back}_{agg}_4096.ckpt')




