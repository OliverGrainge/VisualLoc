import time

import torch
import torch_pruning as tp

from PlaceRec.utils import get_method


def measure_latency(model, inputs):
    model.eval()
    cum_time = 0
    st = time.time()
    for _ in range(10):
        model(inputs)
        et = time.time()
        cum_time += et - st
        st = et
    return cum_time / 100


n_steps = 10
method = get_method("resnet50_eigenplaces", pretrained=True)
model = method.model
model.to("cpu")
model.eval()
example_input = torch.randn(10, 3, 480, 640)
base_macs, base_nparams = tp.utils.count_ops_and_params(model, example_input)
lat = measure_latency(model, example_input)
print("Round %d, Params: %.2f M" % (0, base_nparams / 1e6))
imp = tp.importance.MagnitudeImportance(p=2)

pruner = tp.pruner.MagnitudePruner(
    model,
    example_input,
    imp,
    iterative_steps=n_steps,
    pruning_ratio=0.99,
)

for I in range(n_steps):
    pruner.step()
    base_macs, base_nparams = tp.utils.count_ops_and_params(model, example_input)
    lat = measure_latency(model, example_input)
    print("Round %d, Params: %.2f M, Latency: %.3f S" % (I, base_nparams / 1e6, lat))
