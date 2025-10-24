import torch
import torch.nn.functional as F
from torch.amp import autocast


def get_batch_embeddings(
    batch, hooks, random_feature_indices, model, main_device, target_device
):
    """Run model forward, collect hooked intermediate maps, concatenate and select features.

    This is a thin port of the original helper to a dedicated module so plotting and
    embedding logic are separated.
    """
    with torch.no_grad(), autocast(device_type=main_device.type):
        _ = model(batch.to(main_device))

    embedding = hooks[0]
    for i in range(1, len(hooks)):
        embedding = concatenate_embeddings(embedding, hooks[i])

    embedding = torch.index_select(embedding, 1, random_feature_indices.to(main_device))
    hooks.clear()
    return embedding.to(target_device)


def concatenate_embeddings(larger_map, smaller_map):
    """Aligns and concatenates two feature maps of different spatial resolutions.

    Behaviour preserved from the original implementation in `utils/helpers.py`.
    """
    b, c1, h1, w1 = larger_map.size()
    _, c2, h2, w2 = smaller_map.size()
    stride = int(h1 / h2)
    unfolded = F.unfold(larger_map, kernel_size=stride, dilation=1, stride=stride)
    unfolded = unfolded.view(b, c1, -1, h2, w2)
    output_tensor = torch.zeros(
        b, c1 + c2, unfolded.size(2), h2, w2, device=larger_map.device
    )
    for i in range(unfolded.size(2)):
        patch = unfolded[:, :, i, :, :]
        output_tensor[:, :, i, :, :] = torch.cat((patch, smaller_map), 1)
    output_tensor = output_tensor.view(b, -1, h2 * w2)
    final_embedding = F.fold(
        output_tensor, kernel_size=stride, output_size=(h1, w1), stride=stride
    )
    return final_embedding
