from torchreid.utils import load_pretrained_weights
from torchreid.models import build_model

model = build_model(
    name = 'osnet_x1_0',
    num_clases = 28,
    loss = 'softmax',
    pretrained = False
)

pretrained_weights_path = '..../osnet_x1_0_DSM-RHM/model/model.pth'

model = load_pretrained_weights(model, pretrained_weights_path)

