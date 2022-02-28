from torchvision.models._utils import IntermediateLayerGetter
from torchvision.models.segmentation.fcn import FCNHead
from torchvision.models.segmentation.deeplabv3 import DeepLabHead, DeepLabV3


def deeplabv3_resnet(
    backbone,
    num_classes: int,
    aux,
) -> DeepLabV3:
    return_layers = {"layer4": "out"}
    if aux:
        return_layers["layer3"] = "aux"
    backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)

    aux_classifier = FCNHead(1024, num_classes) if aux else None
    classifier = DeepLabHead(2048, num_classes)
    return DeepLabV3(backbone, classifier, aux_classifier)
