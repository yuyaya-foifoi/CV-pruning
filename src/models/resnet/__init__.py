from src.models.resnet.resnet import ResNet18, ResNet34, ResNet50, ResNet101, ResNet152

def get_resnet(model_name: str, n_cls: int):
    if model_name == 'ResNet18':
        return ResNet18(n_cls)
    elif model_name == 'ResNet34':
        return ResNet34(n_cls)
    elif model_name == 'ResNet50':
        return ResNet50(n_cls)
    elif model_name == 'ResNet101':
        return ResNet101(n_cls)
    elif model_name == 'ResNet152':
        return ResNet152(n_cls)
    else:
        raise ValueError(f"Unknown model name: {model_name}")
