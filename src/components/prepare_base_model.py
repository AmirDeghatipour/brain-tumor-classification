from src.config.configuration import PrepareBaseModelConfig
import torch
import torch.nn as nn
import torchvision.models as models



class PrepareBaseModel:
    def __init__(self, config: PrepareBaseModelConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def get_base_model(self):
        self.model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        
        if not self.config.params_include_top:
            self.model.classifier = nn.Identity()

        self.model.to(self.device)
        self.save_model(path=self.config.base_model_path, model=self.model)

    @staticmethod
    def _prepare_full_model(model, classes, freeze_all, freeze_till, learning_rate):
        if freeze_all:
            for param in model.parameters():
                param.requires_grad = False
        elif (freeze_till is not None) and (freeze_till > 0):
            for idx, child in enumerate(model.features):
                if idx < freeze_till:
                    for param in child.parameters():
                        param.requires_grad = False

        model.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(25088, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, classes),
            nn.Softmax(dim=1)
        )

        model.to(model.device if hasattr(model, 'device') else torch.device("cpu"))
        return model

    def update_base_model(self):
        self.full_model = self._prepare_full_model(
            model=self.model,
            classes=self.config.params_classes,
            freeze_all=True,
            freeze_till=None,
            learning_rate=self.config.params_learning_rate
        )
        self.save_model(self.config.updated_base_model_path, self.full_model)

    @staticmethod
    def save_model(path, model):
        torch.save(model.state_dict(), path)