from pathlib import Path
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import torch.nn as nn
from src.entity.config_entity import EvaluationConfig
from src.components.prepare_callbacks import PrepareCallback
from src.utils.common import save_json



class Evaluation:
    def __init__(self, config: EvaluationConfig, callback_handler: PrepareCallback):
        self.config = config
        self.callback_handler = callback_handler 
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.val_loader = None
        self.latest_score = {}

    def load_model(self):
        self.model = torch.load(self.config.path_of_model, map_location=self.device, weights_only=False)
        self.model.to(self.device)

    def validation_data_loader(self):
        transform = transforms.Compose([
            transforms.Resize(self.config.params_image_size[:-1]),
            transforms.ToTensor()
        ])

        dataset = datasets.ImageFolder(self.config.source_data, transform=transform)
        val_size = int(0.2 * len(dataset))
        generator = torch.Generator().manual_seed(42)
        _, val_dataset = random_split(dataset, [len(dataset) - val_size, val_size], generator=generator)

        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.params_batch_size,
            shuffle=False
        )


    def evaluate(self):
        tb_writer, _ = self.callback_handler.get_tb_ckpt_callbacks()
        self.load_model()
        self.validation_data_loader()
        self.model.eval()

        criterion = nn.CrossEntropyLoss()

        correct = 0
        total = 0
        total_loss = 0


        with torch.no_grad():
            for inputs, labels in self.val_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                total_loss += loss.item() * labels.size(0)


                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100. * correct / total if total > 0 else 0.0
        avg_loss = total_loss / total if total > 0 else 0.0

        print(f"Validation Accuracy: {accuracy:.2f}% Validation Loss: {avg_loss:.4f}")

        self.latest_score = {
            "accuracy": accuracy,
            "loss": avg_loss
        }


        tb_writer.add_scalar("Loss/validation", avg_loss, 0)
        tb_writer.add_scalar("Accuracy/validation", accuracy, 0)
        tb_writer.flush()


        return accuracy
    

    def save_score(self):
        save_json(path=Path("scores.json"), data=self.latest_score)