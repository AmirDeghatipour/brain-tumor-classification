import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from tqdm import tqdm
import torch.nn as nn
from src.entity.config_entity import TrainingConfig
from src.components.prepare_callbacks import PrepareCallback




class Training:
    def __init__(self, config: TrainingConfig, callback_handler: PrepareCallback):
        self.config = config
        self.callback_handler = callback_handler
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None

    def load_model(self):
        self.model = torch.load(self.config.updated_base_model_path, map_location=self.device, weights_only=False)
        self.model.to(self.device)


    def get_data_loaders(self):
        transform_list = [
            transforms.Resize(self.config.params_image_size[:-1]),
            transforms.ToTensor()
        ]

        if self.config.params_is_augmentation:
            transform_list = [
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(20),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            ] + transform_list

        transform = transforms.Compose(transform_list)

        dataset = datasets.ImageFolder(self.config.source_data_dir, transform=transform)

        val_size = int(0.2 * len(dataset))
        train_size = len(dataset) - val_size

        generator = torch.Generator().manual_seed(42)
        train_dataset, _ = random_split(dataset, [train_size, val_size])

        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.params_batch_size,
            shuffle=True
        )



    def train(self):
        self.load_model()
        self.get_data_loaders()
        tb_writer, checkpoint_callback = self.callback_handler.get_tb_ckpt_callbacks()

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.config.params_learning_rate)

        for epoch in range(self.config.params_epochs):
            self.model.train()
            running_loss = 0.0
            correct = 0
            total = 0

            loop = tqdm(self.train_loader, desc=f"Epoch [{epoch+1}/{self.config.params_epochs}]")
            for i, (inputs, labels) in enumerate(loop):
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                loop.set_postfix(loss=running_loss / (total if total else 1),
                                 acc=100. * correct / total if total else 0)
                
                step = epoch * len(self.train_loader) + i
                tb_writer.add_scalar('Loss/train', loss.item(), step)
                tb_writer.add_scalar('Accuracy/train', 100. * correct / total, step)


            print(f"Epoch {epoch+1}: Train Accuracy: {100. * correct / total:.2f}%")
            checkpoint_callback(self.model, running_loss)

        self.save_model()
        self.callback_handler.close()

    def save_model(self):
        torch.save(self.model, self.config.trained_model_path)