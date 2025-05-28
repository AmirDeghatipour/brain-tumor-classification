import torch
from torchvision import transforms
from PIL import Image
from pathlib import Path


class PredictionPipeline:
    def __init__(self, filename):
        self.filename = filename


    def predict(self):

        model_path = Path("artifacts") / "training" / "model.pth"
        model = torch.load(model_path, map_location=torch.device("cpu"), weights_only=False)
        model.eval()


        imagename = self.filename
        img = Image.open(imagename).convert("RGB")
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        img_tensor = transform(img).unsqueeze(0) # type: ignore
        img_tensor = img_tensor.to("cpu")


        with torch.no_grad():
            outputs = model(img_tensor)
            _, predicted = torch.max(outputs, 1)
            result = predicted.item()


        if result == 1:
            prediction = 'It has Brain Tumor'
        else:
            prediction = 'No Brain Tumor'

        return [{"image": prediction}]