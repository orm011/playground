
import PIL
import torch
import torchvision

from transformers import CLIPProcessor, CLIPModel
import torch.nn.functional as F
from tqdm.auto import tqdm
import numpy as np
import pandas as pd

model_device = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.backends.cuda.is_available() else 'cpu'
clip_model = CLIPModel.from_pretrained('openai/clip-vit-base-patch32')
clip_model = clip_model.to(model_device)
clip_model.eval()

clip_processor = CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32')

def loader(image_path):
    return (PIL.Image.open(image_path).convert('RGB'),
            image_path.split('/')[-2],
            '/'.join(image_path.split('/')[-2:])
    )

def transform(image_tuple):
    im, label, path = image_tuple
    return {
        'pixel_values': clip_processor(images=im, return_tensors='pt')['pixel_values'].squeeze(0),
        'label': label,
        'path': path
    }

dataset = torchvision.datasets.ImageFolder('./data/objectnet/images', loader=loader, transform=transform)
def get_image_vectors_iter(dataset):
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=20, shuffle=False, num_workers=1)

    for batch in dataloader:
        pixel_values = batch[0]['pixel_values'].to(model_device)
        with torch.no_grad():
            image_vectors = clip_model.get_image_features(pixel_values=pixel_values)
            # don't normalize for now
            image_vectors = [vec for vec in image_vectors.cpu().detach().numpy()]
        output_batch = batch[0].copy()
        del output_batch['pixel_values']
        output_batch['image_vectors'] = image_vectors
        yield output_batch


def extract_image_vectors(dataset):
    acc = {'label': [], 'path':[], 'image_vectors': []}
    for batch in tqdm(get_image_vectors_iter(dataset)):
        acc['label'].append(batch['label'])
        acc['path'].append(batch['path'])
        acc['image_vectors'].append(batch['image_vectors'])

    df = pd.DataFrame({'label':sum(acc['label'], start=[]), 'path':sum(acc['path'], start=[]), 'vectors':sum(acc['image_vectors'], start=[])})
    return df