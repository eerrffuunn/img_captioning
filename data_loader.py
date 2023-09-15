import os
import nltk
import torch
import torchvision.transforms as transforms
import torch.utils.data as data
from PIL import Image
from pycocotools.coco import COCO
from build_vocab import Vocabulary


class CocoDataset(data.Dataset):
    """COCO Custom Dataset compatible with torch.utils.data.DataLoader."""
    def __init__(self, image_dir, annotation_path, vocab, transform=None):
        self.image_dir = image_dir
        self.coco = COCO(annotation_path)
        self.ids = list(self.coco.anns.keys())
        self.vocab = vocab
        self.transform = transform

    def __getitem__(self, index):
        ann_id = self.ids[index]
        caption = self.coco.anns[ann_id]['caption']
        img_id = self.coco.anns[ann_id]['image_id']
        img_path = self.coco.loadImgs(img_id)[0]['file_name']

        image = Image.open(os.path.join(self.image_dir, img_path)).convert('RGB')
        if self.transform:
            image = self.transform(image)

        tokens = nltk.tokenize.word_tokenize(caption.lower())
        caption_tokens = [self.vocab('<start>')] + [self.vocab(token) for token in tokens] + [self.vocab('<end>')]
        return image, torch.Tensor(caption_tokens)

    def __len__(self):
        return len(self.ids)


def collate_fn(batch_data):
    batch_data.sort(key=lambda x: len(x[1]), reverse=True)
    images, captions = zip(*batch_data)

    images = torch.stack(images, 0)
    max_len = max(len(cap) for cap in captions)
    padded_captions = torch.zeros(len(captions), max_len).long()

    for i, cap in enumerate(captions):
        padded_captions[i, :len(cap)] = cap
    return images, padded_captions, [len(cap) for cap in captions]


def get_data_loader(image_dir, annotation_path, vocab, transform, batch_size, shuffle=True, num_workers=0):
    dataset = CocoDataset(image_dir, annotation_path, vocab, transform)
    return torch.utils.data.DataLoader(dataset=dataset, 
                                       batch_size=batch_size,
                                       shuffle=shuffle,
                                       num_workers=num_workers,
                                       collate_fn=collate_fn)
