import argparse
import torch
import torch.nn as nn
import numpy as np
import os
import pickle
from data_loader import get_loader 
from build_vocab import Vocabulary
from model import EncoderCNN, DecoderRNN
from torch.nn.utils.rnn import pack_padded_sequence
from torchvision import transforms

def train(args):
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Ensure model directory exists
    os.makedirs(args.model_path, exist_ok=True)
    
    # Image preprocessing
    transform = transforms.Compose([ 
        transforms.RandomCrop(args.crop_size),
        transforms.RandomHorizontalFlip(), 
        transforms.ToTensor(), 
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    
    # Load vocabulary and data
    with open(args.vocab_path, 'rb') as f:
        vocab = pickle.load(f)
    data_loader = get_loader(args.image_dir, args.caption_path, vocab, transform, args.batch_size, shuffle=True, num_workers=args.num_workers) 

    # Initialize models, loss, and optimizer
    encoder = EncoderCNN(args.embed_size).to(device)
    decoder = DecoderRNN(args.embed_size, args.hidden_size, len(vocab), args.num_layers).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(list(decoder.parameters()) + list(encoder.linear.parameters()) + list(encoder.bn.parameters()), lr=args.learning_rate)
    
    # Training loop
    total_step = len(data_loader)
    for epoch in range(args.num_epochs):
        for i, (images, captions, lengths) in enumerate(data_loader):
            images, captions = images.to(device), captions.to(device)
            targets = pack_padded_sequence(captions, lengths, batch_first=True)[0]
            
            # Model forward pass
            features = encoder(images)
            outputs = decoder(features, captions, lengths)
            
            # Compute loss and optimize
            loss = criterion(outputs, targets)
            decoder.zero_grad()
            encoder.zero_grad()
            loss.backward()
            optimizer.step()

            # Logging and saving
            if i % args.log_step == 0:
                print(f'Epoch [{epoch}/{args.num_epochs}], Step [{i}/{total_step}], Loss: {loss.item():.4f}, Perplexity: {np.exp(loss.item()):5.4f}')
            if (i+1) % args.save_step == 0:
                torch.save(decoder.state_dict(), os.path.join(args.model_path, f'decoder-{epoch+1}-{i+1}.ckpt'))
                torch.save(encoder.state_dict(), os.path.join(args.model_path, f'encoder-{epoch+1}-{i+1}.ckpt'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train an image captioning model on the COCO dataset.")
    # ... [rest of the argparse code remains unchanged]
    args = parser.parse_args()
    print(args)
    train(args)
