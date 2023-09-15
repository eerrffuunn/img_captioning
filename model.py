import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence


class ImageEncoder(nn.Module):
    def __init__(self, embedding_dim):
        """Initialize the layers of the encoder."""
        super(ImageEncoder, self).__init__()
        resnet = models.resnet152(pretrained=True)
        layers = list(resnet.children())[:-1]  # Remove the last fully connected layer
        self.resnet = nn.Sequential(*layers)
        self.fc = nn.Linear(resnet.fc.in_features, embedding_dim)
        self.bn = nn.BatchNorm1d(embedding_dim, momentum=0.01)

    def forward(self, x):
        """Forward pass."""
        with torch.no_grad():
            x = self.resnet(x)
        x = x.view(x.size(0), -1)
        x = self.bn(self.fc(x))
        return x


class CaptionDecoder(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, vocab_size, num_layers, max_length=20):
        """Initialize the layers of the decoder."""
        super(CaptionDecoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        self.max_length = max_length

    def forward(self, features, captions, lengths):
        """Decode the feature vectors into captions."""
        embeddings = self.embedding(captions)
        embeddings = torch.cat((features.unsqueeze(1), embeddings), dim=1)
        packed = pack_padded_sequence(embeddings, lengths, batch_first=True)
        lstm_out, _ = self.lstm(packed)
        outputs = self.fc(lstm_out[0])
        return outputs

    def sample(self, features, states=None):
        """Generate captions from features."""
        sampled_ids = []
        inputs = features.unsqueeze(1)
        for _ in range(self.max_length):
            lstm_out, states = self.lstm(inputs, states)
            outputs = self.fc(lstm_out.squeeze(1))
            _, predicted = outputs.max(dim=1)
            sampled_ids.append(predicted)
            inputs = self.embedding(predicted).unsqueeze(1)
        return torch.stack(sampled_ids, dim=1)
