import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from torch.autograd import Variable

# Load pre-trained ResNet model
resnet = models.resnet18(pretrained=True)
modules = list(resnet.children())[:-1]  # Remove the last fully connected layer
resnet = nn.Sequential(*modules)
for param in resnet.parameters():
    param.requires_grad = False
resnet.eval()

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])

# Simple LSTM-based captioning model
class CaptioningModel(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(CaptioningModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)

    def forward(self, features, captions):
        captions = captions[:, :-1]
        embeds = self.embedding(captions)
        inputs = torch.cat((features.unsqueeze(1), embeds), 1)
        lstm_out, _ = self.lstm(inputs)
        outputs = self.linear(lstm_out)
        return outputs

# Function to generate captions for images
def generate_caption(image_path, model, vocab, max_len=20):
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0)
    image = Variable(image)

    # Extract features using ResNet
    resnet.eval()
    with torch.no_grad():
        features = resnet(image)
    resnet.train()

    # Initialize captioning model
    embed_size = 256
    hidden_size = 512
    vocab_size = len(vocab)
    num_layers = 1
    captioning_model = CaptioningModel(embed_size, hidden_size, vocab_size, num_layers)

    # Load pre-trained captioning model weights
    captioning_model.load_state_dict(torch.load('captioning_model_weights.pth'))
    captioning_model.eval()

    # Generate caption
    caption = []
    word = torch.tensor(vocab('<start>')).view(1, -1)
    h, c = None, None

    for _ in range(max_len):
        with torch.no_grad():
            out = captioning_model(features, word)
            _, predicted = out.max(2)
            word = predicted[:, -1].view(1, -1)

            if word.item() == vocab('<end>'):
                break

            caption.append(word.item())

    caption_words = [vocab.idx2word[idx] for idx in caption]
    caption_sentence = ' '.join(caption_words)

    return caption_sentence

# Example usage
if __name__ == "__main__":
    # You need to have a vocabulary mapping for your dataset
    # Example vocab: {'<start>': 0, '<end>': 1, 'a': 2, 'cat': 3, ...}
    vocab = {'<start>': 0, '<end>': 1, 'a': 2, 'dog': 3, 'is': 4, 'running': 5, 'in': 6, 'the': 7, 'park': 8, '.': 9}

    image_path = 'path/to/your/image.jpg'
    caption = generate_caption(image_path, captioning_model, vocab)
    print(f"Generated Caption: {caption}")
