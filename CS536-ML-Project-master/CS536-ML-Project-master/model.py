import torch
import torch.optim as optim
import mvtecDataset as md
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from torch.nn.functional import l1_loss

from VisionTransformer import VisionTransformer
from config import IMAGE_SIZE, PATCH_SIZE, INPUT_CHANNELS, MVTEC_CLASSES, EMBEDDING_SIZE, NO_OF_HEADS, MLP_RATIO, \
    QKV_BIAS, DROPOUT_PROB, ATTENTION_DROPOUT, NO_OF_BLOCKS_TEXTURES, LEARNING_RATE, EPOCHS, BATCH_SIZE, INTERPOL, \
    SAVE_PATH
from ssimSimilarityLoss import SSIMLoss

from torch.utils.data import DataLoader

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
)

batch_size = BATCH_SIZE
image_shape = IMAGE_SIZE
interpol = INTERPOL
catg = 'carpet'

training_set = md.MVTEC(root='mvtec', train=True, transform=transform, resize=image_shape,
                        interpolation=interpol, category=catg)

train_loader = DataLoader(training_set, batch_size=batch_size, shuffle=True)

testing_set = md.MVTEC(root='mvtec', train=False, transform=transform, resize=image_shape,
                       interpolation=interpol, category=catg)

test_loader = DataLoader(training_set, batch_size=batch_size, shuffle=False)

classes = ('defective', 'good')

# function to show an image
def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

mvtec_model = VisionTransformer(IMAGE_SIZE,
                                PATCH_SIZE,
                                INPUT_CHANNELS,
                                MVTEC_CLASSES,
                                EMBEDDING_SIZE,
                                NO_OF_BLOCKS_TEXTURES,
                                NO_OF_HEADS,
                                MLP_RATIO,
                                QKV_BIAS,
                                DROPOUT_PROB,
                                ATTENTION_DROPOUT)

criterion_1 = SSIMLoss()
optimizer = optim.Adam(mvtec_model.parameters(), lr=LEARNING_RATE)

for epoch in range(EPOCHS):
    running_loss = 0.0
    for i, data in enumerate(train_loader):
        inputs, labels = data
        labels_data = torch.zeros((len(labels), 2))
        labels_data[np.arange(len(labels)), labels] = 1
        optimizer.zero_grad()

        outputs = mvtec_model(inputs)
        loss_1 = criterion_1(outputs, labels_data)
        loss_2 = l1_loss(outputs, labels_data)
        loss = loss_1 + loss_2
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        print('[%d, %5d] loss: %.3f' %
              (epoch + 1, i + 1, running_loss))
        running_loss = 0.0

print('Training End')
torch.save(mvtec_model.state_dict(), SAVE_PATH)



