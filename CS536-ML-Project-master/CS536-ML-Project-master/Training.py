import torch
import torchvision as tv
import mvtecDataset as md
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

from config import BATCH_SIZE, IMAGE_SIZE, INTERPOL, EPOCHS
from torch.utils.data import DataLoader, sampler, random_split

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


# get a random training image
dataiter = iter(train_loader)
images, labels = next(dataiter)

# show images
imshow(images[0])
# print label
print(classes[labels[0]])


