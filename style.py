#%%
style_weights_dict = {

    'High': {'conv1_1': 1,
             'conv2_1': 1,
             'conv3_1': 0.1,
             'conv4_1': 0.05,
             'conv5_1': 0.05},
    
    'Normal': {'conv1_1': 1.,
               'conv2_1': 0.75,
               'conv3_1': 0.2,
               'conv4_1': 0.2,
               'conv5_1': 0.2},

    'Low': {'conv1_1': 0.05,
            'conv2_1': 0.5,
            'conv3_1': 0.1,
            'conv4_1': 1,
            'conv5_1': 1},
    
        'Classic': {'conv1_1': 0.2,
            'conv2_1': 0.2,
            'conv3_1': 0.2,
            'conv4_1': 0.2,
            'conv5_1': 0.2}
}



import glob
from itertools import product
from pathlib import Path

import imageio
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
from torchvision import models, transforms
from tqdm import tqdm

from utils import get_features, gram_matrix, im_convert, load_image

#%%
# get the "features" portion of VGG19 (we will not need the "classifier" portion)
vgg = models.vgg19(pretrained=True).features

# freeze all VGG parameters since we're only optimizing the target image
for param in vgg.parameters():
    param.requires_grad_(False)


#%%
# move the model to GPU, if available
device = torch.device("cpu" if torch.cuda.is_available() else "cpu")
vgg.to(device)
print('Using', "GPU" if torch.cuda.is_available() else "CPU" )


#%%

CONTENT = Path('content')
STYLE = Path('style')

content_image = 'magic.jpg'
style_image = 'fire.jpg'

content_fn = content_image.split('.')[0]
style_fn = style_image.split('.')[0]

#%%
# load in content and style image
content = load_image(CONTENT/content_image).to(device)
# Resize style to match content, makes code easier
style = load_image(STYLE/style_image, shape=content.shape[-2:]).to(device)
print('-'*10, 'loading image','-'*10)
#%%
print('Print some image, for checking only, delete when done')
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
# content and style ims side-by-side
ax1.imshow(im_convert(content))
ax2.imshow(im_convert(style))

#%%
print('Model Strucure')
print(vgg)


# get content and style features only once before training
content_features = get_features(content, vgg)
style_features = get_features(style, vgg)

# calculate the gram matrices for each layer of our style representation
style_grams = {layer: gram_matrix(style_features[layer]) for layer in style_features}

# Start with copy of content image
target = content.clone().requires_grad_(True).to(device)
