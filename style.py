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

style_levels =  ['Classic','Normal','High','Low']
content_style_ratios = [0.000001, 0.0001, 0.01]

list(product(style_levels,content_style_ratios))


for style_level, content_style_ratio in tqdm(list(product(style_levels,content_style_ratios))):
    # Start from a fresh photo
    target = content.clone().requires_grad_(True).to(device)
    # for displaying the target image, intermittently
    show_every = 1
    style_weights = style_weights_dict[style_level]
    # iteration hyperparameters
    optimizer = optim.LBFGS([target], lr=0.2)
    steps = 20  # decide how many iterations to update your image (5000)
    run = [0]

    OUTPUT = Path('output')
    OUTPUT = OUTPUT/f'{content_fn}_{style_fn}'
    OUTPUT.mkdir(exist_ok=True)
    OUTPUT_SUBPATH = OUTPUT/f"{style_level}_{content_style_ratio}"
    OUTPUT_SUBPATH.mkdir(exist_ok=True)

    for ii in tqdm(range(1, steps+1)):

        def closure():
            # get the features from your target image
            target_features = get_features(target, vgg)

            # the content loss
            content_loss = torch.mean((target_features['conv4_2'] - content_features['conv4_2'])**2)

            # initialize the style loss to 0
            style_loss = 0
            # then add to it for each layer's gram matrix loss
            for layer in style_weights:
                # get the "target" style representation for the layer
                target_feature = target_features[layer]
                target_gram = gram_matrix(target_feature)
                _, d, h, w = target_feature.shape
                # get the "style" style representation
                style_gram = style_grams[layer]
                # the style loss for one layer, weighted appropriately
                layer_style_loss = style_weights[layer] * torch.mean((target_gram - style_gram)**2)
                # add to the style loss
                style_loss += layer_style_loss / (d * h * w)

            # calculate the *total* loss
            total_loss =  (content_loss * content_style_ratio +  style_loss) * 1e6

            # update your target image
            optimizer.zero_grad()
            total_loss.backward()

            run[0] += 1
#             if run[0] % 20 == 0:
#                 print("run {}:".format(run))
#                 print('Total Loss: {:4f}'.format(total_loss))
#                 print()
#                 print(f'Step {ii} -------')
            return total_loss
        
        # Save the first image
        plt.imsave(OUTPUT_SUBPATH/f"{ii}.jpg",im_convert(target))
        optimizer.step(closure)
              
    # Finish 1 set of parameter                  
    images = []
    for filename in OUTPUT_SUBPATH.glob('*.jpg'):
         images.append(imageio.imread(filename))
    
    imageio.mimsave(OUTPUT/f"{content_fn}_{style_fn}_{style_level}_{content_style_ratio}.gif", images, fps=2) # Output gif
    plt.imsave(OUTPUT/f"{content_fn}_{style_fn}_{style_level}_{content_style_ratio}.jpg",im_convert(target)) # Output the last image as well
#%%