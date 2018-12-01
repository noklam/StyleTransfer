#%%
style_weights_dict = {

    'High': {'conv1_1': 0.2,
             'conv2_1': 0.2,
             'conv3_1': 0.2,
             'conv4_1': 0.75,
             'conv5_1': 1},
    
    'Normal': {'conv1_1': 1.,
               'conv2_1': 0.75,
               'conv3_1': 0.2,
               'conv4_1': 0.2,
               'conv5_1': 0.2},

    'Low': {'conv1_1': 0.5,
            'conv2_1': 0.3,
            'conv3_1': 0.2,
            'conv4_1': 0.2,
            'conv5_1': 0.2},
    
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
style_images = ['fire.jpg']

content_fn = content_image.split('.')[0]
style_fns = [i.split('.')[0] for i in style_images]

#%%
# load in content and style image
content = load_image(CONTENT/content_image).to(device)
# Resize style to match content, makes code easier
styles = [load_image(STYLE/i, shape=content.shape[-2:]).to(device) for i in style_images]
print('-'*10, 'loading image','-'*10)
#%%
# display the images
fig, ax1,  = plt.subplots(1, 1, figsize=(5, 5))
# content and style ims side-by-side

if len(styles)> 1:
    fig, axs = plt.subplots(1, len(styles), figsize= (5, 5*len(styles)))
    for ax, style in zip(axs, styles):
        ax.imshow(im_convert(style))
else:
    fig, ax = plt.subplots(1,1, figsize=(5, 5))
    ax.imshow(im_convert(styles[0]))
#%%
print('Model Strucure')
print(vgg)

# get content and style features only once before training
content_features = get_features(content, vgg)
style_features_ls = [get_features(i, vgg) for i in styles]

# calculate the gram matrices for each layer of our style representation
style_grams_ls = [{layer: gram_matrix(style_features[layer])  for style_features in style_features_ls for layer in style_features}]

# Start with copy of content image
target = content.clone().requires_grad_(True).to(device)

style_levels =  ['Classic','Normal','High','Low']
style_content_ratios = [100,10000,1000000]

list(product(style_levels, style_content_ratios))
for style_level, style_content_ratio in tqdm(list(product(style_levels,style_content_ratios))):
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
    output_dir = f"{content_fn}_{'_'.join([fn for fn in style_fns])}"
    OUTPUT = OUTPUT/output_dir
    OUTPUT.mkdir(exist_ok=True)
    OUTPUT_SUBPATH = OUTPUT/f"{style_level}_{style_content_ratio}"
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
                for style_grams in style_grams_ls:
                    # get the "style" style representation
                    style_gram = style_grams[layer]
                    # the style loss for one layer, weighted appropriately
                    layer_style_loss = style_weights[layer] * torch.mean((target_gram - style_gram)**2)
                    # add to the style loss
                    style_loss += layer_style_loss / (d * h * w)

            # calculate the *total* loss
            style_loss =  style_loss * style_content_ratio            
            total_loss =  content_loss  + style_loss

            # update your target image
            optimizer.zero_grad()
            total_loss.backward()

            run[0] += 1
            if run[0] % 20 == 0:
                print("run {}:".format(run))
                print('Style Loss: {:4f}'.format(style_loss), 'Content Loss: {:4f}'.format(content_loss))
                print('Total Loss: {:4f}'.format(total_loss))
                print()
                print(f'Step {ii} -------')
            return total_loss
        
        # Save the first image
        plt.imsave(OUTPUT_SUBPATH/f"{ii}.jpg",im_convert(target))
        optimizer.step(closure)
              
    # Finish 1 set of parameter                  
    images = []
    for filename in OUTPUT_SUBPATH.glob('*.jpg'):
         images.append(imageio.imread(filename))
    
    imageio.mimsave(OUTPUT/f"{output_dir}_{style_level}_{style_content_ratio}.gif", images, fps=4) # Output gif
    plt.imsave(OUTPUT/f"{output_dir}_{style_level}_{style_level}_{style_content_ratio}.jpg",im_convert(target)) # Output the last image as well


#%%
h,w = len(style_content_ratios), len(style_levels)
fig, axs = plt.subplots(h, w, figsize = (16,12), sharex=True, sharey=True)
plt.tight_layout()

imgs = []
styles = []
ratios = []
for file in OUTPUT.glob('*.jpg'):
    img = plt.imread(file)
    imgs.append(img)
    styles.append(file.parts[-1].split('_')[-2])
    ratios.append(file.parts[-1].split('_')[-1].split('.jpg')[0])

for i,ax in enumerate(axs):  
    for j, axe  in enumerate(ax):
        if i==0:
            axe.set_title(ratios[j*len(style_content_ratios) + i])
        if j==0:
            axe.set_ylabel(styles[j*len(style_content_ratios) + i])
        axe.imshow(imgs[j*len(style_content_ratios) + i])

fig.savefig(f"{output_dir}.jpg")
