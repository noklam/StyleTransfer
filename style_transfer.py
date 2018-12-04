import fire
from config import *
from style import style_transfer

class StyleTransfer(object):
    """A simple StyleTransfer class take images and return stylish images."""
    def __init__(self, files, use_gpu=True, lr=0.2):
        self.style_weights_dict = style_weights_dict
        self.files_list = files.split(" ")
        self.content, self.styles = self.files_list[0], self.files_list[1:]
        print('Content Image:', self.content)
        print('Style Image(s):',self.styles)
        style_transfer(self.content, self.styles, use_gpu=use_gpu, lr=lr)

if __name__ == '__main__':
    fire.Fire(StyleTransfer)
