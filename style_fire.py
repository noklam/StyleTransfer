import fire
from config import *
from style import style_transfer

class StyleTransfer(object):
    """A simple calculator class."""
    def __init__(self, files_names, use_gpu=True, lr=0.2):
        self.style_weights_dict = style_weights_dict
        self.files_list = files_names.split(" ")
        self.content, self.styles = self.files_list[0], self.files_list[1:]
        print(self.content)
        print(self.styles)
        style_transfer(self.content, self.styles)


if __name__ == '__main__':
    fire.Fire(StyleTransfer)
