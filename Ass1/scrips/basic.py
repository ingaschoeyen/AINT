import subprocess
from __future__ import annotations

import os
import hashlib
import io
import warnings
import pickle
import re
import requests
from types import ModuleType
from typing import Tuple, Union

from tqdm.notebook import trange, tqdm
import matplotlib.pyplot as plt
import matplotlib.image as img
from matplotlib import cm
import numpy as np
np.bool = np.bool_ # needs to be here for mx to import. don't question it 🤫
import mxnet as mx
from mxnet import autograd, Context, gluon, nd, symbol
from mxnet.gluon import loss, nn
from mxnet.gluon.nn import Conv2D, Dense, HybridBlock, HybridSequential, LeakyReLU
from mxnet.gluon.parameter import Parameter
from mxnet.initializer import Zero
from mxnet.ndarray import NDArray
from mxnet.io import NDArrayIter
import nibabel as nib
from nilearn import plotting
from PIL import Image
from scipy.stats import t, zscore
from sklearn.linear_model import RidgeCV

mx.npx.set_np()

def basic_setup():
  # Install nvidia-cuda-toolkit
  subprocess.run(['apt', 'install', '-qq', 'nvidia-cuda-toolkit', '--yes'])
  
  # Install required Python packages
  subprocess.run(['pip', 'install', '-q', 'mxnet-cu112', 'nilearn', 'nibabel', 'tqdm'])
  if mx.gpu:
    ctx = mx.gpu()
  else:
    ctx = mx.cpu()  
  print(ctx)

class Pixelnorm(HybridBlock):
    def __init__(self, epsilon: float = 1e-8) -> None:
        super(Pixelnorm, self).__init__()
        self.eps = epsilon

    def hybrid_forward(self, F, x) -> nd:
        return x * F.rsqrt(F.mean(F.square(x), 1, True) + self.eps)


class Bias(HybridBlock):
    def __init__(self, shape: Tuple) -> None:
        super(Bias, self).__init__()
        self.shape = shape
        with self.name_scope():
            self.b = self.params.get("b", init=Zero(), shape=shape)

    def hybrid_forward(self, F, x, b) -> nd:
        return F.broadcast_add(x, b[None, :, None, None])

# helper functions to download and load all the data files spread around the internet

def _url_to_name(url:str) -> str:
  """Calculates the md5 hash of an url string as a proxy filename"""
  return hashlib.md5(url.encode()).hexdigest()

def get_drive_file_id_from_share_link(share_url:str) -> str:
    """From a google drive public sharing link, get the file_id we can use to directly download the file"""
    return re.match(r".*/d/(?P<file_id>[a-zA-Z0-9-_]+)/?.*", share_url)['file_id']

def make_drive_downloadable_url(file_id:str) -> str:
    """from a google drive file-id, generate a direct-download link to be used in code"""
    return r"https://drive.usercontent.google.com/download?id={file_id}&export=download&confirm=y".format(file_id=file_id)

def load_from_url(url:str, name:str=None) -> bytes:
  """Loads data from a URL and returns the byte response. Implements simple caching"""
  if name is None:
    warnings.warn("Consider giving the downloaded file a unique name, to avoid accidental overwriting.", )
    name = _url_to_name(url)
  file_path = __save_location / name

  if file_path.is_file():
    with open(file_path, "rb") as f:
      res = f.read()
  else:
    with requests.get(url) as response:
      if response.status_code != 200:
        raise ConnectionError(f"Unsuccessful GET request for: {url}")
      with open(file_path, "wb") as f:
        res = response.content
        f.write(res)

  return res

def get_downloaded_file_path(name:str=None, url:str=None) -> str:
  """Some libraries deal better with file names than byte streams. Adds access to downloaded files"""
  if name is None:
    if url is None:
      raise ValueError("Name and URL cannot both be None")

    name = _url_to_name(url)

  file_path = __save_location / name
  if not file_path.is_file():
    raise FileNotFoundError(f"No file found at {file_path}\nWas it downloaded yet?")

  return str(file_path)
class Block(HybridSequential):
    def __init__(self, channels: int, in_channels: int) -> None:
        super(Block, self).__init__()
        self.channels = channels
        self.in_channels = in_channels
        with self.name_scope():
            self.add(Conv2D(channels, 3, padding=1, in_channels=in_channels))
            self.add(LeakyReLU(0.2))
            self.add(Pixelnorm())
            self.add(Conv2D(channels, 3, padding=1, in_channels=channels))
            self.add(LeakyReLU(0.2))
            self.add(Pixelnorm())

    def hybrid_forward(self, F, x) -> nd:
        x = F.repeat(x, 2, 2)
        x = F.repeat(x, 2, 3)
        for i in range(len(self)):
            x = self[i](x)
        return x


class Generator(HybridSequential):
    def __init__(self) -> None:
        super(Generator, self).__init__()
        with self.name_scope():
            self.add(Pixelnorm())
            self.add(Dense(8192, use_bias=False, in_units=512))
            self.add(Bias((512,)))
            self.add(LeakyReLU(0.2))
            self.add(Pixelnorm())
            self.add(Conv2D(512, 3, padding=1, in_channels=512))
            self.add(LeakyReLU(0.2))
            self.add(Pixelnorm())
            self.add(Block(512, 512))
            self.add(Block(512, 512))
            self.add(Block(512, 512))
            self.add(Block(256, 512))
            self.add(Block(128, 256))
            self.add(Block(64, 128))
            self.add(Block(32, 64))
            self.add(Block(16, 32))
            self.add(Conv2D(3, 1, in_channels=16))

    def hybrid_forward(self, F: Union(nd, symbol), x: nd, layer: int) -> nd:
        x = F.Reshape(self[1](self[0](x)), (-1, 512, 4, 4))
        for i in range(2, len(self)):
            x = self[i](x)
            if i == layer + 7:
                return x
        return x


class GAN():
  def __init__():
    
