# Import assignment functions
import os
from basic import GAN
import matplotlib.pyplot as plt
import numpy as np

def main(image_path):
  pass

if __name__ == "__main__":
  cur_path = os.path.dirname(os.path.abspath(__file__))
  image_path = os.path.dirname(cur_path) + "/plots/"
  main(image_path)
