import numpy as np
import matplotlib.pyplot as plt
import os 

def main(plot_path):
  pass
 
if __name__ == "__main__":
  cur_path = os.path.dirname(os.path.abspath(__file__))
  plot_path = os.path.dirname(cur_path) + "/plots/"
  main(plot_path)
