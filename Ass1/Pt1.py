# Import assignment functions
import os

def main(image_path):
  pass

if __name__ == "__main__":
  cur_path = os.path.dirname(os.path.abspath(__file__))
  image_path = os.path.dirname(cur_path) + "/plots/"
  main(image_path)
