"""
"""
import DuckDuckGoImages as ddg

from pathlib import Path
from argparse import ArgumentParser



path_images = Path.cwd().parent /"input" 
def download_images(title, urls=10):
    """function to download images from DuckDuckGo api"""

    ddg.download(title, folder= path_images, max_urls = urls)


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("-i", "--image", type=str, nargs="+",default="cats",\
        help="image to be search on DuckDuckGo" )

    parser.add_argument("-u", "--urls", default=10,\
        help="No. of images to be searched")
    
    args = parser.parse_args()

    download_images(title= " ".join(args.image), urls=int(args.urls))