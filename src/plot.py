from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from glob import glob

file_path = Path("../input")
jpg_path = f"{file_path}/*.jpg"
png_path = f"{file_path}/*.png"
files = glob(jpg_path) + glob(png_path)

random_file = np.random.randint(0, len(files))


def generate_images():
    """"""

    input_file = files[random_file]
    pred_file = files[random_file].replace("input", "output").replace("jpg","png")
    pred_abs_file = files[random_file].replace("input", "output").replace(".jpg","_aligned.png")

    new_files = [input_file, pred_file,pred_abs_file]
    fig = plt.figure(figsize=(128, 128))
    grid = fig.add_gridspec(1, 3)

    axes = [fig.add_subplot(grid[0,i]) for i in range(3)]
    
    for i,zipped in enumerate(zip(axes, ["original","UnAligned", "Aligned"])):
        zipped[0].axis("off")
        zipped[0].imshow(np.array(Image.open(new_files[i])))
        zipped[0].set_title(f"{zipped[1]}")

    plt.show()

generate_images()