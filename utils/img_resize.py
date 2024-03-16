import os
import numpy as np
import cv2
from tqdm import tqdm


def resize_dir(path, output, dim):
    dir = np.array(os.listdir(path))
    for idx, i in enumerate(tqdm(dir, leave=True)):
        if i.split(".")[-1] == "jpg":
            img = cv2.imread(f"{path}{i}")
            img_res = cv2.resize(img, dim)
            cv2.imwrite(f"{output}{i}", img_res)
            print(f"{idx+1}) Wrote img {i}")
    print(f"Write complete: {idx+1} Images...")


if __name__ == "__main__":
    resize_dir("./ns_opt_res/images_full/", "./ns_opt_res/images/", (960, 540))
