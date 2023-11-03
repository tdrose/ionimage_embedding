import numpy as np

def flip_images(images: np.ndarray) -> np.ndarray:

    imgs = images.copy()

    for i in range(imgs.shape[0]):
        flips = np.random.randint(0, 4, 1)[0]
        for f in range(flips):
            imgs[i] = np.flip(imgs[i].transpose(), axis=1)

    return imgs
