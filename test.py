import time
from contextlib import contextmanager

import cv2
from PIL import Image
import skimage.io
import torchvision
import torchvision.transforms.functional as TF
from torchvision.io import read_image


@contextmanager
def timer(name):
    t0 = time.time()
    yield
    print(f"[{name}] done in {(time.time() - t0)*1000:.03f} ms")


N_ITERS = 100
for FILENAME in ["./image.jpg","./image.png"]:
    print(FILENAME)
    with timer("PIL"):
        for i in range(N_ITERS):
            img = Image.open(FILENAME).convert("RGB")
            img = TF.to_tensor(img)

    with timer("OpenCV"):
        for i in range(N_ITERS):
            img = cv2.imread(FILENAME)[:, :, ::-1].copy()
            img = TF.to_tensor(img)

    with timer("default_torchvision"):
        for i in range(N_ITERS):
            img = read_image(FILENAME)

    with timer("accimage_torchvision"):
        torchvision.set_image_backend("accimage")
        for i in range(N_ITERS):
            img = read_image(FILENAME)

    with timer("skimage"):
        for i in range(N_ITERS):
            img = skimage.io.imread(FILENAME)
            img = TF.to_tensor(img)

