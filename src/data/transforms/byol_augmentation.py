import random
import torchvision.transforms as T
from PIL import Image, ImageFilter, ImageOps

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

class GaussianBlur:
    def __init__(self, p: float = 0.5, radius_range=(0.1,2.0)):
        self.p = p
        self.radius_range = radius_range

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() > self.p:
            return img
        radius = random.uniform(*self.radius_range)
        return img.filter(ImageFilter.GaussianBlur(radius=radius))

class Solarize:
    def __init__(self, p: float = 0.0, threshold: int = 128):
        self.p = p
        self.threshold = threshold

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() > self.p:
            return img
        return ImageOps.solarize(img, threshold=self.threshold)

def _make_view_transform(image_size: int, blur_p: float, solarize_p: float) -> T.Compose:
    return T.Compose(
        [
            T.RandomResizedCrop(image_size, scale=(0.2, 1.0), interpolation=T.InterpolationMode.BICUBIC),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomApply([T.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8),
            T.RandomGrayscale(p=0.2),
            GaussianBlur(p=blur_p),
            Solarize(p=solarize_p),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )

class TwoViewTransform:
    def __init__(self, image_size: int = 224):
        self.transform1 = _make_view_transform(image_size, blur_p=1.0, solarize_p=0.0)
        self.transform2 = _make_view_transform(image_size, blur_p=0.1, solarize_p=0.2)

    def __call__(self, img: Image.Image):
        return self.transform1(img), self.transform2(img)