from PIL import Image
import torchvision.transforms as T

transform_64 = T.Compose([
    T.Resize((64, 64), interpolation=T.InterpolationMode.NEAREST),
    T.ToTensor(),  # [0,1] 범위 + (C,H,W) 형태
])

transform_128 = T.Compose([
    T.Resize((128, 128), interpolation=T.InterpolationMode.NEAREST),
    T.ToTensor(),  # [0,1] 범위 + (C,H,W) 형태
])


# 이미지 리사이즈 및 0~1로 정규화
def preprocess_64(frame):
    img = Image.fromarray(frame)
    return transform_64(img)

def preprocess_128(frame):
    img = Image.fromarray(frame)
    return transform_128(img)


def to_PIL(tensor_frame):
    to_pil = T.ToPILImage()
    return to_pil(tensor_frame)
