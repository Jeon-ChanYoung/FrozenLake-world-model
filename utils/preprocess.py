from PIL import Image
import torchvision.transforms as T

transform = T.Compose([
    T.Resize((64, 64), interpolation=T.InterpolationMode.NEAREST),
    T.ToTensor(),  # [0,1] 범위 + (C,H,W) 형태
])

# 이미지 리사이즈 및 0~1로 정규화
def preprocess(frame):
    img = Image.fromarray(frame)
    return transform(img)