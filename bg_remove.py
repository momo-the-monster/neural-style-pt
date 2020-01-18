# install DeepLab Masking
import numpy as np
import torchvision
import torch
from PIL import Image, ImageFilter
from torchvision import transforms

deeplabv3_resnet101_model = torchvision.models.segmentation.deeplabv3_resnet101(pretrained=True)
deeplabv3_resnet101_model = deeplabv3_resnet101_model.eval()
bg_preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def replace_bg(image_with_person, bg, output_image, size, blur_strength):
    # run model on image
    image_with_person.thumbnail((size,size), Image.ADAPTIVE)
    image_tensor = bg_preprocess(image_with_person)
    input_batch = image_tensor.unsqueeze(0)
    input_batch = input_batch.to('cuda')
    deeplabv3_resnet101_model.to('cuda')
    with torch.no_grad():
        output = deeplabv3_resnet101_model(input_batch)['out'][0]
    output_predictions = output.argmax(0)

    # create empty palette lookup
    palette = torch.zeros([21,3], dtype=torch.uint8)
    # set label 15 (person) to white
    palette[15][0] = 255
    palette[15][1] = 255
    palette[15][2] = 255
    # convert palette to numpy array
    palette = palette.numpy()

    # plot the semantic segmentation predictions of 21 classes in each color
    mask = Image.fromarray(output_predictions.byte().cpu().numpy()).resize(image_with_person.size)
    mask.putpalette(palette)
    mask = mask.convert('L').filter(ImageFilter.GaussianBlur(radius=blur_strength))

    # load new bg
    bg_image = bg.resize(image_with_person.size)

     # load image to mask
    output_image = output_image.resize(image_with_person.size)

    # composite and return
    composited = Image.composite(output_image, bg_image, mask)
    return composited
