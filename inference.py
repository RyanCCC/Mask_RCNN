from mrcnn.mask_rcnn import MASK_RCNN
from PIL import Image

img = './images/2.jpg'
mask_rcnn = MASK_RCNN()
img = Image.open(img)
image = mask_rcnn.detect_image(image = img)
image.show()