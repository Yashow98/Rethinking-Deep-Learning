# @Author  : Yashowhoo
# @File    : 20_pillow.py
# @Description :

from PIL import Image

im = Image.open('./src/glass1.jpg')  # return an image obj
print(im.mode, im.format, im.size)  # The size attribute is a 2-tuple containing width and height (in pixels).
print(im)

im.show()

# Simple geometry transforms
# out_resize = im.resize((224, 224))
# out_rotate = im.rotate(45)

# out_resize.show()
# out_rotate.show()
#
# out = im.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
# out.show()
#
# out = im.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
# out.show()
#
# out = im.transpose(Image.Transpose.ROTATE_90)
# out.show()
#
# out = im.transpose(Image.Transpose.ROTATE_180)
# out.show()
#
# out = im.transpose(Image.Transpose.ROTATE_270)
# out.show()

# color transform
# im_gray = im.convert('L')
# im_gray.show()

# reading from url
# from urllib.request import urlopen
# url = "https://python-pillow.org/images/pillow-logo.png"
# img = Image.open(urlopen(url))
# img.show()

# from a tar archive
# from PIL import Image, TarIO
#
# fp = TarIO.TarIO("Tests/images/hopper.tar", "hopper.jpg")
# im = Image.open(fp)
