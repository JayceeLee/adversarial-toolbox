import sys
import re, itertools
import PIL
sep = re.compile("[ \t\r\n]+")

def chunks(iterable,size):
    """ http://stackoverflow.com/a/434314/309483 """
    it = iter(iterable)
    chunk = tuple(itertools.islice(it,size))
    while chunk:
        yield chunk
        chunk = tuple(itertools.islice(it,size))

""" Emulates the Image class from PIL and some member functions (`getpixel`, `size`). """
class Image:
    """ This class emulates the PIL Image class, and it can parse "plain" PPM's.
        You can use PIL instead. """
    @staticmethod
    def fromstdin():
        return Image()
    def __init__(self): # http://netpbm.sourceforge.net/doc/ppm.html
        self.entities = sep.split("\n".join(list(filter(lambda x: not x.startswith("#"), sys.stdin.read().strip().split("\n")))))
        self.size = tuple(list(map(int,self.entities[1:3])))
        self.high = int(self.entities[3]) # z.b. 255
        self.pixels = list(map(lambda x: tuple(map(lambda y: int(int(y) / self.high * 255), x)), list(chunks(self.entities[4:], 3))))
    def getpixel(self, tupl):
        x = tupl[0]
        y = tupl[1]
        pix = self.pixels[y*self.size[0]+x]
        return pix

image_to_ansi = __import__("image-to-ansi") # __import__ because of minuses in filename. From https://gist.github.com/1687427

def show(x):
    im = x # use Image.open from PIL if using PIL
    im = PIL.Image.fromarray(np.uint8(cm.gist_earth(img)*255))
    for y in range(im.size[1]):
        for x in range(im.size[0]):
            p = im.getpixel((x, y))
            h = "%2x%2x%2x" % (p[0],p[1],p[2])
            short, rgb = image_to_ansi.rgb2short(h)
            sys.stdout.write("\033[48;5;%sm " % short)
        sys.stdout.write("\033[0m\n")
    sys.stdout.write("\n")
