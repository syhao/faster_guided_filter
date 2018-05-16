#! /usr/bin/python
import numpy as np
import fast_guided_filter as gf
import skimage.io as skio
I=skio.imread("/home/sunya/Desktop/toy.jpg")
P=skio.imread("/home/sunya/Desktop/toy-mask.jpg")


# I=skio.imread("/home/sunya/einrfoqeso9r7iu.jpg")
# P=skio.imread("/home/sunya/einrfoqeso9r7iu_ret.jpg")
# I=skio.imread("/home/sunya/Documents/test_koutu_512/5a369c6cN7acc34da.jpg")
# P=skio.imread("/home/sunya/Documents/test_koutu_512/5a369c6cN7acc34da_ret.jpg")
# I=skio.imread("/home/sunya/Documents/test_koutu_512/5a39ccebN58d3d5ff.jpg")
# P=skio.imread("/home/sunya/Documents/test_koutu_512/5a39ccebN58d3d5ff_ret.jpg")
# I=skio.imread("/home/sunya/Documents/test_koutu_512/xepzw8ej65ttucf.jpg")
# P=skio.imread("/home/sunya/Documents/test_koutu_512/xepzw8ej65ttucf_ret.jpg")

#ret=gf.fastGuidedFilter(I,P,1,1e-2*255*255,1,-1)
ret=gf.fastGuidedFilter(I,P,60,1e-6*255*255,2,-1)
print(type(ret))
ret=ret.astype(np.uint8)
skio.imsave("ret.jpg",ret)
