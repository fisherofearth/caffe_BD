import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.pylab as pylab
caffe_root = './' 
import sys
sys.path.insert(0, caffe_root + 'python')
import caffe
import os
from PIL import Image
from PIL import ImageFilter

plt.rcParams['figure.figsize'] = (5, 5)
plt.rcParams['image.interpolation'] = 'nearest' #'gaussian' #'lanczos'
plt.rcParams['image.cmap'] = 'gray'

#Visualization
def plot_single_scale(scale_lst, size, final):
    pylab.rcParams['figure.figsize'] = size*2, size
    plt.figure(0)
    if len(scale_lst) >= 6:
    	n_c = 6 
    else:
    	n_c = len(scale_lst)
    n_r = ((len(scale_lst)-1)//6) + 1
    for i in range(0, len(scale_lst)):
        s=plt.subplot(n_r, n_c,i+1)
        plt.imshow(scale_lst[i], cmap = cm.Greys_r)
        s.set_xticklabels([])
        s.set_yticklabels([])
        s.yaxis.set_ticks_position('none')
        s.xaxis.set_ticks_position('none')
    plt.tight_layout() 
    plt.figure(1)
    s=plt.subplot(1, 1,1)
    plt.imshow(final, cmap = cm.Greys_r)
    s.set_xticklabels([])
    s.set_yticklabels([])
    s.yaxis.set_ticks_position('none')
    s.xaxis.set_ticks_position('none')
    plt.tight_layout()
    
    plt.show()


caffe.set_mode_cpu()
net = caffe.Net(caffe_root + 'examples/BSDS500/BSDS500_test.prototxt',
                caffe_root + 'BSDS500_BD_iter_50000_best.caffemodel',
                caffe.TEST)

# input preprocessing: 'data' is the name of the input blob == net.inputs[0]
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_transpose('data', (2,0,1))
transformer.set_mean('data', 
	np.load(caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy').mean(1).mean(1)) # mean pixel
transformer.set_raw_scale('data', 255)  # the reference model operates on images in [0,255] range instead of [0,1]
transformer.set_channel_swap('data', (2,1,0))  # the reference model has channels in BGR order instead of RGB

net.blobs['data'].reshape(1,3,160,160)


#image_root = 'data/BSDS500/BSDS500_BD/images/train/'
#image_root = 'data/BSDS500/BSDS500_BD_160x160_18000/images/train/'

#image_root = 'examples/BSDS500/160/'
#filenames = [str(file) for file in os.listdir(caffe_root+image_root)]
#idx = input('image=')
#img = Image.open(caffe_root + image_root + filenames[idx])
#img.filter(ImageFilter.DETAIL).save(caffe_root + 'image_tmp/' + filenames[idx], 'JPEG')
#net.blobs['data'].data[...] = transformer.preprocess(
#	'data', caffe.io.load_image(caffe_root + 'image_tmp/' + filenames[idx]))# 

filename = str(input('image='))
img = Image.open(caffe_root + 'data/BSDS500/BSDS500_BD/images/val/' + filename+'.jpg' )
img.filter(ImageFilter.DETAIL).save(caffe_root + 'image_tmp/' + filename+'.jpg')
net.blobs['data'].data[...] = transformer.preprocess(
	'data', caffe.io.load_image(caffe_root + 'image_tmp/' + filename+'.jpg'))


net.params["deconv1"][0].data[...] = net.params["conv1"][0].data.copy()
net.params["deconv2"][0].data[...] = net.params["conv2"][0].data.copy()
net.params["deconv3"][0].data[...] = net.params["conv3"][0].data.copy()


out = net.forward()

def threshold(image_array, T):
	for x in xrange(image_array.shape[0]):
		for y in xrange(image_array.shape[1]):
			if image_array[x,y] < T:
				pass#image_array[x,y] = 0
			elif image_array[x,y] > T * 1:
				pass#image_array[x,y] = image_array[x,y] * 10
	return image_array

T = 0.0002

final = (threshold(net.blobs['pool_bd'].data[0][0,:,:], T)+\
	threshold(net.blobs['pool_bd'].data[0][1,:,:], T)+ \
	threshold(net.blobs['pool_bd'].data[0][2,:,:], T))

scale_lst = [
	net.blobs['data'].data[0][0,:,:],

	net.blobs['conv1'].data[0][0,:,:],
	net.blobs['pool1'].data[0][0,:,:],
	net.blobs['conv2'].data[0][0,:,:],
	net.blobs['pool2'].data[0][0,:,:],
	net.blobs['conv3'].data[0][0,:,:],

	net.blobs['deconv3'].data[0][0,:,:],
	net.blobs['unpool2'].data[0][0,:,:],
	net.blobs['deconv2'].data[0][0,:,:],
	net.blobs['unpool1'].data[0][0,:,:],
	#net.blobs['deconv1'].data[0][0,:,:],

	#threshold(net.blobs['pool_bd'].data[0][0,:,:], T),
	#threshold(net.blobs['pool_bd'].data[0][1,:,:], T),
	#threshold(net.blobs['pool_bd'].data[0][2,:,:], T),
	final,
	
	]


plot_single_scale(scale_lst, 10, final)

