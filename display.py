import numpy as np
import matplotlib.pyplot as plt
#%matplotlib inline
import caffe
caffe_root='/home/tomato/workspace/caffe/'
import os,sys
os.chdir(caffe_root)
sys.path.insert(0,caffe_root+'python')
im = caffe.io.load_image('examples/images/cat.jpg')
print im.shape
plt.imshow(im)
#plt.axis('off')

net = caffe.Net('examples/net_surgery/conv.prototxt', caffe.TEST)
im_input=im[np.newaxis,:,:,:].transpose(0,3,1,2)
print "data-blobs:",im_input.shape
net.blobs['data'].reshape(*im_input.shape)
net.blobs['data'].data[...] = im_input
plt.imshow(net.blobs['data'].data[0].transpose(1,2,0))
plt.axis('off')

#show data
plt.rcParams['image.cmap'] = 'gray'
def show_data(data,head,padsize=1, padval=0):
    data -= data.min()
    data /= data.max()
    # force the number of filters to be square
    n = int(np.ceil(np.sqrt(data.shape[0])))
    padding = ((0, n ** 2 - data.shape[0]), (0, padsize), (0, padsize)) + ((0, 0),) * (data.ndim - 3)
    print data.shape
    data = np.pad(data, padding, mode='constant', constant_values=(padval, padval))
    print data.shape
    # tile the filters into an image
    data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
    print data.shape
    data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])
    print data.shape
    plt.figure()
    plt.title(head)
    plt.imsave("display.jpg", data)
    plt.axis('off')
print "data-blobs:",net.blobs['data'].data.shape
show_data(net.blobs['data'].data[0],'origin images')

#net.forward()
#print "data-blobs:",net.blobs['data'].data.shape
#print "conv-blobs:",net.blobs['conv'].data.shape
#print "weight-blobs:",net.params['conv'][0].data.shape
#show_data(net.params['conv'][0].data[:,0],'conv weights(filter)')
#show_data(net.blobs['conv'].data[0],'post-conv images')

