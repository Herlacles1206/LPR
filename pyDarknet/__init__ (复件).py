from detector import Darknet_ObjectDetector as ObjectDetector
from detector import DetBBox

import requests
from PIL import Image
from PIL import ImageFilter
from StringIO import StringIO

def _get_image(path):
    #return Image.open(StringIO(requests.get(url).content))
	return Image.open(path)

if __name__ == '__main__':
    from PIL import Image
    voc_names = ["aeroplane", "bicycle", "bird", "boat", "bottle","bus", "car", "cat", "chair", "cow", "diningtable","dog", "horse", "motorbike", "person", "pottedplant","sheep", "sofa", "train", "tvmonitor","plate_license"]
    det = ObjectDetector('../cfg/yolo-plate.cfg','/home/himon/code/yolo/yolo-plate_final.weights')
    #url = 'http://farm9.staticflickr.com/8323/8141398311_2fd0af60f7.jpg'
    path = '/home/himon/code/yolo/cars.jpg'
    #for i in xrange(4):
    rst, run_time = det.detect_object(_get_image(path))

    print 'got {} objects in {} seconds'.format(len(rst), run_time)

    for bbox in rst:
        print '{} {} {} {} {} {}'.format(voc_names[bbox.cls], bbox.top, bbox.left, bbox.bottom, bbox.right, bbox.confidence)
