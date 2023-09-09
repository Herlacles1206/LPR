from detector import Darknet_ObjectDetector as ObjectDetector
from detector import DetBBox

import requests
from PIL import Image
from PIL import ImageFilter
from StringIO import StringIO

import cv2

def _get_image(path):
	return Image.open(path)

if __name__ == '__main__':
    from PIL import Image
    voc_names = ["plate_license"]
    det = ObjectDetector('../cfg/yolo-plate.cfg','/home/himon/code/yolo/yolo-plate_final.weights')
    #now  network allready initialized!

   
    path = '/home/himon/code/yolo/cars.jpg'

    rst, run_time = det.detect_object(_get_image(path))

    print 'got {} objects in {} seconds'.format(len(rst), run_time)
   
    for bbox in rst:
        print '{} {} {} {} {} {}'.format(voc_names[bbox.cls], bbox.top, bbox.left, bbox.bottom, bbox.right, bbox.confidence)
    	
