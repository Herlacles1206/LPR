#include "network.h"
#include "detection_layer.h"
#include "cost_layer.h"
#include "utils.h"
#include "parser.h"
#include "box.h"
#include "demo.h"
#include "image.h"
#include <malloc.h>
#include <stdio.h>
#include <stdlib.h>

#ifdef OPENCV
#include <opencv2/core/core_c.h>
#include "opencv2/highgui/highgui_c.h"
#endif

extern char*  plate_recognise_mat(int h,int w,int c,float *data);

#define LEN sizeof(struct BBox)
#define CLASSNUM 1
char *voc_names[] = {"plate_license"};
image voc_labels[CLASSNUM];

//char *voc_names[] = {"aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"};
//image voc_labels[20];

struct BBox{
	int left;
	int right;
	int top;
	int bottom;
	float confidence;
	int cls;
	struct BBox *next;
} _BBox;

int n;	//全局变量,在构建动态结构体数组处用.



void train_yolo(char *cfgfile, char *weightfile)
{
    char *train_images = "/data/voc/train.txt";
    char *backup_directory = "/home/pjreddie/backup/";
    srand(time(0));
    data_seed = time(0);
    char *base = basecfg(cfgfile);
    printf("%s\n", base);
    float avg_loss = -1;
    network net = parse_network_cfg(cfgfile);
    if(weightfile){
        load_weights(&net, weightfile);
    }
    printf("Learning Rate: %g, Momentum: %g, Decay: %g\n", net.learning_rate, net.momentum, net.decay);
    int imgs = net.batch*net.subdivisions;
    int i = *net.seen/imgs;
    data train, buffer;


    layer l = net.layers[net.n - 1];

    int side = l.side;
    int classes = l.classes;
    float jitter = l.jitter;

    list *plist = get_paths(train_images);
    //int N = plist->size;
    char **paths = (char **)list_to_array(plist);

    load_args args = {0};
    args.w = net.w;
    args.h = net.h;
    args.paths = paths;
    args.n = imgs;
    args.m = plist->size;
    args.classes = classes;
    args.jitter = jitter;
    args.num_boxes = side;
    args.d = &buffer;
    args.type = REGION_DATA;

    pthread_t load_thread = load_data_in_thread(args);
    clock_t time;
    //while(i*imgs < N*120){
    while(get_current_batch(net) < net.max_batches){
        i += 1;
        time=clock();
        pthread_join(load_thread, 0);
        train = buffer;
        load_thread = load_data_in_thread(args);

        printf("Loaded: %lf seconds\n", sec(clock()-time));

        time=clock();
        float loss = train_network(net, train);
        if (avg_loss < 0) avg_loss = loss;
        avg_loss = avg_loss*.9 + loss*.1;

        printf("%d: %f, %f avg, %f rate, %lf seconds, %d images\n", i, loss, avg_loss, get_current_rate(net), sec(clock()-time), i*imgs);
        if(i%1000==0 || i == 600){
            char buff[256];
            sprintf(buff, "%s/%s_%d.weights", backup_directory, base, i);
            save_weights(net, buff);
        }
        free_data(train);
    }
    char buff[256];
    sprintf(buff, "%s/%s_final.weights", backup_directory, base);
    save_weights(net, buff);
}

void convert_yolo_detections(float *predictions, int classes, int num, int square, int side, int w, int h, float thresh, float **probs, box *boxes, int only_objectness)
{
    int i,j,n;
    //int per_cell = 5*num+classes;
    for (i = 0; i < side*side; ++i){
        int row = i / side;
        int col = i % side;
        for(n = 0; n < num; ++n){
            int index = i*num + n;
            int p_index = side*side*classes + i*num + n;
            float scale = predictions[p_index];
            int box_index = side*side*(classes + num) + (i*num + n)*4;
            boxes[index].x = (predictions[box_index + 0] + col) / side * w;
            boxes[index].y = (predictions[box_index + 1] + row) / side * h;
            boxes[index].w = pow(predictions[box_index + 2], (square?2:1)) * w;
            boxes[index].h = pow(predictions[box_index + 3], (square?2:1)) * h;
            for(j = 0; j < classes; ++j){
                int class_index = i*classes;
                float prob = scale*predictions[class_index+j];
		//printf("thresh: %f \n",thresh);
                probs[index][j] = (prob > thresh) ? prob : 0;
            }
            if(only_objectness){
                probs[index][0] = scale;
            }
        }
    }
}

struct BBox * draw_detections_bbox(image im, int num, float thresh, box *boxes, float **probs, char **names, image *labels,int classes)
{
				 
	int i;
	struct BBox *head;
	struct BBox *p1,*p2;
	n = 0;
	p1=p2=(struct BBox *)malloc(LEN);
	head = NULL;	

	for (i = 0; i < num; ++i){		
		int classs = max_index(probs[i], classes);//classs:最大confidence值的下标
		float prob = probs[i][classs];
		if (prob > thresh ){
			int width = pow(prob, 1. / 2.) * 10 + 1;
			//int width = 1;
			int offset = classs * 17 % classes;
			float red = get_color(0, offset, classes);
			float green = get_color(1, offset, classes);
			float blue = get_color(2, offset, classes);
			float rgb[3];
			rgb[0] = red;
			rgb[1] = green;
			rgb[2] = blue;
			box b = boxes[i];

			int left = (b.x - b.w / 2.)*im.w;
			int right = (b.x + b.w / 2.)*im.w;
			int top = (b.y - b.h / 2.)*im.h;
			int bot = (b.y + b.h / 2.)*im.h;

			if (left < 0) left = 0;
			if (right > im.w - 1) right = im.w - 1;
			if (top < 0) top = 0;
			if (bot > im.h - 1) bot = im.h - 1;

			//printf("bbox: %s: %.2f, %d, %d, %d, %d\n", names[classs], prob, left, right, top, bot);
			//_BBox bs;
			//bs.left = left; bs.right = right; bs.top = top; bs.bottom = bot; bs.confidence = prob; bs.cls = classs;
				
			//p1 = bs;//给p1赋值//构建结构体的动态链表		
			p1->left = left;
			p1->right = right;
			p1->top = top;
			p1->bottom = bot;
			p1->confidence = prob;
			p1->cls = classs;
			
			draw_box_width(im, left, top, right, bot, width, red, green, blue);//from image.c,画框
			if (labels) draw_label(im, top + width, left, labels[classs], rgb);//如果传入label,画出label
		}//if;

		
		while((p1->confidence > thresh)&&(p1->confidence<1)){
			n = n+1;
			if(n == 1)head = p1;
			else p2->next = p1;
			p2 = p1;
			p1 = (struct BBox *)malloc(LEN);
			break;
		}			

	}//for;
	p2->next = NULL;
	return(head);
}

void print_box(struct BBox * head){
	struct BBox *p;
	printf("\nNow ,These %d bounding boxes are:\n",n);
	p = head;
	if(head!=NULL){
		do{
			printf("%s: %.2f, %d, %d, %d, %d\n",voc_names[p->cls],p->confidence,p->left,p->top,p->right,p->bottom);
			p = p->next;
		}while(p!=NULL);
	}	
}


void go_yolo(char *cfgfile, char *weightfile, char *filename,float thresh)
{

	printf("\nloading network spec from %s \n",cfgfile);
	network net = parse_network_cfg(cfgfile);//返回一个network
	printf("loading network weights from %s \n",weightfile);
	if(weightfile){
		load_weights(&net, weightfile);
    	}
	
	printf("network initialized!\n");
	
	detection_layer layer = get_network_detection_layer(net);//返回detection_layer,
	//detection_layer layer = net.layers[net.n-1];
	set_batch_network(&net, 1); 
	srand(2222222);
	
	while(1){
		detect_object(filename,net,layer,thresh);
		if(filename) break;
	}

}

void detect_object(char *filename,network net,detection_layer layer,float thresh){
	
	char buff[256];
        char *input = buff;
	int j = 0;
	float nms = .5f;
	clock_t time;
	thresh = 0.4;
	struct BBox *b;
	
	//printf("\nlayer.classes:%d layer.side:%d layer.n:%d \n",layer.classes,layer.side,layer.n);
	
	box *boxes = calloc(layer.side*layer.side*layer.n, sizeof(box));//定义了一个l.side*l.side*l.n个长度为sizeof(box)的连续空间.
	//初始化probs
    	float **probs = calloc(layer.side*layer.side*layer.n, sizeof(float *));

	for (j = 0; j < layer.side*layer.side*layer.n; j++)
	{
		probs[j] = (float *)calloc(layer.classes, sizeof(float));
	}	

	//load image
	if(filename){
            strncpy(input, filename, 256);
        } else {
            printf("Enter Image Path: ");
            fflush(stdout);
            input = fgets(input, 256, stdin);
            if(!input) return;
            strtok(input, "\n");
        }
        image im = load_image_color(input,0,0);
	image im_2 = load_image_color(input,0,0);
	//resize图片之后放入网络中预测
	image sized = resize_image(im, net.w, net.h);
	float *X = sized.data;
 	time=clock();
	float *predictions = network_predict(net, X);
	printf("%s: Predicted in %f seconds.\n", input, sec(clock()-time));
	
	
	//printf("\nlayer class: %d \n",layer.classes);
	//主要初始化boxes,probs这两个数组,boxes存放转换回来对应原图中的box.
	//probs是一个二维数组,存放网络求得的confidence.
	convert_yolo_detections(predictions, layer.classes, layer.n, layer.sqrt, layer.side, 1, 1, thresh, probs, boxes, 0);
	if (nms) do_nms_sort(boxes, probs, layer.side*layer.side*layer.n, layer.classes, nms);//一个排序
	//draw_detections(im, layer.side*layer.side*layer.n, thresh, boxes, probs, voc_names, 0, CLASSNUM); //image.c
	b = draw_detections_bbox(im, layer.side*layer.side*layer.n, thresh, boxes, probs, voc_names, 0, CLASSNUM);//建立结构体的动态链表
	//所有bbox区域的图像,并进行每一块的车牌识别，以及输出他们的坐标
	do{
		printf("%s: %.2f, %d, %d, %d, %d\n",voc_names[b->cls],b->confidence,b->left,b->top,b->right,b->bottom);
		image seg_img = crop_image(im_2,b->left,b->top,(b->right-b->left),(b->bottom-b->top));
		//save all  bboxes that were segmented out
		char buff[100];
		char str[10];
		sprintf(str,"%d",b->left);
		sprintf(buff,"/home/himon/code/yolo/darknet-master/results/%s",str);	
		save_image(seg_img, buff);
		
		//在这边调用easypr来识别车牌`
		char *plate_license;
		plate_license = plate_recognise_mat(seg_img.h,seg_img.w,seg_img.c,seg_img.data);
		printf("palte_license:%s\n",plate_license);
		
		b = b->next;
	}while(b!=NULL);	
	

	//save_image(seg_img, "/home/himon/code/yolo/darknet-master/results/seg_img");

	//print_box(b);
	free(b);

	show_image(im, "predictions");
	//save_image(im, "temp");
	//show_image(sized, "resized");
	free_image(im);
	//free_image(sized);
	#ifdef OPENCV
        cvWaitKey(0);
        cvDestroyAllWindows();
	#endif
	
};

/*
void demo_yolo(char *cfgfile, char *weightfile, float thresh, int cam_index, const char *filename,char **names, image *labels, int classes, int frame_skip)
{
	demo( cfgfile,  weightfile,  thresh,  cam_index,  filename,names, labels,  classes, frame_skip);	
}
#ifndef GPU
void demo_yolo(char *cfgfile, char *weightfile, float thresh, int cam_index, const char *filename,char **names, image *labels, int classes, int frame_skip)
{
    fprintf(stderr, "Darknet must be compiled with CUDA for YOLO demo.\n");
}
#endif

*/

void run_yolo(int argc, char **argv)
{
	print_h();
    int i;
    for(i = 0; i < CLASSNUM; ++i){
        char buff[256];
        sprintf(buff, "data/labels/%s.png", voc_names[i]);
        voc_labels[i] = load_image_color(buff, 0, 0);
    }

    float thresh = find_float_arg(argc, argv, "-thresh", .2);
    int cam_index = find_int_arg(argc, argv, "-c", 0);
    int frame_skip = find_int_arg(argc, argv, "-s", 0);
    if(argc < 4){
        fprintf(stderr, "usage: %s %s [go/train/demo] [cfg] [weights (optional)]\n", argv[0], argv[1]);
        return;
    }

    char *cfg = argv[3];
    char *weights = (argc > 4) ? argv[4] : 0;
    char *filename = (argc > 5) ? argv[5]: 0;
    if(0==strcmp(argv[2], "go")) go_yolo(cfg, weights, filename,thresh);
    else if(0==strcmp(argv[2], "train")) train_yolo(cfg, weights);
    else if(0==strcmp(argv[2], "demo")) demo(cfg, weights, thresh, cam_index, filename, voc_names, voc_labels, 20, frame_skip);

	
}
