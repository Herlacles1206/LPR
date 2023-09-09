#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#include "parser.h"
#include "activations.h"
#include "crop_layer.h"
#include "cost_layer.h"
#include "convolutional_layer.h"
#include "activation_layer.h"
#include "normalization_layer.h"
#include "deconvolutional_layer.h"
#include "connected_layer.h"
#include "rnn_layer.h"
#include "maxpool_layer.h"
#include "softmax_layer.h"
#include "dropout_layer.h"
#include "detection_layer.h"
#include "avgpool_layer.h"
#include "local_layer.h"
#include "route_layer.h"
#include "shortcut_layer.h"
#include "list.h"
#include "option_list.h"
#include "utils.h"

typedef struct{
    char *type;
    list *options;
}section;

int is_network(section *s);
int is_convolutional(section *s);
int is_activation(section *s);
int is_local(section *s);
int is_deconvolutional(section *s);
int is_connected(section *s);
int is_rnn(section *s);
int is_maxpool(section *s);
int is_avgpool(section *s);
int is_dropout(section *s);
int is_softmax(section *s);
int is_normalization(section *s);
int is_crop(section *s);
int is_shortcut(section *s);
int is_cost(section *s);
int is_detection(section *s);
int is_route(section *s);
list *read_cfg(char *filename);

void free_section(section *s)
{
    free(s->type);
    node *n = s->options->front;
    while(n){
        kvp *pair = (kvp *)n->val;
        free(pair->key);
        free(pair);
        node *next = n->next;
        free(n);
        n = next;
    }
    free(s->options);
    free(s);
}

void parse_data(char *data, float *a, int n)
{
    int i;
    if(!data) return;
    char *curr = data;
    char *next = data;
    int done = 0;
    for(i = 0; i < n && !done; ++i){
        while(*++next !='\0' && *next != ',');
        if(*next == '\0') done = 1;
        *next = '\0';
        sscanf(curr, "%g", &a[i]);
        curr = next+1;
    }
}

typedef struct size_params{
    int batch;
    int inputs;
    int h;
    int w;
    int c;
    int index;
    int time_steps;
} size_params;

deconvolutional_layer parse_deconvolutional(list *options, size_params params)
{
    int n = option_find_int(options, "filters",1);
    int size = option_find_int(options, "size",1);
    int stride = option_find_int(options, "stride",1);
    char *activation_s = option_find_str(options, "activation", "logistic");
    ACTIVATION activation = get_activation(activation_s);

    int batch,h,w,c;
    h = params.h;
    w = params.w;
    c = params.c;
    batch=params.batch;
    if(!(h && w && c)) error("Layer before deconvolutional layer must output image.");

    deconvolutional_layer layer = make_deconvolutional_layer(batch,h,w,c,n,size,stride,activation);

    char *weights = option_find_str(options, "weights", 0);
    char *biases = option_find_str(options, "biases", 0);
    parse_data(weights, layer.filters, c*n*size*size);
    parse_data(biases, layer.biases, n);
    #ifdef GPU
    if(weights || biases) push_deconvolutional_layer(layer);
    #endif
    return layer;
}

local_layer parse_local(list *options, size_params params)
{
    int n = option_find_int(options, "filters",1);
    int size = option_find_int(options, "size",1);
    int stride = option_find_int(options, "stride",1);
    int pad = option_find_int(options, "pad",0);
    char *activation_s = option_find_str(options, "activation", "logistic");
    ACTIVATION activation = get_activation(activation_s);

    int batch,h,w,c;
    h = params.h;
    w = params.w;
    c = params.c;
    batch=params.batch;
    if(!(h && w && c)) error("Layer before local layer must output image.");

    local_layer layer = make_local_layer(batch,h,w,c,n,size,stride,pad,activation);

    return layer;
}

convolutional_layer parse_convolutional(list *options, size_params params)
{
    int n = option_find_int(options, "filters",1);
    int size = option_find_int(options, "size",1);
    int stride = option_find_int(options, "stride",1);
    int pad = option_find_int(options, "pad",0);
    char *activation_s = option_find_str(options, "activation", "logistic");
    ACTIVATION activation = get_activation(activation_s);

    int batch,h,w,c;
    h = params.h;
    w = params.w;
    c = params.c;
    batch=params.batch;
    if(!(h && w && c)) error("Layer before convolutional layer must output image.");
    int batch_normalize = option_find_int_quiet(options, "batch_normalize", 0);
    int binary = option_find_int_quiet(options, "binary", 0);

    convolutional_layer layer = make_convolutional_layer(batch,h,w,c,n,size,stride,pad,activation, batch_normalize, binary);
    layer.flipped = option_find_int_quiet(options, "flipped", 0);

    char *weights = option_find_str(options, "weights", 0);
    char *biases = option_find_str(options, "biases", 0);
    parse_data(weights, layer.filters, c*n*size*size);
    parse_data(biases, layer.biases, n);
    #ifdef GPU
    if(weights || biases) push_convolutional_layer(layer);
    #endif
    return layer;
}

layer parse_rnn(list *options, size_params params)
{
    int output = option_find_int(options, "output",1);
    int hidden = option_find_int(options, "hidden",1);
    char *activation_s = option_find_str(options, "activation", "logistic");
    ACTIVATION activation = get_activation(activation_s);
    int batch_normalize = option_find_int_quiet(options, "batch_normalize", 0);
    int logistic = option_find_int_quiet(options, "logistic", 0);

    layer l = make_rnn_layer(params.batch, params.inputs, hidden, output, params.time_steps, activation, batch_normalize, logistic);

    l.shortcut = option_find_int_quiet(options, "shortcut", 0);

    return l;
}

connected_layer parse_connected(list *options, size_params params)
{
    int output = option_find_int(options, "output",1);
    char *activation_s = option_find_str(options, "activation", "logistic");
    ACTIVATION activation = get_activation(activation_s);
    int batch_normalize = option_find_int_quiet(options, "batch_normalize", 0);

    connected_layer layer = make_connected_layer(params.batch, params.inputs, output, activation, batch_normalize);

    char *weights = option_find_str(options, "weights", 0);
    char *biases = option_find_str(options, "biases", 0);
    parse_data(biases, layer.biases, output);
    parse_data(weights, layer.weights, params.inputs*output);
    #ifdef GPU
    if(weights || biases) push_connected_layer(layer);
    #endif
    return layer;
}

softmax_layer parse_softmax(list *options, size_params params)
{
    int groups = option_find_int_quiet(options, "groups",1);
    softmax_layer layer = make_softmax_layer(params.batch, params.inputs, groups);
    layer.temperature = option_find_float_quiet(options, "temperature", 1);
    return layer;
}

detection_layer parse_detection(list *options, size_params params)
{
    int coords = option_find_int(options, "coords", 1);
    int classes = option_find_int(options, "classes", 1);
    int rescore = option_find_int(options, "rescore", 0);
    int num = option_find_int(options, "num", 1);
    int side = option_find_int(options, "side", 7);
    detection_layer layer = make_detection_layer(params.batch, params.inputs, num, side, classes, coords, rescore);

    layer.softmax = option_find_int(options, "softmax", 0);
    layer.sqrt = option_find_int(options, "sqrt", 0);

    layer.coord_scale = option_find_float(options, "coord_scale", 1);
    layer.forced = option_find_int(options, "forced", 0);
    layer.object_scale = option_find_float(options, "object_scale", 1);
    layer.noobject_scale = option_find_float(options, "noobject_scale", 1);
    layer.class_scale = option_find_float(options, "class_scale", 1);
    layer.jitter = option_find_float(options, "jitter", .2);
    return layer;
}

cost_layer parse_cost(list *options, size_params params)
{
    char *type_s = option_find_str(options, "type", "sse");
    COST_TYPE type = get_cost_type(type_s);
    float scale = option_find_float_quiet(options, "scale",1);
    cost_layer layer = make_cost_layer(params.batch, params.inputs, type, scale);
    return layer;
}

crop_layer parse_crop(list *options, size_params params)
{
    int crop_height = option_find_int(options, "crop_height",1);
    int crop_width = option_find_int(options, "crop_width",1);
    int flip = option_find_int(options, "flip",0);
    float angle = option_find_float(options, "angle",0);
    float saturation = option_find_float(options, "saturation",1);
    float exposure = option_find_float(options, "exposure",1);

    int batch,h,w,c;
    h = params.h;
    w = params.w;
    c = params.c;
    batch=params.batch;
    if(!(h && w && c)) error("Layer before crop layer must output image.");

    int noadjust = option_find_int_quiet(options, "noadjust",0);

    crop_layer l = make_crop_layer(batch,h,w,c,crop_height,crop_width,flip, angle, saturation, exposure);
    l.shift = option_find_float(options, "shift", 0);
    l.noadjust = noadjust;
    return l;
}

maxpool_layer parse_maxpool(list *options, size_params params)
{
    int stride = option_find_int(options, "stride",1);
    int size = option_find_int(options, "size",stride);

    int batch,h,w,c;
    h = params.h;
    w = params.w;
    c = params.c;
    batch=params.batch;
    if(!(h && w && c)) error("Layer before maxpool layer must output image.");

    maxpool_layer layer = make_maxpool_layer(batch,h,w,c,size,stride);
    return layer;
}

avgpool_layer parse_avgpool(list *options, size_params params)
{
    int batch,w,h,c;
    w = params.w;
    h = params.h;
    c = params.c;
    batch=params.batch;
    if(!(h && w && c)) error("Layer before avgpool layer must output image.");

    avgpool_layer layer = make_avgpool_layer(batch,w,h,c);
    return layer;
}

dropout_layer parse_dropout(list *options, size_params params)
{
    float probability = option_find_float(options, "probability", .5);
    dropout_layer layer = make_dropout_layer(params.batch, params.inputs, probability);
    layer.out_w = params.w;
    layer.out_h = params.h;
    layer.out_c = params.c;
    return layer;
}

layer parse_normalization(list *options, size_params params)
{
    float alpha = option_find_float(options, "alpha", .0001);
    float beta =  option_find_float(options, "beta" , .75);
    float kappa = option_find_float(options, "kappa", 1);
    int size = option_find_int(options, "size", 5);
    layer l = make_normalization_layer(params.batch, params.w, params.h, params.c, size, alpha, beta, kappa);
    return l;
}

layer parse_shortcut(list *options, size_params params, network net)
{
    char *l = option_find(options, "from");   
    int index = atoi(l);
    if(index < 0) index = params.index + index;

    int batch = params.batch;
    layer from = net.layers[index];

    layer s = make_shortcut_layer(batch, index, params.w, params.h, params.c, from.out_w, from.out_h, from.out_c);

    char *activation_s = option_find_str(options, "activation", "linear");
    ACTIVATION activation = get_activation(activation_s);
    s.activation = activation;
    return s;
}


layer parse_activation(list *options, size_params params)
{
    char *activation_s = option_find_str(options, "activation", "linear");
    ACTIVATION activation = get_activation(activation_s);

    layer l = make_activation_layer(params.batch, params.inputs, activation);

    l.out_h = params.h;
    l.out_w = params.w;
    l.out_c = params.c;
    l.h = params.h;
    l.w = params.w;
    l.c = params.c;

    return l;
}

route_layer parse_route(list *options, size_params params, network net)
{
    char *l = option_find(options, "layers");   
    int len = strlen(l);
    if(!l) error("Route Layer must specify input layers");
    int n = 1;
    int i;
    for(i = 0; i < len; ++i){
        if (l[i] == ',') ++n;
    }

    int *layers = calloc(n, sizeof(int));
    int *sizes = calloc(n, sizeof(int));
    for(i = 0; i < n; ++i){
        int index = atoi(l);
        l = strchr(l, ',')+1;
        if(index < 0) index = params.index + index;
        layers[i] = index;
        sizes[i] = net.layers[index].outputs;
    }
    int batch = params.batch;

    route_layer layer = make_route_layer(batch, n, layers, sizes);

    convolutional_layer first = net.layers[layers[0]];
    layer.out_w = first.out_w;
    layer.out_h = first.out_h;
    layer.out_c = first.out_c;
    for(i = 1; i < n; ++i){
        int index = layers[i];
        convolutional_layer next = net.layers[index];
        if(next.out_w == first.out_w && next.out_h == first.out_h){
            layer.out_c += next.out_c;
        }else{
            layer.out_h = layer.out_w = layer.out_c = 0;
        }
    }

    return layer;
}

learning_rate_policy get_policy(char *s)
{
    if (strcmp(s, "poly")==0) return POLY;
    if (strcmp(s, "constant")==0) return CONSTANT;
    if (strcmp(s, "step")==0) return STEP;
    if (strcmp(s, "exp")==0) return EXP;
    if (strcmp(s, "sigmoid")==0) return SIG;
    if (strcmp(s, "steps")==0) return STEPS;
    fprintf(stderr, "Couldn't find policy %s, going with constant\n", s);
    return CONSTANT;
}

void parse_net_options(list *options, network *net)
{
    net->batch = option_find_int(options, "batch",1);
    net->learning_rate = option_find_float(options, "learning_rate", .001);
    net->momentum = option_find_float(options, "momentum", .9);
    net->decay = option_find_float(options, "decay", .0001);
    int subdivs = option_find_int(options, "subdivisions",1);
    net->time_steps = option_find_int_quiet(options, "time_steps",1);
    net->batch /= subdivs;
    net->batch *= net->time_steps;
    net->subdivisions = subdivs;

    net->h = option_find_int_quiet(options, "height",0);
    net->w = option_find_int_quiet(options, "width",0);
    net->c = option_find_int_quiet(options, "channels",0);
    net->inputs = option_find_int_quiet(options, "inputs", net->h * net->w * net->c);

    if(!net->inputs && !(net->h && net->w && net->c)) error("No input parameters supplied");

    char *policy_s = option_find_str(options, "policy", "constant");
    net->policy = get_policy(policy_s);
    if(net->policy == STEP){
        net->step = option_find_int(options, "step", 1);
        net->scale = option_find_float(options, "scale", 1);
    } else if (net->policy == STEPS){
        char *l = option_find(options, "steps");   
        char *p = option_find(options, "scales");   
        if(!l || !p) error("STEPS policy must have steps and scales in cfg file");

        int len = strlen(l);
        int n = 1;
        int i;
        for(i = 0; i < len; ++i){
            if (l[i] == ',') ++n;
        }
        int *steps = calloc(n, sizeof(int));
        float *scales = calloc(n, sizeof(float));
        for(i = 0; i < n; ++i){
            int step    = atoi(l);
            float scale = atof(p);
            l = strchr(l, ',')+1;
            p = strchr(p, ',')+1;
            steps[i] = step;
            scales[i] = scale;
        }
        net->scales = scales;
        net->steps = steps;
        net->num_steps = n;
    } else if (net->policy == EXP){
        net->gamma = option_find_float(options, "gamma", 1);
    } else if (net->policy == SIG){
        net->gamma = option_find_float(options, "gamma", 1);
        net->step = option_find_int(options, "step", 1);
    } else if (net->policy == POLY){
        net->power = option_find_float(options, "power", 1);
    }
    net->max_batches = option_find_int(options, "max_batches", 0);
}

network parse_network_cfg(char *filename)
{
    list *sections = read_cfg(filename);
    node *n = sections->front;
    if(!n) error("Config file has no sections");
    network net = make_network(sections->size - 1);
    size_params params;

    section *s = (section *)n->val;
    list *options = s->options;
    if(!is_network(s)) error("First section must be [net] or [network]");
    parse_net_options(options, &net);

    params.h = net.h;
    params.w = net.w;
    params.c = net.c;
    params.inputs = net.inputs;
    params.batch = net.batch;
    params.time_steps = net.time_steps;

    n = n->next;
    int count = 0;
    free_section(s);
    while(n){
        params.index = count;
        fprintf(stderr, "%d: ", count);
        s = (section *)n->val;
        options = s->options;
        layer l = {0};
        if(is_convolutional(s)){
            l = parse_convolutional(options, params);
        }else if(is_local(s)){
            l = parse_local(options, params);
        }else if(is_activation(s)){
            l = parse_activation(options, params);
        }else if(is_deconvolutional(s)){
            l = parse_deconvolutional(options, params);
        }else if(is_rnn(s)){
            l = parse_rnn(options, params);
        }else if(is_connected(s)){
            l = parse_connected(options, params);
        }else if(is_crop(s)){
            l = parse_crop(options, params);
        }else if(is_cost(s)){
            l = parse_cost(options, params);
        }else if(is_detection(s)){
            l = parse_detection(options, params);
        }else if(is_softmax(s)){
            l = parse_softmax(options, params);
        }else if(is_normalization(s)){
            l = parse_normalization(options, params);
        }else if(is_maxpool(s)){
            l = parse_maxpool(options, params);
        }else if(is_avgpool(s)){
            l = parse_avgpool(options, params);
        }else if(is_route(s)){
            l = parse_route(options, params, net);
        }else if(is_shortcut(s)){
            l = parse_shortcut(options, params, net);
        }else if(is_dropout(s)){
            l = parse_dropout(options, params);
            l.output = net.layers[count-1].output;
            l.delta = net.layers[count-1].delta;
#ifdef GPU
            l.output_gpu = net.layers[count-1].output_gpu;
            l.delta_gpu = net.layers[count-1].delta_gpu;
#endif
        }else{
            fprintf(stderr, "Type not recognized: %s\n", s->type);
        }
        l.dontload = option_find_int_quiet(options, "dontload", 0);
        l.dontloadscales = option_find_int_quiet(options, "dontloadscales", 0);
        option_unused(options);
        net.layers[count] = l;
        free_section(s);
        n = n->next;
        ++count;
        if(n){
            params.h = l.out_h;
            params.w = l.out_w;
            params.c = l.out_c;
            params.inputs = l.outputs;
        }
    }   
    free_list(sections);
    net.outputs = get_network_output_size(net);
    net.output = get_network_output(net);
    return net;
}

int is_shortcut(section *s)
{
    return (strcmp(s->type, "[shortcut]")==0);
}
int is_crop(section *s)
{
    return (strcmp(s->type, "[crop]")==0);
}
int is_cost(section *s)
{
    return (strcmp(s->type, "[cost]")==0);
}
int is_detection(section *s)
{
    return (strcmp(s->type, "[detection]")==0);
}
int is_local(section *s)
{
    return (strcmp(s->type, "[local]")==0);
}
int is_deconvolutional(section *s)
{
    return (strcmp(s->type, "[deconv]")==0
            || strcmp(s->type, "[deconvolutional]")==0);
}
int is_convolutional(section *s)
{
    return (strcmp(s->type, "[conv]")==0
            || strcmp(s->type, "[convolutional]")==0);
}
int is_activation(section *s)
{
    return (strcmp(s->type, "[activation]")==0);
}
int is_network(section *s)
{
    return (strcmp(s->type, "[net]")==0
            || strcmp(s->type, "[network]")==0);
}
int is_rnn(section *s)
{
    return (strcmp(s->type, "[rnn]")==0);
}
int is_connected(section *s)
{
    return (strcmp(s->type, "[conn]")==0
            || strcmp(s->type, "[connected]")==0);
}
int is_maxpool(section *s)
{
    return (strcmp(s->type, "[max]")==0
            || strcmp(s->type, "[maxpool]")==0);
}
int is_avgpool(section *s)
{
    return (strcmp(s->type, "[avg]")==0
            || strcmp(s->type, "[avgpool]")==0);
}
int is_dropout(section *s)
{
    return (strcmp(s->type, "[dropout]")==0);
}

int is_normalization(section *s)
{
    return (strcmp(s->type, "[lrn]")==0
            || strcmp(s->type, "[normalization]")==0);
}

int is_softmax(section *s)
{
    return (strcmp(s->type, "[soft]")==0
            || strcmp(s->type, "[softmax]")==0);
}
int is_route(section *s)
{
    return (strcmp(s->type, "[route]")==0);
}

list *read_cfg(char *filename)
{
    FILE *file = fopen(filename, "r");
    if(file == 0) file_error(filename);
    char *line;
    int nu = 0;
    list *sections = make_list();
    section *current = 0;
    while((line=fgetl(file)) != 0){
        ++ nu;
        strip(line);
        switch(line[0]){
            case '[':
                current = malloc(sizeof(section));
                list_insert(sections, current);
                current->options = make_list();
                current->type = line;
                break;
            case '\0':
            case '#':
            case ';':
                free(line);
                break;
            default:
                if(!read_option(line, current->options)){
                    fprintf(stderr, "Config file error line %d, could parse: %s\n", nu, line);
                    free(line);
                }
                break;
        }
    }
    fclose(file);
    return sections;
}

void save_weights_double(network net, char *filename)
{
    fprintf(stderr, "Saving doubled weights to %s\n", filename);
    FILE *fp = fopen(filename, "w");
    if(!fp) file_error(filename);

    fwrite(&net.learning_rate, sizeof(float), 1, fp);
    fwrite(&net.momentum, sizeof(float), 1, fp);
    fwrite(&net.decay, sizeof(float), 1, fp);
    fwrite(net.seen, sizeof(int), 1, fp);

    int i,j,k;
    for(i = 0; i < net.n; ++i){
        layer l = net.layers[i];
        if(l.type == CONVOLUTIONAL){
#ifdef GPU
            if(gpu_index >= 0){
                pull_convolutional_layer(l);
            }
#endif
            float zero = 0;
            fwrite(l.biases, sizeof(float), l.n, fp);
            fwrite(l.biases, sizeof(float), l.n, fp);

            for (j = 0; j < l.n; ++j){
                int index = j*l.c*l.size*l.size;
                fwrite(l.filters+index, sizeof(float), l.c*l.size*l.size, fp);
                for (k = 0; k < l.c*l.size*l.size; ++k) fwrite(&zero, sizeof(float), 1, fp);
            }
            for (j = 0; j < l.n; ++j){
                int index = j*l.c*l.size*l.size;
                for (k = 0; k < l.c*l.size*l.size; ++k) fwrite(&zero, sizeof(float), 1, fp);
                fwrite(l.filters+index, sizeof(float), l.c*l.size*l.size, fp);
            }
        }
    }
    fclose(fp);
}

void save_connected_weights(layer l, FILE *fp)
{
#ifdef GPU
    if(gpu_index >= 0){
        pull_connected_layer(l);
    }
#endif
    fwrite(l.biases, sizeof(float), l.outputs, fp);
    fwrite(l.weights, sizeof(float), l.outputs*l.inputs, fp);
    if (l.batch_normalize){
        fwrite(l.scales, sizeof(float), l.outputs, fp);
        fwrite(l.rolling_mean, sizeof(float), l.outputs, fp);
        fwrite(l.rolling_variance, sizeof(float), l.outputs, fp);
    }
}

void save_weights_upto(network net, char *filename, int cutoff)
{
    fprintf(stderr, "Saving weights to %s\n", filename);
    FILE *fp = fopen(filename, "w");
    if(!fp) file_error(filename);

    int major = 0;
    int minor = 1;
    int revision = 0;
    fwrite(&major, sizeof(int), 1, fp);
    fwrite(&minor, sizeof(int), 1, fp);
    fwrite(&revision, sizeof(int), 1, fp);
    fwrite(net.seen, sizeof(int), 1, fp);

    int i;
    for(i = 0; i < net.n && i < cutoff; ++i){
        layer l = net.layers[i];
        if(l.type == CONVOLUTIONAL){
#ifdef GPU
            if(gpu_index >= 0){
                pull_convolutional_layer(l);
            }
#endif
            int num = l.n*l.c*l.size*l.size;
            fwrite(l.biases, sizeof(float), l.n, fp);
            if (l.batch_normalize){
                fwrite(l.scales, sizeof(float), l.n, fp);
                fwrite(l.rolling_mean, sizeof(float), l.n, fp);
                fwrite(l.rolling_variance, sizeof(float), l.n, fp);
            }
            fwrite(l.filters, sizeof(float), num, fp);
        } if(l.type == CONNECTED){
            save_connected_weights(l, fp);
        } if(l.type == RNN){
            save_connected_weights(*(l.input_layer), fp);
            save_connected_weights(*(l.self_layer), fp);
            save_connected_weights(*(l.output_layer), fp);
        } if(l.type == LOCAL){
#ifdef GPU
            if(gpu_index >= 0){
                pull_local_layer(l);
            }
#endif
            int locations = l.out_w*l.out_h;
            int size = l.size*l.size*l.c*l.n*locations;
            fwrite(l.biases, sizeof(float), l.outputs, fp);
            fwrite(l.filters, sizeof(float), size, fp);
        }
    }
    fclose(fp);
}
void save_weights(network net, char *filename)
{
    save_weights_upto(net, filename, net.n);
}

void transpose_matrix(float *a, int rows, int cols)
{
    float *transpose = calloc(rows*cols, sizeof(float));
    int x, y;
    for(x = 0; x < rows; ++x){
        for(y = 0; y < cols; ++y){
            transpose[y*rows + x] = a[x*cols + y];
        }
    }
    memcpy(a, transpose, rows*cols*sizeof(float));
    free(transpose);
}

void load_connected_weights(layer l, FILE *fp, int transpose)
{
    fread(l.biases, sizeof(float), l.outputs, fp);
    fread(l.weights, sizeof(float), l.outputs*l.inputs, fp);
    if(transpose){
        transpose_matrix(l.weights, l.inputs, l.outputs);
    }
    if (l.batch_normalize && (!l.dontloadscales)){
        fread(l.scales, sizeof(float), l.outputs, fp);
        fread(l.rolling_mean, sizeof(float), l.outputs, fp);
        fread(l.rolling_variance, sizeof(float), l.outputs, fp);
    }
#ifdef GPU
    if(gpu_index >= 0){
        push_connected_layer(l);
    }
#endif
}

void load_weights_upto(network *net, char *filename, int cutoff)
{
    fprintf(stderr, "Loading weights from %s...", filename);
    fflush(stdout);
    FILE *fp = fopen(filename, "rb");
    if(!fp) file_error(filename);

    int major;
    int minor;
    int revision;
    fread(&major, sizeof(int), 1, fp);
    fread(&minor, sizeof(int), 1, fp);
    fread(&revision, sizeof(int), 1, fp);
    fread(net->seen, sizeof(int), 1, fp);
    int transpose = (major > 1000) || (minor > 1000);

    int i;
    for(i = 0; i < net->n && i < cutoff; ++i){
        layer l = net->layers[i];
        if (l.dontload) continue;
        if(l.type == CONVOLUTIONAL){
            int num = l.n*l.c*l.size*l.size;
            fread(l.biases, sizeof(float), l.n, fp);
            if (l.batch_normalize && (!l.dontloadscales)){
                fread(l.scales, sizeof(float), l.n, fp);
                fread(l.rolling_mean, sizeof(float), l.n, fp);
                fread(l.rolling_variance, sizeof(float), l.n, fp);
            }
            fread(l.filters, sizeof(float), num, fp);
            if (l.flipped) {
                transpose_matrix(l.filters, l.c*l.size*l.size, l.n);
            }
#ifdef GPU
            if(gpu_index >= 0){
                push_convolutional_layer(l);
            }
#endif
        }
        if(l.type == DECONVOLUTIONAL){
            int num = l.n*l.c*l.size*l.size;
            fread(l.biases, sizeof(float), l.n, fp);
            fread(l.filters, sizeof(float), num, fp);
#ifdef GPU
            if(gpu_index >= 0){
                push_deconvolutional_layer(l);
            }
#endif
        }
        if(l.type == CONNECTED){
            load_connected_weights(l, fp, transpose);
        }
        if(l.type == RNN){
            load_connected_weights(*(l.input_layer), fp, transpose);
            load_connected_weights(*(l.self_layer), fp, transpose);
            load_connected_weights(*(l.output_layer), fp, transpose);
        }
        if(l.type == LOCAL){
            int locations = l.out_w*l.out_h;
            int size = l.size*l.size*l.c*l.n*locations;
            fread(l.biases, sizeof(float), l.outputs, fp);
            fread(l.filters, sizeof(float), size, fp);
#ifdef GPU
            if(gpu_index >= 0){
                push_local_layer(l);
            }
#endif
        }
    }
    fprintf(stderr, "Done!\n");
    fclose(fp);
}

void load_weights(network *net, char *filename)
{
    load_weights_upto(net, filename, net->n);
}

