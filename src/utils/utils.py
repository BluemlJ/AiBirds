import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import backend as K


def get_moving_avg(list, n):
    """Computes the moving average with window size n of a list of numbers. The output list
    is padded at the beginning."""
    mov_avg = np.cumsum(list, dtype=float)
    mov_avg[n:] = mov_avg[n:] - mov_avg[:-n]
    mov_avg = mov_avg[n - 1:] / n
    mov_avg = np.insert(mov_avg, 0, int(n / 2) * [None])
    return mov_avg


def plot_scores(scores):
    # Window sizes for moving average
    w1 = 100
    w2 = 500
    w3 = 2000

    if len(scores) > w1:
        mov_avg_ret = get_moving_avg(scores, w1)
        plt.plot(mov_avg_ret, label="Moving average %d" % w1, c='silver')

    if len(scores) > w2:
        mov_avg_ret = get_moving_avg(scores, w2)
        plt.plot(mov_avg_ret, label="Moving average %d" % w2, c='black')

    if len(scores) > w3:
        mov_avg_ret = get_moving_avg(scores, w3)
        plt.plot(mov_avg_ret, label="Moving average %d" % w3, c='#009d81', linewidth=1.5)

    plt.title("Scores gathered so far")
    plt.xlabel("Episode")
    plt.ylabel("Score")
    plt.legend()
    plt.show()

def plot_win_loss_ratio(list_of_wins):
    # n = size of chunks - 1
    n = 100
    chunks = np.array_split(np.array(list_of_wins),n)
    print(chunks)
    
    # count the winning games per chunk divided by the size
    ratios = [np.count_nonzero(l == 1)/n for l in chunks]
    x = [n*(i+1) for i,y in enumerate(ratios)]
    print(ratios)
    
    plt.plot(x, ratios)
    plt.title("Win Loss Ratio")
    plt.xlabel("Episodes")
    plt.ylabel("Percentage")
    plt.ylim(ymin=0, ymax=1)
    plt.legend()
    plt.show()

def plot_state(state):
    # De-normalize state into image
    state = np.reshape(state, (124, 124, 3))
    image = (state * 255).astype(np.int)

    # Plot it
    plt.imshow(image)
    plt.show()

def plot_saliency_map(state, model):
    # NOT FINISHED
    # De-normalize state into the image
    state = np.reshape(state, (124, 124, 3))
    image = (state * 255).astype(np.int)
    
    # Show the image
    #plt.imshow(image)
    #plt.show()
    
    # get the gradients of the last convolutional layer 
    last_conv = model.get_layer('conv2d_3')
    #TODO get the gradients 

    # Global average pooling, returning the pooled gradients as well as the activation maps from the last conv layer
    pooled_grads = K.mean(grads,axis=(0,1,2))
    iterate = K.function([model.input],[pooled_grads,last_conv.output[0]])
    pooled_grads_value,conv_layer_output = iterate([image])

    # Multiplying the gradients and the activation maps to get importance into them 
    for i in range(512):
        conv_layer_output[:,:,i] *= pooled_grads_value[i]
    heatmap = np.mean(conv_layer_output,axis=-1)

    # apply reLU so only positive features are displayed
    for x in range(heatmap.shape[0]):
        for y in range(heatmap.shape[1]):
            heatmap[x,y] = np.max(heatmap[x,y],0)

    # normalize the heatmap
    heatmap = np.maximum(heatmap,0)
    heatmap /= np.max(heatmap)
    plt.imshow(heatmap)

    # combine heatmap and input image to get the saliency map and plot it
    upsample = resize(heatmap, (124,124),preserve_range=True)
    plt.imshow(image)
    plt.imshow(upsample,alpha=0.5)
    plt.show()

def angle_to_vector(alpha):
    rad_shot_angle = np.deg2rad(alpha)

    dx = - np.sin(rad_shot_angle) * 80
    dy = np.cos(rad_shot_angle) * 80

    return int(dx), int(dy)
