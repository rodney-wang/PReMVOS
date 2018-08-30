# -*- coding: utf-8 -*-

import argparse
import json
import colorsys
import sys, os
import cv2
from PIL import Image
import numpy as np
import skimage.io
from skimage.measure import find_contours

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.animation import FFMpegWriter
import matplotlib.animation as animation

# Change the logging level to debug
#animation.verbose.set_level('debug')


def apply_mask(image, mask, color, alpha=0.5):
    """Apply the given mask to the image.
    """
    for c in range(3):
        image[:, :, c] = np.where(mask == 1,
                                  image[:, :, c] *
                                  (1 - alpha) + alpha * color[c] * 255,
                                  image[:, :, c])
    return image


def plot_with_mask(image, masks, palette):
    """
    plot a single image with multiple instance masks.
    :param image:   single frame from a video
    :param masks:   with values from 0-N, where each number represent one single instance
    :param palette:  color palette is used to write out the video with mask
    :return:     masked_image
    """

    # Generate plotting colors
    colors = palette
    masked_image = image.astype(np.uint8).copy()
    mask_idx = np.unique(masks)

    for i in mask_idx:
        color = colors[i]

        # Mask
        mask = (masks ==i )
        masked_image = apply_mask(masked_image, mask, color)

        # Mask Polygon
        # Pad to ensure proper polygons for masks that touch image edges.
        padded_mask = np.zeros(
            (mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.uint8)
        padded_mask[1:-1, 1:-1] = mask
        contours = find_contours(padded_mask, 0.5)

        for verts in contours:
            # Subtract the padding and flip (y, x) to (x, y)
            verts = np.fliplr(verts) - 1
            #p = Polygon(verts, facecolor="none", edgecolor=color)
            #ax.add_patch(p)
            pts = verts.reshape((-1, 1, 2))
            cv2.polylines(masked_image, np.int32([pts]), 1,  color*255 , 4)

    return masked_image

def plot_tracking_with_mask(frame_dir, label_dir, video_path, seq_id):
    """
    :param frame_dir:    path to video frames
    :param label_dir:    path to output label frames
    :param video_path:   output video path
    :param seq_id:       name of the SEQUENCE
    :return:
    """

    display_window = True

    palette = np.loadtxt('config/palette.txt')
    palette = palette/255.0;
    fps=25.0//2
    frames_num = len(os.listdir(frame_dir))

    frame = skimage.io.imread(os.path.join(frame_dir, '%05d.jpg' % 0))
    height, width, channel = frame.shape

    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    cap_out = cv2.VideoWriter(video_path, fourcc, fps, (width, height))


    for pos_frame in range(frames_num):
        print('Plotting frame:', pos_frame)
        #frame = cv2.imread(os.path.join(frame_dir, '%05d.jpg' % pos_frame))
        frame = skimage.io.imread(os.path.join(frame_dir, '%05d.jpg' % pos_frame))

        label_img = Image.open(os.path.join(label_dir, '%05d.png' % pos_frame))
        label = np.array(label_img)

        masked_frame=plot_with_mask(frame, label, palette)
        #masked_frame = cv2.cvtColor(masked_frame, cv2.COLOR_RGB2BGR)

        if display_window:
            cv2.imshow(seq_id, masked_frame)

        cap_out.write(masked_frame)

        if cv2.waitKey(10) == 27:
            break

    cap_out.release()
    cv2.destroyAllWindows()

    print('output to {}'.format(video_path))


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('data_dir', type=str,
                        help='base data path to frames and labels')

    parser.add_argument('seq_id', type=str,
                        help='seqence id')

    parser.add_argument('video_output_path', type=str,
                        help='output video path')

    args = parser.parse_args()

    frame_path=os.path.join(args.data_dir, 'JPEGImages', '480p', args.seq_id)
    mask_path = os.path.join(args.data_dir, 'Results/Segmentations/480p/output/result_vsreid_0828', args.seq_id)
    #video_output_filename=os.path.join(args.data_dir, 'Results', args.seq_id + '_result.avi')
    print(args.video_output_path)

    plot_tracking_with_mask(frame_path, mask_path, args.video_output_path, args.seq_id)



if __name__ == "__main__":


    data_dir = '/home/administrator/data/DAVIS_All/'
    seq_id = 'girl-dog'

    frame_path=os.path.join(data_dir, 'JPEGImages', '480p', seq_id)
    mask_path = os.path.join(data_dir, 'Results/Segmentations/480p/output/result_vsreid_0828', seq_id)
    video_output_filename=os.path.join(data_dir, 'Results', seq_id + '_result.avi')
    print(video_output_filename)

    plot_tracking_with_mask(frame_path, mask_path, video_output_filename, seq_id)
