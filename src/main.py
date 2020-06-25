#! /usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function, absolute_import

from timeit import time
import os
import sys
import warnings
import cv2
import matplotlib.pyplot as plt
import numpy as np
import rospy
from PIL import Image
from cnn.yolo import YOLO

from deep_sort import preprocessing
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet
from video_reader.videocaptureasync import VideoCaptureAsync
from bebop.reactivebebop import ReactiveBebop
from video_reader.videoreader import VideoReader
import imutils.video
import argparse

current_path=os.path.dirname(os.path.abspath(__file__))

# cosntruimos los argumentos
parser = argparse.ArgumentParser(description='Bebop Tracking Script')

parser.add_argument('--path',default='/dev/video0', help='path del video a usar.\n \
 											\'bebop_cam\' para usar el dron.\
 											\n Si se deja vacio se tomara /dev/video0')
parser.add_argument("--sync", default=False, help='Si vamos a abrir un vídeo marcar a True')
parser.add_argument("--interval", default=3, help='Cada cuántos fotogramas hacemos detección')
parser.add_argument("--res", default='original', help='resolucion del video indicado')
parser.add_argument("--output", default=None, help="Path y nombre del archivo donde guardaremos la salida del tracker")
parser.add_argument("--fps_out", default=5, help="FPS del vídeo de salida. Más fps -> cámara rápida. Menos fps -> cámara lenta")

warnings.filterwarnings('ignore')

def main(yolo):
    try:
        args = parser.parse_args()
        path = args.path
        res = args.res
        output = args.output
        sync = args.sync
        interval = int(args.interval)
    except:
        args = rospy.myargv(argv=sys.argv)
        path = args[1]
        res = args[2]
        output = None
        sync = False
        interval = 3

    if res=='original':
        print('Debe indicar la resolución: width,heigh')
        exit()
    if res!='original':
        res = res.split(',')
        res = (int(res[0]), int(res[1]))

    if interval<=0: interval = 1

    max_track_ls=[0]
    min_fps_ls=[144]
    max_fps_ls=[0]
    fps_list_ls=[[]]
    ntracks_list_ls=[[]]

    # Definition of the parameters
    max_cosine_distance = 0.3
    nn_budget = None
    nms_max_overlap = 1.0
    
    # Deep SORT
    model_filename = current_path+'/../model_data/mars-small128.pb'
    encoder = gdet.create_box_encoder(model_filename,batch_size=1)
    
    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    tracker1 = Tracker(metric)
    trackers=[tracker1]

    bebop = ReactiveBebop(res)
    bebop.start()
    real_path = path
    if path=='bebop_cam':
        real_path='bebop'

    video_reader = VideoReader(src = real_path, res = res, write_path = output, sync = sync)
    readers=[video_reader]
    titulos=[path]
    if path=='bebop_cam':
        max_track_ls.append(0)
        min_fps_ls.append(144)
        max_fps_ls.append(0)
        fps_list_ls.append([])
        ntracks_list_ls.append([])

        metric2 = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
        tracker2 = Tracker(metric2)
        trackers.append(tracker2)

        cam_reader = VideoReader(src = '/dev/video0', res = (426,240))
        cam_reader.start()
        readers.append(cam_reader)
        titulos=['bebop','cam']

    if not sync: readers[0].start()

    contador=0
    con_cam=False
    while True:
        ret1, frame1 = readers[0].read()  # frame shape 640*480*3
        frames=[frame1]
        
        if not ret1:
            break
        
        if path=='bebop_cam':
            ret2, frame2 = readers[1].read()  # frame shape 640*480*3
            frames.append(frame2)
            if not ret2:
                break
        
        if not sync or contador==interval:
            contador=0

            for i, reader, frame, titu, tracker, fps_list, ntracks_list in zip([0,1],readers,frames,titulos,trackers,fps_list_ls,ntracks_list_ls):

                reader.setIniTime()
                image = Image.fromarray(frame[...,::-1])  # bgr to rgb
                #boxs = yolo.detect_image(image)[0]
                #confidence = yolo.detect_image(image)[1]
                boxs,confidence = yolo.detect_image(image)

                features = encoder(frame,boxs)

                detections = [Detection(bbox, confidence, feature) for bbox, confidence, feature in zip(boxs, confidence, features)]
                
                # Run non-maxima suppression.
                boxes = np.array([d.tlwh for d in detections])
                scores = np.array([d.confidence for d in detections])
                indices = preprocessing.non_max_suppression(boxes, nms_max_overlap, scores)
                detections = [detections[i] for i in indices]
                
                # Call the tracker
                tracker.predict()
                tracker.update(detections)
                confirmed_tracks = []
                
                for track in tracker.tracks:
                    if not track.is_confirmed() or track.time_since_update > 1:
                        continue 
                    bbox = track.to_tlbr()
                    cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),(255,255,255), 2)
                    cv2.putText(frame, str(track.track_id),(int(bbox[0]), int(bbox[1])),0, 5e-3 * 200, (0,255,0),2)
                    confirmed_tracks.append(track)
                    if track.track_id > max_track_ls[i]:
                        max_track_ls[i]=track.track_id
                
                if titu=='bebop':
                    bebop.update_tracks(confirmed_tracks)
                elif titu=='cam':
                    bebop.update_cam_tracks(confirmed_tracks)

                for det in detections:
                    bbox = det.to_tlbr()
                    score = "%.2f" % round(det.confidence * 100, 2)
                    cv2.rectangle(frame,(int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),(255,0,0), 2)
                    cv2.putText(frame, score + '%', (int(bbox[0]), int(bbox[3])), 0, 5e-3 * 130, (0,255,0),2)
                
                    
                cv2.imshow(titu, frame)
                reader.write(frame)

                fps = reader.getFPS()
                if fps < min_fps_ls[i]:
                    min_fps_ls[i]=fps
                elif fps>max_fps_ls[i]:
                    max_fps_ls[i]=fps

                fps_list.append(fps)
                ntracks_list.append(len(tracker.tracks))
                #print("FPS = %f"%(fps))
                
            # Press Q to stop!
            if cv2.waitKey(1) & 0xFF == ord('q') or not bebop.menu_alive():
                break

        contador+=1

    for reader in readers:
        reader.stopRead() # termina la lectura del video
        reader.releaseWrite() # termina la escritura de video
    bebop.stop()

    cv2.destroyWindow(real_path)
    if path=='bebop_cam':
        cv2.destroyWindow('cam')

    for titu, max_track, min_fps, max_fps, fps_list, ntracks_list in zip(titulos,max_track_ls,min_fps_ls,max_fps_ls,fps_list_ls,ntracks_list_ls):
        print('') # para dar buen formato de salida
        print(titu) # para dar buen formato de salida
        print('Max FPS: '+str(max_fps))
        print('Min FPS: '+str(min_fps))
        print('Mean FPS: '+str(sum(fps_list)/len(fps_list)))
        print('Max track: '+str(max_track))

        if output!=None:
            number = 1
            try:
                f = open('output/'+output+'.txt', "r")
                number = int(f.read().split('*')[-2])+1
                f.close()
            except: pass
            try:
                f = open('output/'+output+'.txt', "a")
                f.write('Ejecución: '+str(number)+'\n')
                f.write('Título: '+str(titu)+'\n')
                f.write('res: '+str(res)+'\n')
                f.write('Max FPS: '+str(max_fps)+'\n')
                f.write('Min FPS: '+str(min_fps)+'\n')
                f.write('Mean FPS: '+str(sum(fps_list)/len(fps_list))+'\n')
                f.write('Max track: '+str(max_track)+'\n')
                f.write('*'+str(number)+'*\n')
                f.close()
            except: pass

        fig, (ax1, ax2) = plt.subplots(2)
        fig.suptitle(titu)
        ax1.plot(fps_list)
        ax1.set_ylabel('fps')
        ax1.set_xlabel('frame')
        ax1.set_title('fps per frame')

        ax2.plot(ntracks_list)
        ax2.set_ylabel('tracks')
        ax2.set_xlabel('frame')
        ax2.set_title('tracks per frame')

        plt.show()

if __name__ == '__main__':
    main(YOLO())
