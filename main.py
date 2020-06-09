#! /usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function, absolute_import

from timeit import time
import warnings
import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from yolo import YOLO

from deep_sort import preprocessing
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet
from videocaptureasync import VideoCaptureAsync
from reactivebebop import ReactiveBebop
from videoreader import VideoReader
import imutils.video
import argparse

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
args = parser.parse_args()

warnings.filterwarnings('ignore')

def main(yolo):
    path = args.path
    res = args.res
    if res=='original':
        print('Debe indicar la resolución: width,heigh')
        exit()
    if res!='original':
        res = res.split(',')
        res = (int(res[0]), int(res[1]))
    output = args.output
    sync = args.sync
    interval = int(args.interval)
    if interval<=0: interval = 1

    max_track=0
    min_fps=144
    max_fps=0
    fps_list=[]
    ntracks_list=[]

    # Definition of the parameters
    max_cosine_distance = 0.3
    nn_budget = None
    nms_max_overlap = 1.0
    
    # Deep SORT
    model_filename = 'model_data/mars-small128.pb'
    encoder = gdet.create_box_encoder(model_filename,batch_size=1)
    
    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    tracker = Tracker(metric)

    bebop = ReactiveBebop()
    bebop.start()
    video_reader = VideoReader(src = path, res = res, write_path = output, sync = sync)
    if not sync: video_reader.start()

    contador=0
    while True:
        ret, frame = video_reader.read()  # frame shape 640*480*3
        if ret != True:
            break
        if not sync or contador==interval:
            contador=0

            video_reader.setIniTime()

            image = Image.fromarray(frame[...,::-1])  # bgr to rgb
            boxs = yolo.detect_image(image)[0]
            confidence = yolo.detect_image(image)[1]

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
                if track.track_id > max_track:
                    max_track=track.track_id
            
            bebop.update_tracks(confirmed_tracks)

            
            for det in detections:
                bbox = det.to_tlbr()
                score = "%.2f" % round(det.confidence * 100, 2)
                #cv2.rectangle(frame,(int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),(255,0,0), 2)
                cv2.putText(frame, score + '%', (int(bbox[0]), int(bbox[3])), 0, 5e-3 * 130, (0,255,0),2)
            
                
            cv2.imshow('', frame)
            video_reader.write(frame)

            fps = video_reader.getFPS()
            if fps < min_fps:
                min_fps=fps
            elif fps>max_fps:
                max_fps=fps
            fps_list.append(fps)
            ntracks_list.append(len(tracker.tracks))
            #print("FPS = %f"%(fps))
            
            # Press Q to stop!
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            if not bebop.menu_alive(): break

        contador+=1

    video_reader.stopRead() # termina la lectura del video
    video_reader.releaseWrite() # termina la escritura de video
    bebop.stop()

    cv2.destroyAllWindows()

    print('') # para dar buen formato de salida
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
            f.write('res: '+str(res)+'\n')
            f.write('Max FPS: '+str(max_fps)+'\n')
            f.write('Min FPS: '+str(min_fps)+'\n')
            f.write('Mean FPS: '+str(sum(fps_list)/len(fps_list))+'\n')
            f.write('Max track: '+str(max_track)+'\n')
            f.write('*'+str(number)+'*\n')
            f.close()
        except: pass

    fig, (ax1, ax2) = plt.subplots(2)
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
