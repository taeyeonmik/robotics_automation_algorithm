import argparse
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel

import mediapipe as mp


def detect(source, weights, view_img, save_conf, save_txt, classes, imgsz, trace,
           project, name, exist_ok, device, img_size, conf_thres, iou_thres, inj_conf_thres,
           agnostic_nms, augment, save_img=False):

    # source, weights, view_img, save_txt, imgsz, trace = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size, not opt.no_trace
    # save_img = not opt.nosave and not source.endswith('.txt')  # save inference images
    # source.isnumeric() -> True
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))

    # Directories
    save_dir = Path(increment_path(Path(project) / name, exist_ok=exist_ok))  # increment run
    # save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
    (save_dir / 'bbox' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir
    (save_dir / 'kpts' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Initialize
    set_logging()
    device = select_device(device)
    # device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size

    if trace:
        # model = TracedModel(model, device, opt.img_size)
        model = TracedModel(model, device, img_size)

    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        # creation of a cap (cv2.VideoCapture('0')
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    old_img_w = old_img_h = imgsz
    old_img_b = 1

    # mediapipe
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_hands = mp.solutions.hands
    RightLeftHand = []
    errcnt = 0

    t0 = time.time()
    for gi, (path, img, im0s, vid_cap) in enumerate(dataset):
        if gi == 21: break
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Warmup
        if device.type != 'cpu' and (old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
            old_img_b = img.shape[0]
            old_img_h = img.shape[2]
            old_img_w = img.shape[3]
            for i in range(3):
                model(img, augment=augment)[0]
                # model(img, augment=opt.augment)[0]

        # Inference
        t1 = time_synchronized()
        pred = model(img, augment=augment)[0]
        # pred = model(img, augment=opt.augment)[0]
        t2 = time_synchronized()

        # Apply NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes=classes, agnostic=agnostic_nms)
        # pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t3 = time_synchronized()

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # img.jpg
            txt_path_bbox = str(save_dir / 'bbox' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
            txt_path_kpts = str(save_dir / 'kpts' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh

            # (modified)
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if conf >= inj_conf_thres:
                        c1, c2 = (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3]))
                        center_point = round((c1[0]+c2[0])/2), round((c1[1]+c2[1])/2)
                        # print(f"center_point: {center_point}")
                        circle = cv2.circle(im0,center_point,5,(0,255,0),2)
                        #text_coord = cv2.putText(im0,str(center_point),center_point, cv2.FONT_HERSHEY_PLAIN,3,(255,0,1), thickness=4)


                        #L = round(c2[0]-c1[0]), round(c2[1]-c1[1])
                        #circle_1 = cv2.circle(im0,center_point,5,(0,255,0),2)
                        #text_coord_1 = cv2.putText(im0,str(L),L, cv2.FONT_HERSHEY_PLAIN,2,(0,0,255))

                        #linex = cv2.line(im0,(0,center_point[1]),(640,center_point[1]),(255,0,0),3)
                        #liney = cv2.line(im0,(center_point[0],0),(center_point[0],640),(255,0,0),3)


                        #linez = cv2.line(im0,c1,c2,(255,0,0),3)
                        circle = cv2.circle(im0,c1,5,(0,255,0),2)
                        #text_coord = cv2.putText(im0,str(c1),c1, cv2.FONT_HERSHEY_PLAIN,3,(255,0,0))
                        circle = cv2.circle(im0,c2,5,(0,255,0),2)
                        #text_coord = cv2.putText(im0,str(c2),c2, cv2.FONT_HERSHEY_PLAIN,3,(255,0,0))

                        text_coord = cv2.putText(im0,str( c1  ),c1, cv2.FONT_HERSHEY_PLAIN,3,(255,0,0))


                        c3, c4 =( c2[0]  , c1[1] ), (c1[0] ,  c2[1] )
                        circle = cv2.circle(im0,c3,5,(0,255,0),2)
                        #text_coord = cv2.putText(im0,str( c3  ) , c3 ,  cv2.FONT_HERSHEY_PLAIN,3,(255,0,0), thickness=4)
                        circle = cv2.circle(im0,c4,5,(0,255,0),2)
                        #text_coord = cv2.putText(im0,str( c4  ),c4, cv2.FONT_HERSHEY_PLAIN,3,(255,0,0), thickness=4)


                        #m=(1,1)
                        #m1 =(7,7)
                        #m2=(640,640)
                        #text_coord = cv2.putText(im0,str(m),m, cv2.FONT_HERSHEY_PLAIN,3,(0,0,250))
                        #linef = cv2.line(im0,m,c2,(0,0,250),3)
                        #linek = cv2.line(im0,m,m2,(0,0,250),3)


                        #txt_path_bbox = "C:/Users/MSN/Desktop/main/runs/detect"
                        if gi % 2 == 0:
                            if save_txt:  # Write to file
                                """
                                    xyxy = bottom-left, top-right -> Unnormalized
                                    xywh = center (x, y), w h -> Normalized
                                    gn : normalized depeding on the image size
                                         [1920, 1080, 1920, 1080]
                                """
                                xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                                # print(f"xyxy: {xyxy}")  # [tensor(1001.), tensor(395.), tensor(1237.), tensor(494.)]
                                # print(f"xywh: {xywh}")  # [0.582812488079071, 0.4115740656852722, 0.12291666865348816, 0.09166666865348816]
                                line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                                # line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)  # label format
                                with open(txt_path_bbox + '.txt', 'a') as f:
                                    f.write(('%g ' * len(line)).rstrip() % line + '\n')

                            if save_img or view_img:  # Add bbox to image
                                label = f'{names[int(cls)]} {conf:.2f}'
                                # draw bounding boxes
                                plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=5)

            # # Print time (inference + NMS)
            # print(f'{s}Done. ({(1E3 * (t2 - t1)):.1f}ms) Inference, ({(1E3 * (t3 - t2)):.1f}ms) NMS')

            # articulations
            hands = mp_hands.Hands(model_complexity=0,
                                   min_detection_confidence=0.5,
                                   min_tracking_confidence=0.5)
            im0 = cv2.cvtColor(im0, cv2.COLOR_BGR2RGB)
            results = hands.process(im0)

            im0 = cv2.cvtColor(im0, cv2.COLOR_RGB2BGR)
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        im0,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style())
                    RightLeftHand.append(results.multi_handedness[0].classification[0].label)
            if gi % 2 == 0:
                if save_txt:
                    f = open(txt_path_kpts + '.txt', 'a')
                    try:
                        for idx in range(len(results.multi_hand_landmarks[0].landmark)):
                            kptx = results.multi_hand_landmarks[0].landmark[idx].x
                            kpty = results.multi_hand_landmarks[0].landmark[idx].y
                            kptz = results.multi_hand_landmarks[0].landmark[idx].z
                            line = (str(kptx), str(kpty), str(kptz))
                            f.write(('%s ' * len(line)).rstrip() % line + '\n')
                        f.close()
                    except:
                        print(f"Placez votre main de manière pertinente !\nVous avez {5 - errcnt} fois de tentative.")
                        if errcnt == 5:
                            import sys
                            print("\nExiting the program...")
                            sys.exit(0)
                        errcnt += 1

            # Stream results
            if view_img:
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if gi % 2 == 0:
                if save_img:
                    if dataset.mode == 'image':
                        cv2.imwrite(save_path, im0)
                        print(f" The image with the result is saved in: {save_path}")
                    else:  # 'video' or 'stream'
                        if vid_path != save_path:  # new video
                            vid_path = save_path
                            if isinstance(vid_writer, cv2.VideoWriter):
                                vid_writer.release()  # release previous video writer
                            if vid_cap:  # video
                                fps = vid_cap.get(cv2.CAP_PROP_FPS)
                                w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                                h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                            else:  # stream
                                fps, w, h = 30, im0.shape[1], im0.shape[0]
                                # save_path += '.mp4'
                                save_path_video = save_path + '.mp4'
                            vid_writer = cv2.VideoWriter(save_path_video, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                        save_path_image = save_path + '_' + str(gi) + '.jpg'
                        cv2.imwrite(save_path_image, im0)
                        vid_writer.write(im0)

    if save_txt or save_img:
        s1 = f"\n{len(list(save_dir.glob('bbox/*.txt')))} labels of bbox saved to {save_dir / 'bbox'}" if save_txt else ''
        s2 = f"\n{len(list(save_dir.glob('kpts/*.txt')))} kpts of the hands saved to {save_dir / 'kpts'}" if save_txt else ''
        print(f"Results saved to {save_dir}{s1}{s2}")

    print(f'Done. ({time.time() - t0:.3f}s)')

    return save_dir, RightLeftHand


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov7.pt', help='model.pt path(s)')
    # parser.add_argument('--source', type=str, default='inference/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--source', type=str, default='0', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    # parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    # parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    # parser.add_argument('--augment', action='store_true', help='augmented inference')
    # parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    # parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--no-trace', action='store_true', help='don`t trace model')
    
   
    opt = parser.parse_args()
    print(opt)
    #conf = parser.parse_args('--conf-thres')
    #print(conf)
    #check_requirements(exclude=('pycocotools', 'thop'))

    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['yolov7.pt']:
                detect()
                strip_optimizer(opt.weights)
        else:
            detect()
