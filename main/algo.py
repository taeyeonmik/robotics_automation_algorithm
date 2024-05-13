import os
import glob
import argparse
import torch

from detect import detect
from injection import Injection
from serial.tools import list_ports
from dobotcode.dobot import Dobot


def main(save_dir, RightLeftHand):
    """ Etape.2
        Après la détection, on a plusieurs résultats.
        Dans cette étape, le programme choisira le meilleur
        et on va travailler sur ce qu'il a choisi.
    """
    # Etape.2-1 : Récupérer les sources d'image, de bbox, et de key points
    Imgs = sorted(glob.glob(str(save_dir) + '/' + '*.jpg'))
    Injs = sorted(glob.glob(str(save_dir) + '/bbox/*.txt'))
    Kpts = sorted(glob.glob(str(save_dir) + '/kpts/*.txt'))
    RightLeftHand = max(RightLeftHand, key=RightLeftHand.count)
    print(f"max RightLeftHand : {RightLeftHand}")

    # Etape.2-2 : Comparer le nombre de résultats
    """
        Pour votre information,
        Il y a l'écart du nombre de résultats
        entre la détection de plaie et celle d'articulation
        parce que les vitesses des modèles (YOLO et MIDEAPIPE) sont différentes. 
        La détection du plaie alors ne se faite pas à chaque frame
        à cause de la vitesse du modèle.
    """
    if len(Injs) != len(Kpts):
        nbmin = min(len(Injs), len(Kpts))
        rmv1, rmv2 = [], []
        list4loop = []
        if nbmin == len(Injs):
            files = [os.path.basename(file) for file in Injs]
            list4loop = Kpts
        elif nbmin == len(Kpts):
            files = [os.path.basename(file) for file in Kpts]
            list4loop = Injs
        for file in list4loop:
            filename = os.path.basename(file)
            filename2, _ = os.path.splitext(filename)
            if filename not in files:
                rmv1.append(file)
                rmv2.append(filename2)
        for r1, r2 in zip(rmv1, rmv2):
            # remove in the list
            list4loop.remove(r1)
            Imgs.remove(str(save_dir) + '/' + r2 + '.jpg')
            # remove in the directory
            os.remove(r1)
            os.remove(str(save_dir) + '/' + r2 + '.jpg')

    # Etape.2-3 : Selectionner l'un qui a la confiance la plus haute
    # Etape.2-3 (1) Garder les résultats qui ont leur confiance qui est supérieure à threshold
    thres = opt.inj_conf_thres
    Injs = sorted(glob.glob(str(save_dir) + '/bbox/*.txt'))
    for i, inj in enumerate(Injs):
        # read
        f = open(inj, 'r')
        lines = f.readlines()
        # rewrite
        rewrite = []
        for line in lines:
            elm = line.split(' ')
            conf = float(elm[-1])
            if conf >= thres:
                rewrite.append(line)
        f.close()
        f = open(inj, 'w')
        for rw in rewrite:
            f.write(rw)
        f.close()

    # Etape.2-3 (2) Faire la liste des résultats qui ont le plus nombre de plaies
    nbs, maxInjs = [], []
    for i in range(len(Injs)):
        f = open(Injs[i], 'r')
        nb = len(f.readlines())
        nbs.append(nb)
    maxinj = max(nbs)
    for i, n in enumerate(nbs):
        if n == maxinj:
            maxInjs.append(Injs[i])
        else: pass
    print(f"maxInjs : {maxInjs}")

    # Etape.2-3 (3) Comparer les confiances entre chaque résultat et prendre l'un qui a la meilleure confiance
    maxi = None
    maxj, maxconf = None, 0.0
    for i, inj in enumerate(maxInjs):
        f = open(inj, 'r')
        lines = f.readlines()
        print(f"\n{inj} : {lines}")
        for j, line in enumerate(lines):
            elm = line.split(' ')
            if len(elm) > 5:
                conf = float(elm[-1])
                print(f"conf: {conf}")
                if conf >= maxconf:
                    maxi = i
                    maxj = j
                    maxconf = conf
            else: continue
    assert maxi != None, "Le plaie non claire. Essayer une fois de plus."
    print(f"\nmaxconf info: {maxInjs[maxi]}")
    print(f"maxconf: {maxconf}")

    # Etape.2-3 (4) Définir un résultat(image) de détection sur lequel on travaille
    inj_path = os.path.dirname(maxInjs[maxi])
    kpt_path = os.path.dirname(maxInjs[maxi].replace("bbox", "kpts"))
    workon = os.path.basename(maxInjs[maxi])
    # le nombre de plaies
    f = open(inj_path + "/" + workon, 'r')
    nbinj = len(f.readlines()) - 1
    f.close()

    # Etape.2-3 (5) Ne faire rester qu'un seul résultat qu'on a choisi
    for filepath in Injs:
        if os.path.basename(filepath) == workon:
            pass
        else:
            os.remove(filepath)
    for filepath in Kpts:
        if os.path.basename(filepath) == workon:
            pass
        else:
            os.remove(filepath)
    for filepath in Imgs:
        if os.path.basename(filepath) == (os.path.splitext(workon)[0] + ".jpg"):
            pass
        else:
            os.remove(filepath)

    """ Etape.3
        Classification
        - right-front | left-back | right-back | left-front
        - CoatNet : l'état de l'art de la classification d'image
    """
    autocls = opt.auto_classification
    if autocls:
        # (deep learning model à l'avenir)
        rf = RightLeftHand
    else:
        rf = opt.right_left # right-left

    """ Etape.4
        Trouver les meiilleurs points d'injection
        en employant la classe : injection.Injection
    """
    # Etape.4-1 : L'instanciation de model d'injection
    injectionModel = Injection(inj_path, kpt_path, workon, nbinj, rf)

    # Etape.4-2 : Trouver les meilleurs point à injecter
    # injections = injectionModel.FindInjectionPoints()
    injectionModel.FindInjectionPoints()

    # Etape.4-3 : Visuliser le résultat et le sauvegarder en un fichier
    injectionModel.VisualizeInjectionPoint(save=True)
    print(f"Nous avons trouvé {len(injectionModel.injections)} point(s) d'injection")

    # Etape.4-4 : Dénormaliser les valeurs des coordonnés par rapport à la résolution d'image d'origine
    dinjections = DenormRatio(injectionModel, denormRatio=opt.denormRatio) # denormRatio = 6.87 : cette valeur peut varier dépendant du caméra

    """ Etape.5
        Envoyer les coordonnées des injections au robot.
        Et puis, le robot commence les injections sur la main.
    """
    injectionDobot(injectionModel, dinjections)
    print("Les injections finissent.\nQuitter le programme ...")

def DenormRatio(injectionModel, denormRatio):
    dinjections = []
    for idx, ij in enumerate(injectionModel.injections):
        denormX = ij[0] * injectionModel.w / denormRatio  # width of image
        denormY = ij[1] * injectionModel.h / denormRatio  # height of image
        dinjections.append((denormX, denormY))
    print(f"Coordonnées dénormalisées d'injection : {dinjections}")
    return dinjections

def injectionDobot(injectionModel, dinjections):
    print(f"Nous avons trouvé {len(dinjections)} point(s) d'injection")
    for i, injp in enumerate(dinjections):
        if i == 0:
            print(f"{i + 1}ère injection : {injp}")
        else:
            print(f"{i + 1}ème injection : {injp}")
    print("vont être transmises au robot.")

    # get device
    device, depart = InitPoseRobot()

    tryInjection = 0
    while True:
        if tryInjection == len(dinjections):
            # situer le robot à une place de départ
            device.move_to(*depart, wait=True)
            print(f'x:{depart[0]} y:{depart[1]} z:{depart[2]} r:{depart[3]}')
            break
        yRobot = dinjections[tryInjection][0] # y de Robot
        xRobot = dinjections[tryInjection][1] # x de Robot

        # déplacer le robot au-dessus de la plaie
        if not injectionModel.handTypeCheck:
            ajustement = opt.Rajustement
        else:
            ajustement = opt.Lajustement
        x, y, z, r = ajustement[0] + xRobot, \
                     ajustement[1] + yRobot, \
                     30.0, \
                     0.0
        device.move_to(x, y, z, r, wait=True)
        print(f'x:{x} y:{y} z:{z} r:{r}')

        # injéction
        x, y, z, r = ajustement[0] + xRobot, \
                     ajustement[1] + yRobot, \
                     -10.0, \
                     0.0
        device.move_to(x, y, z, r, wait=True)
        print(f'x:{x} y:{y} z:{z} r:{r}')
        print("En train d'injecter ...")
        device.wait(1000) # en train d'injécter

        # déplacer le robot au-dessus de la plaie
        x, y, z, r = ajustement[0] + xRobot, \
                     ajustement[1] + yRobot, \
                     30.0, \
                     0.0
        device.move_to(x, y, z, r, wait=True)
        print(f'x:{x} y:{y} z:{z} r:{r}')

        tryInjection += 1

    device.wait(1000)
    device.close() # Robot closed

def InitPoseRobot():
    available_ports = list_ports.comports()
    available_ports.sort()
    print(f'available ports: {[x.device for x in available_ports]}')
    port = available_ports[1].device

    # instantiation de Dobot avec un port donné
    device = Dobot(port, verbose=False)
    device.pose()

    # init pose
    depart = (145.0, 0.0, 173.0, 0.0) # x, y, z, r
    device.move_to(*depart, wait=True)

    return device, depart

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Detection
    parser.add_argument('--weights', nargs='+', type=str, default='yolov7_custom_2.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='0', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.5, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--no-trace', action='store_true', help='don`t trace model')
    # Injection
    # * l'explication d'ajustement et de donormRatio s'est faite en détail sur la documentation.
    parser.add_argument('--auto-classification', action='store_true', help='classification du type de main')
    parser.add_argument('--right-left', type=tuple, default=None, help='indication du type de main')
    parser.add_argument('--inj-conf-thres', type=float, default=0.10, help='object confidence threshold for injection')
    parser.add_argument('--Rajustement', type=tuple, default=(125.0, -143.5), help='a cause d une différence d une mesure des valeurs entre Yolo et le robot, il faut calculer les valeurs pour le ajustement')
    parser.add_argument('--Lajustement', type=tuple, default=(125.0, -136.1), help='a cause d une différence d une mesure des valeurs entre Yolo et le robot, il faut calculer les valeurs pour le ajustement')
    parser.add_argument('--denormRatio', type=float, default=1920/(143.5+136.1), help='pour fournir au robot les valeurs qui sont appropriées au système du robot, il faut calculer un ratio')

    opt = parser.parse_args()
    print(opt)
    """
        --weights yolov7_custom_2.pt --conf 0.5 --img-size 640 --source 0 --view-img --no-trace --save-txt --save-conf --auto-classification
    """
    source, weights, view_img, save_txt, imgsz, trace = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size, not opt.no_trace
    save_img = not opt.nosave and not source.endswith('.txt')  # save inference images
    save_conf = opt.save_conf
    classes = opt.classes
    project = opt.project
    name = opt.name
    exist_ok = opt.exist_ok
    device = opt.device
    img_size = opt.img_size
    conf_thres = opt.conf_thres
    iou_thres = opt.iou_thres
    agnostic_nms = not opt.agnostic_nms
    augment = not opt.augment
    update = not opt.update
    inj_conf_thres = opt.inj_conf_thres

    # Init Robot
    InitPoseRobot()

    """ Etape.1
        Détections des plaies et des articulations en utilisant
        - YOLO
        - MEDIAPIPE
    """
    # Détection
    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['yolo.pt']:
                save_dir, RightLeftHand = detect(source, weights, view_img, save_conf, save_txt, classes, imgsz, trace,
                                  project, name, exist_ok, device, img_size, conf_thres, iou_thres, inj_conf_thres,
                                  agnostic_nms, augment, save_img)  # runs/detect/expN
                from utils.general import strip_optimizer
                strip_optimizer(opt.weights)
        else:
            save_dir, RightLeftHand = detect(source, weights, view_img, save_conf, save_txt, classes, imgsz, trace,
                              project, name, exist_ok, device, img_size, conf_thres, iou_thres, inj_conf_thres,
                              agnostic_nms, augment, save_img)  # runs/detect/expN
    print(RightLeftHand)

    main(save_dir, RightLeftHand)