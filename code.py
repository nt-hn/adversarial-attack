import numpy as np
import math
import torch
import torch.nn as nn
from torch.nn import DataParallel
import torchvision.utils as vutils
from torch.autograd import Variable
from models import *
from PIL import Image, ImageDraw, ImageFilter
import cv2
from imutils import face_utils
import imutils
import dlib

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("--- Loading dlib model ...")
predic = "shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predic)
print("--- Done")

print("--- Loading resnet model ...")
def load_model(model, model_path):
    model_dict = model.state_dict()
    pretrained_dict = torch.load(model_path, map_location = device)
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

netClassifier = resnet_face18(False)
netClassifier = DataParallel(netClassifier)
load_model(netClassifier, './checkpoints/resnet18_110.pth')
netClassifier.load_state_dict(torch.load('./checkpoints/resnet18_110.pth', map_location = device))
netClassifier.to(device)
print("--- Done")

def find__left_cheek(orig_image, orig_image_size):

    ratio = orig_image_size / 500
    image = orig_image.clone()
    image = F.interpolate(image, size=500).cpu().numpy()[0]

    r = image[0]
    g = image[1]
    b = image[2]
    rgb = []

    for i in range(500):
        tmp = []

        for j in range(500):
            tmp.append([r[i][j], g[i][j], b[i][j]])
        rgb.append(tmp)

    rgb = np.ndarray.round(np.array(rgb)*255).astype(np.uint8)
    gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 1)
 
    target = [ 'mouth' , 'nose', 'left_eye']

    mouth_loc =(0,0)
    left_eye_loc=(0,0)
    nose_loc = (0,0)
    patch_center = (0,0)

    for (i, rect) in enumerate(rects):
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        for (name, (i, j)) in face_utils.FACIAL_LANDMARKS_IDXS.items():
            clone = rgb.copy()

            if name in target:
                temp = (0,0)
                point_num = len(shape[i:j])

            for (x, y) in shape[i:j]:               
                temp = (temp[0]+ x, temp[1]+y)
                cv2.circle(clone, (x, y), 1, (0, 0, 255), -1)

                if name is 'mouth':
                    mouth_loc = (int(temp[0]/point_num), int(temp[1]/point_num))

                elif name is 'left_eye':
                    left_eye_loc = (int(temp[0]/point_num) , int(temp[1]/point_num))

                elif name is 'nose':
                    nose_loc = (int(temp[0]/point_num) , int(temp[1]/point_num))

            else:
                pass

            x_dif = nose_loc[0] - left_eye_loc[0]
            y_dif = nose_loc[1] - left_eye_loc[1]
            patch_center = (round((left_eye_loc[0]-x_dif*0.4)*ratio) , round((nose_loc[1]-int(y_dif*0.1))*ratio))

    return patch_center, round(y_dif*0.6*ratio)

def find_right_cheek(orig_image, orig_image_size):

    ratio = orig_image_size / 500
    image = orig_image.clone()
    image = F.interpolate(image, size=500).cpu().numpy()[0]

    r = image[0]
    g = image[1]
    b = image[2]
    rgb = []

    for i in range(500):
        tmp = []
        for j in range(500):
            tmp.append([r[i][j], g[i][j], b[i][j]])
        rgb.append(tmp)

    rgb = np.ndarray.round(np.array(rgb)*255).astype(np.uint8)
    gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 1)
 
    target = [ 'mouth' , 'right_eye' , 'nose' ]

    mouth_loc =(0,0)
    right_eye_loc = (0,0)
    nose_loc = (0,0)
    patch_center = (0,0)

    for (i, rect) in enumerate(rects):
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        for (name, (i, j)) in face_utils.FACIAL_LANDMARKS_IDXS.items():
            clone = rgb.copy()

            if name in target:
                temp = (0,0)
                point_num = len(shape[i:j])

            for (x, y) in shape[i:j]:
                
                temp = (temp[0]+ x, temp[1]+y)                
                cv2.circle(clone, (x, y), 1, (0, 0, 255), -1)
              
                if name is 'mouth':
                    mouth_loc = (int(temp[0]/point_num), int(temp[1]/point_num))

                elif name is 'right_eye':
                    right_eye_loc = (int(temp[0]/point_num) , int(temp[1]/point_num))

                elif name is 'nose':
                    nose_loc = (int(temp[0]/point_num) , int(temp[1]/point_num))

            else:
                pass

            x2_dif= nose_loc[0] - right_eye_loc[0]
            y2_dif= nose_loc[1] - right_eye_loc[1]
            patch_center = (round((right_eye_loc[0]-x2_dif*0.4)*ratio) , round((nose_loc[1]-int(y2_dif*0.1))*ratio))

    return patch_center, round(y2_dif*0.6*ratio)

def cosin_metric(x1, x2):

    d1 = x1.clone().detach().numpy()
    d2 = x2.clone().detach().numpy()
    d1 = np.reshape(d1, 1024)
    d2 = np.reshape(d2, 1024)

    return np.array([np.dot(d1, d2) / (np.linalg.norm(d1) * np.linalg.norm(d2))])


def load_image(img_path):

    image = cv2.imread(img_path, 1)
    if image is None:
        return None
    image = image.transpose((2, 0, 1))
    image = image[np.newaxis, :, :, :]
    image = image.astype(np.float32, copy=False)
    image -= 127.5
    image /= 127.5
    image = torch.from_numpy(image).float()

    return image


def load_image_for_face(img_path):

    image = cv2.imread(img_path, 0)
    if image is None:
        print('--------face image error----------')
        return None
    image = np.dstack((image, np.fliplr(image)))
    image = image.transpose((2, 0, 1))
    image = image[:, np.newaxis, :, :]
    image = image.astype(np.float32, copy=False)
    image -= 127.5
    image /= 127.5

    return image

def load_image_for_patch(img_path):

    image = img_path
    if image is None:
        print('-------patch image error--------')
        return None
    image = np.dstack((image, np.fliplr(image)))
    image = image.transpose((2, 0, 1))
    image = image[:, np.newaxis, :, :]
    image = image.astype(np.float32, copy=False)
    image -= 127.5
    image /= 127.5

    return image

def resize_patch(patch_img):

    print('-----------Resizing Image----------------')
    img = cv2.imread(patch_img , 0)
    if img is None:
        print('Not working')
        return None
     
    print('Original Dimensions : ',img.shape)
     
    width = 36
    height = 36
    dim = (width, height)
     
    resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    
    print('Resized Dimensions : ',resized.shape)
    print('-------------Resizing DONE!--------------')
    cv2.imwrite('resized.jpg',resized)


    return resized




image_size = 128
patch_size = 0.05
learning_rate = 0.1
face_file = "sharon-pittaway-iMdsjoiftZo-unsplash.jpg"  #mask image
patch_tobe = "omid-armin-xOjzehJ49Hk-unsplash.jpg" #face image

def resize_background(background_img):

    print('-----------Resizing background----------------')
    img = cv2.imread(background_img , 0)
    if img is None:
        print('Not working')
        return None
     
    print('Original Dimensions : ',img.shape)
     
    width = 128
    height = 128
    dim = (width, height)
     
    resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    
    print('Resized Dimensions : ',resized.shape)
    print('-------------Resizing DONE!--------------')
    cv2.imwrite('background.jpg',resized)


    return resized

def attach_to_background(img,x,y,x1,y1): #attach to the (x,y) right and (x1,y1) left position

    print('-----attaching to blank page------')
    im1 = Image.open('background.jpg')
    im2 = Image.open(img)
    back_im = im1.copy()
    back_im.paste(im2,(x-18,y-18))
    back_im.paste(im2,(x1-18,y1-18))
    back_im.save('patch+background.jpg', quality=95)
    print('-----attaching done---------')

    return back_im

def main():
    netClassifier.eval()

    
    face_image = load_image(face_file)
    left_patch_loc, left_radius = find__left_cheek(face_image, image_size) #left
    right_patch_loc, right_radius = find_right_cheek(face_image, image_size) #right
    resize_patch(patch_tobe)
    resize_background('white.jpg')

    (cx, cy) = right_patch_loc
    x = cx
    y = cy
    (cx,cy) = left_patch_loc
    x1 = cx
    y1 = cy

    attach_to_background('resized.jpg',x,y,x1,y1)
    image = Image.open('patch+background.jpg')
    patch_file = np.asarray(image)

    face_shape_in_zeroes = np.zeros(face_image.shape)
 
    
    print('-------ADDITIONAL INFORMATION-----------')
    print('the patch file shape is ' + str(patch_file.shape))
    print('the layout shape is ' + str(face_shape_in_zeroes.shape))
    print('left patch location is' + str(left_patch_loc))
    print('right patch location is ' + str(right_patch_loc))
    print('left radius is ' + str(left_radius))
    print('right radius is ' + str(right_radius))
    print('---------STARTING SEQUENCE--------------')
 
    (cx, cy) = left_patch_loc # The center of patch on the left side


    for i in range(cy-left_radius, cy+left_radius, 1):
        for j in range(cx-left_radius,cx+left_radius, 1):
            if math.sqrt(((i-cy)**2)+((j-cx)**2)) <= left_radius:
                
                face_shape_in_zeroes[0][0][i][j] = 1
                face_shape_in_zeroes[0][1][i][j] = 1
                face_shape_in_zeroes[0][2][i][j] = 1
            else:
                pass


    (cx1, cy1) = right_patch_loc # The center of patch on the right

    for i in range(cy1-right_radius, cy1+right_radius, 1):
        for j in range(cx1-right_radius,cx1+right_radius, 1):
            if math.sqrt(((i-cy1)**2)+((j-cx1)**2)) <= right_radius:
                
                face_shape_in_zeroes[0][0][i][j] = 1
                face_shape_in_zeroes[0][1][i][j] = 1
                face_shape_in_zeroes[0][2][i][j] = 1
            else:
                pass



    face_shape_in_zeroes = torch.FloatTensor(face_shape_in_zeroes)
    if device.type == 'cuda':
        face_shape_in_zeroes = face_shape_in_zeroes.cuda() 
    face_shape_in_zeroes = Variable(face_shape_in_zeroes)

    

    face_only = load_image_for_face(face_file)
    patch = load_image_for_patch(patch_file)
    face_only = torch.from_numpy(face_only).to(device)
    patch = torch.from_numpy(patch).to(device)

    

    print("face_only and patch shapes: ", face_only.shape, patch.shape)

    tmp = face_shape_in_zeroes.clone()
    mask = torch.stack([tmp[:,1,:,:],tmp[:,1,:,:]], dim = 0)

    print("new mask: ", mask.shape)
    print("patch: ", patch.shape)


    face_patch = torch.mul((1-mask), face_only) + torch.mul(mask, patch)



    vutils.save_image(face_patch.clone().data, "ori_adv_x.png")

    face_only_vec = netClassifier(face_only).cpu()



    count = 0
    cosin_sim = 1
    while cosin_sim > 0.55:
        count += 1

        face_patch = Variable(face_patch.data, requires_grad=True)
        face_patch_vec = netClassifier(face_patch).cpu()

        sim_loss = F.cosine_similarity(face_patch_vec.view(1, 1024), face_only_vec.view(1, 1024))

        Loss = sim_loss
        print( count, "sim_loss: {:.4f}\t".format(sim_loss.data.item()))


        Loss.backward()
        face_patch_grad = face_patch.grad.clone()
        
        face_patch.grad.data.zero_()

        patch -= learning_rate * face_patch_grad


        face_patch = torch.mul((1-mask), face_only) + torch.mul(mask, patch)

        face_patch_vec = face_patch_vec.detach()
        face_only_vec = face_only_vec.detach()

        cosin_sim = cosin_metric(face_only_vec, face_patch_vec)


        if(count%100 == 0 or count==1):
            tmp = face_patch.clone()
            vutils.save_image(tmp.data, "logs/{}.png".format(count))

    vutils.save_image(face_patch.data, "adv_final_"+str(count)+".png")


if __name__ == '__main__':
    main()
