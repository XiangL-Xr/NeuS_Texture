import os
import requests
from glob import glob
import cv2

url = "http://10.8.1.0:20123/mat"

def get_mask(img_list, save_path):

    for i, img_path in enumerate(img_list):
        myobj = {'photo': open(img_path, "rb").read()}
        res = requests.post(url, files = myobj)
        save_name = os.path.join(save_path, '{:03d}.png'.format(i))

        with open(save_name, 'wb') as fout:
            fout.write(res.content)

        img = cv2.imread(save_name, cv2.IMREAD_UNCHANGED)
        img = cv2.cvtColor(img[...,-1], cv2.COLOR_GRAY2RGB)
        #cv2.imwrite(save_name, img[...,-1])
        cv2.imwrite(save_name, img)
        #cv2.imwrite(save_name, img[:, :, ::-1])