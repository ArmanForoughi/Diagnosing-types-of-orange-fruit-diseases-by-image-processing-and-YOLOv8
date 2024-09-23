import cv2
import numpy as np
import matplotlib.pyplot as plt
import smtplib
import os
import time
from ultralytics import YOLO
model = YOLO("yolo_orange.pt")

start_time = 14
email_receiver = "inter your email"
email_sender = "inter your email"
password_email = "inter your password two stagefor email"

path1 = os.getcwd()+"/pic_camera"
os.chdir(path1)

def remove_noise(gray, num):
    Y, X = gray.shape
    nearest_neigbours = [[
        np.argmax(
            np.bincount(
                gray[max(i - num, 0):min(i + num, Y),
                    max(j - num, 0):min(j + num, X)].ravel()))
        for j in range(X)] for i in range(Y)]
    result = np.array(nearest_neigbours, dtype=np.uint8)
    return result

def send_mail_function(msg):
    recipientEmail = email_receiver
    recipientEmail = recipientEmail.lower()
    try:
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.ehlo()
        server.starttls()
        server.login(email_sender, password_email )
        server.sendmail(email_sender, recipientEmail, msg)
        print("sent to {}".format(recipientEmail))
        server.close()
    except Exception as e:
    	print(e)

while True:
    if time.localtime().tm_hour == start_time:
        list_img = os.listdir()
        for path in list_img:
            results = model.predict(path, conf=0.6)
            img = cv2.imread(path)
            
            for i, result in enumerate(results):
                boxes = result.boxes
                if boxes.cls.size()[0] != 0:
                    for i ,box in enumerate(boxes):
                        x1 = int(box.xyxy[0][0])
                        x2 = int(box.xyxy[0][2])
                        y1 = int(box.xyxy[0][1])
                        y2 = int(box.xyxy[0][3])
                        img_box = img[y1:y2 , x1:x2]
                        img_box = cv2.resize(img_box,(150,150))
                        blur = cv2.GaussianBlur(img_box, (21, 21), 0)
                        hsl = cv2.cvtColor(img_box, cv2.COLOR_BGR2HLS)

                        lower_orenge_color1 = np.array([0,100,110])
                        lower_orenge_color2 = np.array([35,200,255])
                        mask1 = cv2.inRange(hsl, lower_orenge_color1, lower_orenge_color2)

                        upper_orenge_color1 = np.array([220,100,80])
                        upper_orenge_color2 = np.array([255,200,255])
                        mask2 = cv2.inRange(hsl, upper_orenge_color1, upper_orenge_color2)

                        mask_orenge_color = mask1+mask2

                        mask_remove_noise = remove_noise(mask_orenge_color, 10)

                        output_hsl = cv2.bitwise_and(hsl, hsl, mask=mask_remove_noise)
                        output_img = cv2.bitwise_and(hsl, hsl, mask=mask_remove_noise)

                        mask_1 = cv2.inRange(output_hsl, lower_orenge_color1, lower_orenge_color2)
                        mask_2 = cv2.inRange(output_hsl, upper_orenge_color1, upper_orenge_color2)

                        mask_out = mask_1 + mask_2

                        lower_canker = [2, 50, 30]
                        upper_canker = [41, 180, 100]
                        lower_blackspot = [0, 0, 0]
                        upper_blackspot = [140, 90, 140]
                        lower_melanuses = [5, 60, 50]
                        upper_melanuses = [20, 180, 130]

                        lower_canker = np.array(lower_canker, dtype="uint8")
                        upper_canker = np.array(upper_canker, dtype="uint8")
                        lower_blackspot = np.array(lower_blackspot, dtype="uint8")
                        upper_blackspot = np.array(upper_blackspot, dtype="uint8")
                        lower_melanuses = np.array(lower_melanuses, dtype="uint8")
                        upper_melanuses = np.array(upper_melanuses, dtype="uint8")

                        mask_canker = cv2.inRange(hsl, lower_canker, upper_canker)
                        mask_blackspot = cv2.inRange(img_box, lower_blackspot, upper_blackspot)
                        mask_melanuses = cv2.inRange(hsl, lower_melanuses, upper_melanuses)

                        count_canker = cv2.countNonZero(mask_canker)
                        count_blackspot = cv2.countNonZero(mask_blackspot)
                        count_melanuses = cv2.countNonZero(mask_melanuses)

                        print(cv2.countNonZero(mask_orenge_color))
                        print(cv2.countNonZero(mask_out))
                        if not (cv2.countNonZero(mask_orenge_color) - cv2.countNonZero(mask_out)) > 650:
                            print(f"{path} _ {i} healthy")
                            cv2.imshow(f"img_{i}",img_box)
                        else:
                            mask=[]
                            if count_canker > count_blackspot and count_canker > count_melanuses:
                                msg=f"{path} _ {i} is spoiled. is Canker disease is a skin and bacterial disease. The disease causes the branches to dry. To prevent this white paint be applied evenly to the main trunk of the tree. To treat this disease When 70% of the leaves have fallen, Bordofix poison is used after the formation of petals and fruits. In addition, to eliminate the disease with a knife, the affected part was pruned in winter in dry weather"
                                #send_mail_function(msg)
                                print(msg)
                                mask=mask_canker
                            elif count_blackspot > count_canker and count_blackspot > count_melanuses:
                                msg=f"{path} _ {i} is spoiled. is black spot. Treatments with guazatine or imazalil decrease the viability of the pathogen in black spot lesions. Fungicides such as strobilurins, dithiocarbamates and benzimidazoles are also efficient against the fungus, but resistances have also developed in many areas"
                                #send_mail_function(msg)
                                print(msg)
                                mask=mask_blackspot
                            elif count_melanuses > count_canker and count_melanuses > count_blackspot:
                                msg=f"{path} _ {i} is spoiled. is melanuses. Protectant copper sprays are the only product registered for melanose control. timing of spray applications is very important. With Washington navel and Valencia oranges the spray should be applied at full petal fall."
                                #send_mail_function(msg)
                                print(msg)
                                mask=mask_melanuses
                            out = cv2.bitwise_and(hsl,hsl,mask)
                            cv2.imshow(f"Hsl_{i}",hsl)
                            cv2.imshow(f"Mask_{i}",out)
                else:
                    print(f"{path} _ {i} not orange")
        cv2.waitKey(0)
        cv2.destroyAllWindows()    
        # os.remove(path)
    time.sleep(60*60)


    
