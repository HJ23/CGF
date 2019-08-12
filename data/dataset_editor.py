import cv2
import os
import zipfile


pathtoimages="./face_dataset"
destination="./dataset"
pathtocascade="./"
counter=0

print(" Zip file extraction started ! ")
with zipfile.ZipFile("./faces.zip",'r') as facezip:
    facezip.extractall()

dirs=os.listdir(pathtoimages)
print(" Extracted !")


print(" Starting image editing !")
cascade=cv2.CascadeClassifier("./haar.xml")
for dir in dirs:
    insidedir=os.listdir(pathtoimages+"/"+dir)
    for image in insidedir:
        imagefile=cv2.imread(pathtoimages+"/"+dir+"/"+image)
        
        imagefile=cv2.cvtColor(imagefile,cv2.COLOR_BGR2GRAY)
        bounding_box=cascade.detectMultiScale(imagefile,1.1)
        # get just first rectangle coordinates one is enough
        rectangle_first=bounding_box[0]
        imagefile=imagefile[rectangle_first[0]:rectangle_first[0]+rectangle_first[2],rectangle_first[1]:rectangle_first[1]+rectangle_first[3]]
        imagefile=cv2.resize(imagefile,(64,64))
        cv2.imwrite(destination+"/"+str(counter)+".png",imagefile)
        counter+=1
        print("Image edited : "+image+" image number :"+str(counter))











