# import the OpenCV package and six
from __future__ import print_function
import cv2, sys, six, os, os.path, numpy as np

Hr = "000" #must be in double quotes (i.e. "000")
ChipType = "G2" #must be in double quotes (i.e. "G1")

# Raw image directory path and folder to save isolated images path
imageDir = "/Users/seankelly/Desktop/2018.03.03_G2_Half_Test_1" #specify your path here (no spaces allowed)
imageDirSave = "/Users/seankelly/Desktop/Python/YellenLab/ImageCropTestG2" #specify your path here (no spaces allowed)

############################### EDITORS ONLY #################################
# Debug info OpenCV version
print ("OpenCV version: " + cv2.__version__)

# Test whether specified file path exists for imageDir and imageDirSave
existTrip = 0  #Gets flipped to 1 if can't find path to folder containing raw images
existTrip2 = 0 #Gets flipped to 1 if can't find path to intended save folder for isolate images
def dirExists(filename, source):
    if(os.path.exists(filename)):
        #Testing if raw image and intended save folders exist
        return
    else:
        if source == "raw":
            print("The folder containing raw images does not exist, please try again")
            dummyTrip = 1
            return dummyTrip
        elif source == "save":
            print("The folder to save isolated images does not exist, please try again")
            dummyTrip = 1
            return dummyTrip
        else:
            print("The specified folder type must be 'raw' or 'save', please try again")
            dummyTrip = 1
            return dummyTrip

existTrip = dirExists(imageDir, "raw")
existTrip2 = dirExists(imageDirSave, "save")
if existTrip == 1 or existTrip2 == 1:
    sys.exit()

possibleChipTypes = ["G1", "G2", "G3"] #list of all possible chip types
if (ChipType in possibleChipTypes) == 0: #testing if chip type is allowed
    print("Invalid chip type provided, please try again")
    sys.exit()

image_path_list = []
valid_image_extensions = [ ".tif" ] #specify your vald extensions here
valid_image_extensions = [item.lower() for item in valid_image_extensions]

#Create a list all files in directory and append files with a valid extention to image_path_list
for file in os.listdir(imageDir):
    extension = os.path.splitext(file)[1]
    if extension.lower() not in valid_image_extensions:
        continue
    image_path_list.append(os.path.join(imageDir, file))

#Loop through image_path_list to open each image
for imagePath in image_path_list:
    img = cv2.imread(imagePath)
    height, width, channels = img.shape
    #Display the image on screen with imshow() after checking that it loaded
    if img is not None and height == 1200 and width == 1600:
    	imageName = imagePath[-31:]
    	AptName = imageName[18:21] # Range within name of apt number
        AptInt00 = int(AptName)
        StName = imageName[10:13] # Range within name of st number
        StInt00 = int(StName)

        #Insert chip name as if statement and copy parameters to add a chip
        if ChipType == "G1":
            XStart = 1350 #X center of bottom right apartment
            YStart = 720 #Y center of bottom right apartment
            XPeriod = 320 #horizontal apartment period (integer; in pixels)
            YPeriod = 320 #vertical apartment period (integer; in pixels)
            XBuffFactor = 1.2 #multiplicative factor of XPeriod for isolated image size
            YBuffFactor = 1.2 #multiplicative factor of XPeriod for isolated image size
            NumApts = 4 #number of apartments in each raw image (integer)
            NumSts = 2 #number of streets in each raw image (integer)
            TotalApts = 31 #total number of apartments in a row (integer; largest in chip)
            TotalSts = 46 #total number of streets in chip

        if ChipType == "G2":
            XStart = 1288
            YStart = 580
            XPeriod = 317
            YPeriod = 346
            XBuffFactor = 0.9
            YBuffFactor = 1.4
            NumApts = 4
            NumSts = 2
            TotalApts = 31
            TotalSts = 41

        if ChipType == "G3":
            XStart = 1278
            YStart = 750
            XPeriod = 415
            YPeriod = 314
            XBuffFactor = 1
            YBuffFactor = 1.2
            NumApts = 3
            NumSts = 2
            TotalApts = 15
            TotalSts = 57

        #Setting horizontal and vertical image size (in pixels)
        XBuffer = int((XBuffFactor*XPeriod)//2)
        YBuffer = int((YBuffFactor*XPeriod)//2)

        #Looping through each apt and st, cropping, naming, and saving image
        for k in range(0, NumApts):
            for j in range(0, NumSts):
                SaveTrigger = 0 #if 0, image will save; if triggered to 1, will not save image
                if j % 2 == 1: #testing if street is odd or even
                    if ChipType == "G1" or ChipType == "G3":
                        BumpOver = 0.5 #shift over to account for street offset
                    elif ChipType == "G2":
                        BumpOver = -0.5
                else:
                    BumpOver = 0
                #Cropping images, assigning st and apt number, and changing SaveTrigg
                crop_img = img[int(YStart - j*YPeriod - YBuffer):int(YStart - j*YPeriod + YBuffer),
                               int(XStart - (BumpOver+k)*XPeriod - XBuffer):int(XStart - (BumpOver+k)*XPeriod + XBuffer)]
                StNum = str(StInt00 + 1000 + j)[1:4]
                AptNum = str(AptInt00 + 1000 + k)[1:4]
                if int(StNum) > TotalSts or int(AptNum) > TotalApts:
                    SaveTrigger = 1
                if (j % 2 == 1) and (int(AptNum) > (TotalApts - 1)) and (ChipType == "G1" or ChipType == "G3"):
                    SaveTrigger = 1

                #Saving cropped, isolated images
                if SaveTrigger == 0:
                    imageName = "BF_ST_" + StNum + "_APT_" + AptNum + "_Hr_" + Hr + ".jpg"
                    dirName = imageDirSave + "/" + imageName
                    cv2.imwrite(dirName, crop_img)
                    print(imageName)

    elif img is None:
        print ("Error loading: " + imagePath)
        #End this loop iteration and move on to next image
        continue
    #wait time in milliseconds
    #this is required to show the image
    #0 = wait indefinitely
    #exit when escape key is pressed
        key = cv2.waitKey(0)
        if key == 27: #escape
            break
#Close any open windows
cv2.destroyAllWindows()
