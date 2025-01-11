import cv2
from cv2 import dnn
import numpy as np


class ConvertingBlackAndWhiteImagesToColoredImages:
    """
    This module converts black and white images to colored images
    initalizing the model files and parameters
    """
    def __init__(self):
        ############## model file paths ################
        """
        For the model files contact me at arun.arunisto2@gmail.com or +91 8137856143
        """
        self.proto_file = "colorization_deploy_v2.prototxt"
        self.model_file = "colorization_release_v2.caffemodel"
        self.hull_pts = "pts_in_hull.npy"
        #-----------------------------------------------#
        ############ readibng the model params ##########
        self.net = dnn.readNetFromCaffe(self.proto_file, self.model_file)
        self.kernel = np.load(self.hull_pts)
        #-----------------------------------------------#
    
    def colorize(self, image: np.ndarray) -> np.ndarray:
        """
        This function converts black and white images to colored images
        args: image (numpy.ndarray) - black and white image
        """
        # preprocessing the image
        if image.ndim == 2:
            image = np.stack((image,)*3, axis=-1)
        scaled = image.astype("float32") / 255.0
        lab_img = cv2.cvtColor(scaled, cv2.COLOR_BGR2LAB)
        #-----------------------------------------------#
        # adding the cluster centers as 1x1 convolutions to the model
        class8 = self.net.getLayerId("class8_ab")
        conv8 = self.net.getLayerId("conv8_313_rh")
        pts = self.kernel.transpose().reshape(2, 313, 1, 1)
        self.net.getLayer(class8).blobs = [pts.astype("float32")]
        self.net.getLayer(conv8).blobs = [np.full([1, 313], 2.606, dtype="float32")]
        #-----------------------------------------------#
        # resizing the image for the network
        resized = cv2.resize(lab_img, (224, 224))
        #-----------------------------------------------#
        # splitting the L channels
        L = cv2.split(resized)[0]
        #-----------------------------------------------#
        # subtracting the mean
        L -= 50
        #-----------------------------------------------#
        # predicting the ab channels from the input L channel
        self.net.setInput(cv2.dnn.blobFromImage(L))
        ab_channel = self.net.forward()[0, :, :, :].transpose((1, 2, 0))
        #-----------------------------------------------#
        # resizing the predictedd 'ab' volume to the same dimensions as our input image
        ab_channel = cv2.resize(ab_channel, (image.shape[1], image.shape[0]))
        #-----------------------------------------------#
        # Taking the L channel from the image
        L = cv2.split(lab_img)[0]
        #-----------------------------------------------#
        # joining the L channel with predicted ab channel
        colorized = np.concatenate((L[:, :, np.newaxis], ab_channel), axis=2)
        #-----------------------------------------------#
        # Converting the image from LAB to BGR
        colorized = cv2.cvtColor(colorized, cv2.COLOR_LAB2BGR)
        colorized = np.clip(colorized, 0, 1)
        #-----------------------------------------------#
        # changing the image to 0-255 range and convert it from float32 to int
        colorized = (255*colorized).astype("uint8")
        #-----------------------------------------------#
        #resizing the image to the original size
        colorized = cv2.resize(colorized, (image.shape[1], image.shape[0]))
        return colorized
        #-----------------------------------------------#

if __name__ == "__main__":
    bnw2color = ConvertingBlackAndWhiteImagesToColoredImages()
    img = cv2.imread("tesla.jpeg")
    colorized = bnw2color.colorize(img)
    result = cv2.hconcat([img, colorized])
    cv2.imshow("Result", result)
    cv2.waitKey(0)
