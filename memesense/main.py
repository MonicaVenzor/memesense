# Import

import cv2
import pytesseract

from memesense.preprocess import preprocess_image
from memesense.params import *


def extract_text(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, thresh1 = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV) #revision de ret con rect (variable) mas abajo
    rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (18, 18))
    # Applying dilation on the threshold image
    dilation = cv2.dilate(thresh1, rect_kernel, iterations = 1)
    # Finding contours
    contours, _ = cv2.findContours(dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    # Creating a copy of image
    im2 = image.copy()

    #Variable to save the text of the meme

    texto = ""

    # Looping through the identified contours
    # Then rectangular part is cropped and passed on
    # to pytesseract for extracting text from it
    # Extracted text is then written into the text file
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)

    # Drawing a rectangle on copied image - para que?
        rect = cv2.rectangle(im2, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Cropping the text block for giving input to OCR
        cropped = im2[y:y + h, x:x + w]

    # Open the file in append mode
    #file = open("recognized.txt", "a")

    # Apply OCR on the cropped image
        text = pytesseract.image_to_string(cropped)

    # Appending the text into file
    #file.write(text)
    #file.write("\n")
        texto = texto + " " + text

    return texto
