from flask import Flask, request
import pytesseract
from PIL import Image
import easyocr
import matplotlib.pyplot as plt
import cv2
import re
import numpy as np 

app = Flask(__name__)

def noise_removal(image):
    kernel = np.ones((1, 1), np.uint8)
    image = cv2.dilate(image, kernel, iterations=1)
    kernel = np.ones((1, 1), np.uint8)
    image = cv2.erode(image, kernel, iterations=1)
    image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    image = cv2.medianBlur(image, 3)
    return (image)


def thin_font(image):
    image = cv2.bitwise_not(image)
    kernel = np.ones((2,2),np.uint8)
    image = cv2.erode(image, kernel, iterations=1)
    image = cv2.bitwise_not(image)
    return (image)

def thick_font(image):
    image = cv2.bitwise_not(image)
    kernel = np.ones((2,2),np.uint8)
    image = cv2.dilate(image, kernel, iterations=1)
    image = cv2.bitwise_not(image)
    return (image)

def remove_borders(image):
    contours, heiarchy = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cntsSorted = sorted(contours, key=lambda x:cv2.contourArea(x))
    cnt = cntsSorted[-1]
    x, y, w, h = cv2.boundingRect(cnt)
    crop = image[y:y+h, x:x+w]
    return (crop)


def grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def recognize_text(img_path):
    '''loads an image and recognizes text.'''
    
    reader = easyocr.Reader(['pt'])
    text =  reader.readtext(img_path)

    most_probable_result = max(text, key=lambda x: x[2])
    most_probable_text = most_probable_result[1]

    return {
        'text': text,
        'most_problable_text': most_probable_text
    }


def sanitize_text(result_list):
    texts = [result[1] for result in result_list]
    text_elements = [text for text in texts if re.search('[a-zA-Z]', text)]
    return text_elements

def overlay_ocr_text(img_path, save_name):
    '''loads an image, recognizes text, and overlays the text on the image.'''
    
    # loads image
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    dpi = 80
    fig_width, fig_height = int(img.shape[0]/dpi), int(img.shape[1]/dpi)
    plt.figure()
    f, axarr = plt.subplots(1,2, figsize=(fig_width, fig_height)) 
    axarr[0].imshow(img)
    
    # recognize text
    result = recognize_text(img_path)
    fullText = ''
     
    # if OCR prob is over 0.5, overlay bounding box and text
    for (bbox, text, prob) in result['text']:
        if prob >= 0.55:
            # display 
            print(f'Detected text: {text} (Probability: {prob:.2f})')
            fullText += text + ' '

            # get top-left and bottom-right bbox vertices
            (top_left, top_right, bottom_right, bottom_left) = bbox
            top_left = (int(top_left[0]), int(top_left[1]))
            bottom_right = (int(bottom_right[0]), int(bottom_right[1]))

            # create a rectangle for bbox display
            cv2.rectangle(img=img, pt1=top_left, pt2=bottom_right, color=(255, 0, 0), thickness=10)

            # put recognized text
            cv2.putText(img=img, text=text, org=(top_left[0], top_left[1] - 10), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 0, 0), thickness=8)
        
    # show and save image
    axarr[1].imshow(img)
    plt.savefig(f'{save_name}_overlay.jpg', bbox_inches='tight')
    return fullText


@app.route('/extract-text', methods=['POST'])
def extract_text():
    teste = cv2.imread('test.png')
    img = cv2.imwrite('image.png', teste)
    im_1_path = 'image.png'
    im_1_overlay_path = 'im_1_overlay.jpg'
    img_1 = cv2.imread(im_1_path)
    img_1 = cv2.cvtColor(img_1, cv2.COLOR_BGR2RGB)
    fullText = overlay_ocr_text(im_1_path, 'im_1')

    return {
        'resultOverlay': fullText
    }

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
