import cv2
'''
def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                for i in range(3):                
                    name = './dados_treino/'+[i]+'.jpg'
                    image = cv2.imread(name)
                    angle = float(batch_sample[3])
                    if abs(angle) >= 0:
                        correction = 0.2 
                        if i == 1:
                            angle = angle + correction
                        if i == 2:
                            angle = angle - correction
                        images.append(image)
                        angles.append(angle)
                        image_flipped = np.fliplr(image)
                        angle_flipped = -angle
                        images.append(image_flipped)
                        angles.append(angle_flipped)

            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)

#https://scikit-learn.org/stable/modules/cross_validation.html
import sklearn

from sklearn.model_selection import train_test_split

exemplos = []
with open('./gravacoes/video2fr_teste.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)

train_samples, validation_samples = train_test_split(samples, test_size=0.2)

train_generator = generator(train_samples, batch_size=32)

'''
import cv2 
import numpy as np

filename = 'gravacoes/video2fr/270.jpg'
img = cv2.imread(filename)
#gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
gray = cv2.imread(img, cv2.IMREAD_GRAYSCALE)

cv2.imshow('original', gray)



#edge detection kernel (combines vertical and horizontal lines)
kernel = np.array([[1, 0, -1],
                   [1, 0, -1],
                   [1, 0, -1]])

filtered = cv2.filter2D(src=image, kernel=kernel, ddepth=-1)
cv2.imshow('horizontal edges', filtered)

# vertical edge detector
kernel = np.array([[1,  1,  1],
                   [0,  0,  0],
                   [-1, -1, -1]])
filtered = cv2.filter2D(src=image, kernel=kernel, ddepth=-1)
cv2.imshow('vertical edges', filtered)

# blurring ("box blur", because it's a box of ones)
kernel = np.array([[1, 1, 1],
                   [1, 1, 1],
                   [1, 1, 1]]) / 9.0
filtered = cv2.filter2D(src=image, kernel=kernel, ddepth=-1)
cv2.imshow('blurred', filtered)

# sharpening
kernel = (np.array([[-1, -1, -1],
                    [-1,  9, -1],
                    [-1, -1, -1]]))
filtered = cv2.filter2D(src=image, kernel=kernel, ddepth=-1)
cv2.imshow('sharpened', filtered)

# wait and quit
cv2.waitKey(0)
cv2.destroyAllWindows()