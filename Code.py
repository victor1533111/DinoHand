
import cv2
import numpy as np
from matplotlib.pyplot import imshow
import sys
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import pyautogui
import seaborn as sns

def neural_network():

    img_height = 50
    img_width = 50
    
    from keras.preprocessing.image import ImageDataGenerator
    
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
      "Traindata",
      validation_split=0.2,
      subset="training",
      seed=123,
      image_size=(img_height, img_width))
    
    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
      "Traindata",
      validation_split=0.2,
      subset="validation",
      seed=123,
      image_size=(img_height, img_width))
    
    import matplotlib.pyplot as plt
    
    class_names = train_ds.class_names
    
    plt.figure(figsize=(10, 10))
    for images, labels in train_ds.take(1):
      for i in range(len(class_names)):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(class_names[labels[i]])
        plt.axis("off")
        
    
    from tensorflow.keras import layers
    
    normalization_layer = tf.keras.layers.experimental.preprocessing.Rescaling(1./255)
    
    normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
    image_batch, labels_batch = next(iter(normalized_ds))
    first_image = image_batch[0]
    # Notice the pixels values are now in `[0,1]`.
    print(np.min(first_image), np.max(first_image))
    
    AUTOTUNE = tf.data.experimental.AUTOTUNE
    
    train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
    
    num_classes = 8
    
    model = tf.keras.Sequential([
      layers.experimental.preprocessing.Rescaling(1./255),
      layers.Conv2D(32, 3, activation='relu'),
      layers.MaxPooling2D(),
      layers.Conv2D(32, 3, activation='relu'),
      layers.MaxPooling2D(),
      layers.Conv2D(32, 3, activation='relu'),
      layers.MaxPooling2D(),
      layers.Flatten(),
      layers.Dense(128, activation='relu'),
      layers.Dense(num_classes)
    ])
    
    model.compile(
      optimizer='adam',
      loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
      metrics=['accuracy'])
    
    model.fit(
      train_ds,
      validation_data=val_ds,
      epochs=3
    )

    model.save('model.h5')
    
def test_red_neural():
    model = keras.models.load_model('model.h5')
    test = tf.keras.preprocessing.image_dataset_from_directory(
    "Test",
    seed=123,
    image_size=(50, 50))
    imgs = []
    labels = []
    y_pred=[]
    y_true=[]
    for img, label in test:
        y_pred.append( np.argmax(model.predict(img), axis=1).tolist())
        y_true.append( label.numpy().tolist())
    
    y_pred=[item for sublist in y_pred for item in sublist]
    y_true=[item for sublist in y_true for item in sublist]
    y_pred=np.array(y_pred)
    y_true=np.array(y_true)
    confusion_mtx = tf.math.confusion_matrix(y_true, y_pred) 
   
    plt.figure(figsize=(10, 8))
    sns.heatmap(confusion_mtx,  
            annot=True, fmt='g')
    plt.xlabel('Prediction')
    plt.ylabel('Label')
    plt.show()
    test_acc = sum(y_pred == y_true) / len(y_true)
    print(f'Test set accuracy: {test_acc:.0%}')
def predecir(model):
      img = tf.keras.preprocessing.image_dataset_from_directory(
      "manos",
      seed=123,
      image_size=(50, 50))
      array=model.predict_classes(img)
      return array[0]
    
def detectar_color_mano(frame,video):

    clicado=False
    fragmento=[]
    x=400
    y=70
    cv2.rectangle(frame, (x, y), (x+150, y+150), (255, 0, 0), 2) 
    frame=cv2.putText(frame,'Coloque la mano en el rectangulo y clique la tecla K',(10, 20), 1, 1.3,(0,255,255),1,cv2.LINE_AA)
    if cv2.waitKey(1) == ord('k'):
        ret, frame = video.read()
        frame = cv2.flip(frame, 1)
        fragmento= frame[70:220,400:550,:]
        clicado=True
    return clicado,fragmento
    
def color_mano(frame):
    frame1=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame2=cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    val = np.reshape(frame2[:,:,0], -1)
    val2= np.reshape(frame2[:,:,1], -1)
    val3 = np.reshape(frame2[:,:,2], -1)
    media1=np.mean(val)  
    media2=np.mean(val2)  
    media3=np.mean(val3)  
    desviacion1= np.std(val)*2 
    desviacion2= np.std(val2)*2     
    desviacion3= np.std(val3)*2
    if(media1+desviacion1>255):
        max1=255
    else :
        max1=media1+desviacion1
    if(media2+desviacion2>255):
        max2=255
    else:
        max2=media2+desviacion2
    if(media3+desviacion3>255):
        max3=255
    else:
        max3=media3+desviacion3
        
    rango1=[media1-desviacion1,max1]
    rango2=[media2-desviacion2,max2]
    rango3=[media3-desviacion3,max3]
    val = np.reshape(frame1[:,:,0], -1)
    val2= np.reshape(frame1[:,:,1], -1)
    val3 = np.reshape(frame1[:,:,2], -1)
    media1=np.mean(val)  
    media2=np.mean(val2)  
    media3=np.mean(val3)  
    desviacion1= np.std(val) *2
    desviacion2= np.std(val2) *2    
    desviacion3= np.std(val3)*2
    if(media1+desviacion1>255):
        max1=255
    else :
        max1=media1+desviacion1
    if(media2+desviacion2>255):
        max2=255
    else:
        max2=media2+desviacion2
    if(media3+desviacion3>255):
        max3=255
    else:
        max3=media3+desviacion3
        
    rango1.append(media1-desviacion1) 
    rango1.append(max1)
    rango2.append(media2-desviacion2)
    rango2.append(max2)
    rango3.append(media3-desviacion3)
    rango3.append(max3)
    return rango1,rango2,rango3

def segmentar_color2(img,rango1,rango2,rango3):
      
      img1=cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
      img=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
      Mascara1=cv2.inRange(img1, (rango1[0], rango2[0], rango3[0]), (rango1[1],rango2[1],rango3[1]))
      Mascara2=cv2.inRange(img, (rango1[2], rango2[2], rango3[2]), (rango1[3],rango2[3],rango3[3]))
      resultado=cv2.bitwise_not(Mascara1)
      resultado2=cv2.bitwise_not(Mascara2)
      a= cv2.bitwise_not(resultado, resultado2)  

      return a
      
  
def segmentar_color(img):

    #Convertimos de RGB a HSV
   
    img_HSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    #Definimos los umbrales de color HSV de la piel humana.
    HSV_mask = cv2.inRange(img_HSV, (0, 15, 0), (17,170,255)) 
    HSV_mask = cv2.morphologyEx(HSV_mask, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))

    #Convertimos de RGB a YCbCr
    img_YCrCb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    #Definimos los umbrales de color YCbCr de la piel humana.
    YCrCb_mask = cv2.inRange(img_YCrCb, (0, 135, 85), (255,180,135)) 
    YCrCb_mask = cv2.morphologyEx(YCrCb_mask, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))
    plt.imshow(HSV_mask,'gray')
    plt.show()
    plt.imshow(YCrCb_mask,'gray')
    plt.show()
    #Combinamos YCbCr y hsv
    global_mask=cv2.bitwise_and(YCrCb_mask,HSV_mask)
    global_mask=cv2.medianBlur(global_mask,3)
    global_mask = cv2.morphologyEx(global_mask, cv2.MORPH_OPEN, np.ones((4,4), np.uint8))


    HSV_result = cv2.bitwise_not(HSV_mask)
    YCrCb_result = cv2.bitwise_not(YCrCb_mask)
    image=cv2.bitwise_not(global_mask)-255
    return image

def Eliminar_cara(img,faceCascade):
    
    
    faces = faceCascade.detectMultiScale(img, 1.1, 4)
    #Dibujamos rectangulo con interior negro
    if len(faces)!=0: 
        (x, y, w, h) =  faces[0]
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 0), -1)
    return img

def test_face_remove():
 
    import glob
    caras_eliminadas=0
    i=0
    faceCascade = cv2.CascadeClassifier( 'face_detector.xml')
    for filename in glob.glob('BD/FACES_TEST/.jpg'):
        img=cv2.imread(filename)
        b_pixeles = np.sum(img == 0)
        img2=Eliminar_cara(img,faceCascade)
        name="img"+str(i)+".jpg"
        cv2.imwrite(os.path.join("BD/FACE_TEST_REMOVED" , name), img2)
        b_pixeles2 = np.sum(img2 == 0)
        if (b_pixeles2-b_pixeles)>100000:
            caras_eliminadas=caras_eliminadas+1
        i=i+1

    porcentaje=caras_eliminadas*100/500
    return porcentaje

porcentaje=test_face_remove()

porcentaje=test_face_remove()
def deteccion_de_mano(img,model):
    # Tratamiento de la imagen
   
    
    (cnt, _) = cv2.findContours(img.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    if cnt:
        mano = max(cnt, key=cv2.contourArea)
        x,y,w,h = cv2.boundingRect(mano)
        fragmento=img[y:y+h,x:x+w]
        cv2.imwrite(os.path.join("manos/img" , 'img.jpg'), fragmento)
        prediccion=predecir(model)
    else:
        prediccion=10
        fragmento=np.zeros([50,50])

    return prediccion,fragmento

  
def pruebas():

    img=cv2.imread('6.jpg')
    img2=cv2.imread('piel0.jpg')
    imgg=Eliminar_cara(img)
    rango1,rango2,rango3=color_mano(img2)
    imgg=segmentar_color2(imgg, rango1, rango2, rango3)
    img=deteccion_de_mano(imgg,img)



def main():
    faceCascade = cv2.CascadeClassifier( 'face_detector.xml')
    model = keras.models.load_model('model.h5')
    video = cv2.VideoCapture(0)
    conseguido=False
    i=0
    lista_gestos=dict()
    j=0

    while video.isOpened():
        start=time.time()
        #Cojemos los frames de la imagen
        ret, frame = video.read()
        #Giramos el video para generar efecto espejo
        frame = cv2.flip(frame,1)
        if conseguido==False:
            conseguido,imgmano=detectar_color_mano(frame,video)
            
            if len(imgmano)!=0:
                rango1,rango2,rango3=color_mano(imgmano)
                plt.imshow(imgmano)
                plt.show()
                time.sleep(1)
        else:
           

            start=time.time()
            imgg=Eliminar_cara(frame,faceCascade)
            start=time.time()
            imgg=segmentar_color2(imgg,rango1,rango2,rango3)
            start=time.time()
            gesto,manofinal=deteccion_de_mano(imgg,model)
           
            if gesto not in lista_gestos:
                lista_gestos[gesto]=[1]
            else:
                lista_gestos[gesto][0]+=1
            
            if j==3:
                gestomax=max(lista_gestos, key=lista_gestos.get)
                if lista_gestos[gestomax][0]==3:
                  
                    if gestomax == 6 or gestomax == 1 :
                        pyautogui.press('down')
                        cv2.putText(frame,"AJUPIR", (200, 400),  1, 5,(0,255,255),1,cv2.LINE_AA)
                       
                    elif gestomax == 0 or gestomax == 7 :
                        pyautogui.press('up')
                        cv2.putText(frame,"SALTAR", (200, 400), 1, 5,(0,255,255),1,cv2.LINE_AA)
                       
                    elif gestomax == 5 or gestomax == 2 :
                        pyautogui.press('space')
                        cv2.putText(frame,"INICI", (200, 400), 1, 5,(0,255,255),1,cv2.LINE_AA)
                    elif gestomax == 3 or gestomax == 4 :
                        pyautogui.keyDown('alt')
                        pyautogui.press('f4')
                        pyautogui.keyUp('alt')
                        cv2.putText(frame,"TANCAR", (200, 400),  1, 5,(0,255,255),1,cv2.LINE_AA)
                  
                        
                lista_gestos=dict()
                j=0
            j=j+1
           
            
        if ret:
            
            cv2.imshow("Capturing",frame)
           
            key = cv2.waitKey(1)   
           


        #Al clicar la tecla x se detiene la ejecucion
        if cv2.waitKey(1) == ord('x'):
            break

    #Liberamos los recursos utilizados
    video.release()
    cv2.destroyAllWindows()
    
def pruebas():

    img=cv2.imread('6.jpg')
    img2=cv2.imread('piel0.jpg')
    plt.imshow(img)
    plt.show()
    plt.imshow(img2)
    plt.show()
    imgg=Eliminar_cara(img)
    rango1,rango2,rango3=color_mano(img2)
    imgg=segmentar_color(imgg)
    plt.imshow(imgg,'gray')
    plt.show()
    img=deteccion_de_mano(imgg,img)
    
def crear_carpeta(nombre):
    faceCascade = cv2.CascadeClassifier( 'face_detector.xml')
    model = keras.models.load_model('model.h5')
    if not os.path.exists("Test/"+nombre):
                os.makedirs("Test/"+nombre)
    video = cv2.VideoCapture(0)
    conseguido=False
    i=0

    while video.isOpened():
        #Cojemos los frames de la imagen
        ret, frame = video.read()
        #Giramos el video para generar efecto espejo
        frame = cv2.flip(frame,1)
        if conseguido==False:
            conseguido,imgmano=detectar_color_mano(frame,video)
            
            if len(imgmano)!=0:
                rango1,rango2,rango3=color_mano(imgmano)
                plt.imshow(imgmano)
                plt.show()
        else:
           
            
            imgg=Eliminar_cara(frame,faceCascade)
            imgg=segmentar_color2(imgg,rango1,rango2,rango3)
            (cnt, _) = cv2.findContours(imgg.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            mano = max(cnt, key=cv2.contourArea)
            x,y,w,h = cv2.boundingRect(mano)
            fragmento=imgg[y:y+h,x:x+w]
            cv2.imwrite(os.path.join("Test/"+nombre , str(i) + '.jpg'), fragmento)
            i+=1
            print(i)
        if ret:
            
            cv2.imshow("Capturing",frame)
            key = cv2.waitKey(1)   


test_red_neural()