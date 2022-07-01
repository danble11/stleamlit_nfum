import numpy as np
import streamlit as st
import cv2

face_cascade = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')
smile_cascade = cv2.CascadeClassifier('./haarcascade_smile.xml')


st.title('笑顔')
st.write('笑顔には様々な効果があることが知られている。')


#写真撮影
img_file_buffer = st.camera_input("笑顔を撮影してください")


#写真撮影後の処理
if img_file_buffer is not None:
    #o
    bytes_data = img_file_buffer.getvalue()
    cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(cv2_img,cv2.COLOR_BGR2GRAY)
    #顔があるかどうか
    faces = face_cascade.detectMultiScale(gray,1.1,5)
    for (x,y,w,h) in faces:
        cv2.rectangle(cv2_img,(x,y),(x+w,y+h),(255, 0, 0),2) # blue
        #Gray画像から，顔領域を切り出す．
        roi_gray = gray[y:y+h, x:x+w] 

        #サイズを縮小
        roi_gray = cv2.resize(roi_gray,(100,100))

        #輝度で規格化
        lmin = roi_gray.min() #輝度の最小値
        lmax = roi_gray.max() #輝度の最大値
        for index1, item1 in enumerate(roi_gray):
            for index2, item2 in enumerate(item1) :
                roi_gray[index1][index2] = int((item2 - lmin)/(lmax-lmin) * item2)
        
        #笑顔識別
        st.write(roi_gray)
        smiles= smile_cascade.detectMultiScale(roi_gray,scaleFactor= 1.1, minNeighbors=0, minSize=(50, 50))
        st.write(smiles)
        if len(smiles) >0 :
            smile_neighbors = len(smiles)
            #笑顔の数値を表示
            st.write("smile_neighbors=",smile_neighbors)
            LV = 2/100
            intensityZeroOne = smile_neighbors  * LV 
            if intensityZeroOne > 1.0: 
                intensityZeroOne = 1.0 
            
            st.write(intensityZeroOne)
            for(sx,sy,sw,sh) in smiles:
                cv2.circle(cv2_img,(int(x+(sx+sw/2)*w/100),int(y+(sy+sh/2)*h/100)),int(sw/2*w/100), (255*(1.0-intensityZeroOne), 0, 255*intensityZeroOne),2)#red
    

    
    st.write(cv2_img.shape)
    cv2_img = cv2.cvtColor(cv2_img,cv2.COLOR_BGR2RGB)

    st.image(cv2_img)
