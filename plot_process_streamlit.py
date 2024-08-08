#!/usr/bin/env python
# coding: utf-8

import cv2
import streamlit as st
import numpy as np
import pytesseract
from matplotlib import pyplot as plt
import pandas as pd
import datetime as dt
from io import BytesIO
from PIL import Image


def process_line_plot_image(image, mode='datetime'):


    ### Read image and generate grayscale and binary images
    ##image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    binary_wo_txt = binary.copy()

    ### Use OCR to generate a list of strings in the image
    custom_config = r'--psm 6 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz'
    strings = pytesseract.image_to_string(gray, config=custom_config)
    txt_list = strings.split('\n')
    boxes = pytesseract.image_to_boxes(gray,config=custom_config)
    box_list = boxes.strip().split('\n')
    while '' in txt_list: txt_list.remove('')
    for txt in txt_list:
        if len(txt) > 6:
            txt_n = txt.split(' ')
            txt_list.remove(txt)
            txt_list.extend(txt_n)



    ### removing stray text and determinning axis locations
    y_ax_xpts= []
    y_ax_ypts= []
    x_ax_ypts= []
    x_ax_xpts= []
    im_ht = np.shape(binary)[0]-1
    im_ln = np.shape(binary)[0]-1
    padding = 2
    for box in box_list:
        ch,x1b,y1b,x2b,y2b,_ = box.split(' ')
        binary_wo_txt[im_ht - int(y2b)-padding:im_ht-int(y1b)+padding,int(x1b)-padding:int(x2b)+padding] = 0
        #detecting axes
        if (max(int(x1b),int(x2b)) > 0.5*np.shape(binary)[1]) & (max(int(y1b),int(y2b)) < 0.5*np.shape(binary)[0]):
            if (abs(int(x1b)-int(x2b)) < 0.05*im_ln) & (abs(int(y1b)-int(y2b)) < 0.05*im_ht):
                x_ax_ypts.extend([int(y1b),int(y2b)])
                x_ax_xpts.extend([int(x1b),int(x2b)])
                #print('x ax:',ch, 'x: ',x1b,x2b, 'y: ',y1b,y2b)
        if (max(int(x1b),int(x2b)) < 0.5*np.shape(binary)[1]) & (max(int(y1b),int(y2b)) > 0.5*np.shape(binary)[0]):
            if (abs(int(x1b)-int(x2b)) < 0.05*im_ln) & (abs(int(y1b)-int(y2b)) < 0.05*im_ht):
                y_ax_xpts.extend([int(x1b),int(x2b)])
                y_ax_ypts.extend([int(y1b),int(y2b)])
                #print('y ax: ',ch,'y: ',y1b,y2b, 'x: ', x1b,x2b)

    ### removing axes
    binary_wo_txt[:,0:np.array(y_ax_xpts).max()] = 0
    binary_wo_txt[im_ht-np.array(x_ax_ypts).max():-1,:] = 0


    pos= np.argmax(binary_wo_txt,axis=0)
    pos_flip = im_ht - pos
    x_loc = pos_flip < im_ht
    data = pos_flip[x_loc]
    x_shift = np.argmax(x_loc)



    ### Calulating boundaries of each string in txt_list

    counter = 0
    txt_x1 = []
    txt_x2 = []
    txt_y1 = []
    txt_y2 = []
    for txt in txt_list:
        
        c_counter=0
        for c in txt:
            if c == box_list[counter].split(' ')[0]:
                if c_counter == 0:
                    txt_x1.append(box_list[counter].split(' ')[1]) 
                    txt_y1.append(box_list[counter].split(' ')[2])
                counter+=1
                #print(txt,c, counter, c_counter)
                c_counter+=1
        txt_x2.append(box_list[counter-1].split(' ')[3]) 
        txt_y2.append(box_list[counter-1].split(' ')[4])


    ### Classifying strings into x-tick labels and y-tick labels
    count_y = 1
    count_x = 1
    real_y, real_x = [],[]
    img_y, img_x = [],[]
    for i,txt in enumerate(txt_list):
        #print('i: ', i ,'txt: ', txt, 'txt_y1[i]: ', txt_y1[i])
        if int(txt_y1[i]) in y_ax_ypts:
            if count_y <= 3:
                
                real_y.append(int(txt))
                img_y.append(round(0.5*(int(txt_y1[i]) + int(txt_y2[i]))))
                print(txt, ': detected in y axis at pixel (row) ', round(0.5*(int(txt_y1[i]) + int(txt_y2[i]))) )
                count_y+=1
        if int(txt_x1[i]) in x_ax_xpts:
            if count_x <= 3:
                if mode == 'datetime':
                    real_x.append(pd.to_datetime(txt+ ' '+str(dt.date.today().year)))
                    img_x.append(round(0.5*(int(txt_x1[i]) + int(txt_x2[i]))))
                    print(txt, ': detected in x axis at pixel (col) ', round(0.5*(int(txt_x1[i]) + int(txt_x2[i]))) )
                    count_x+=1
                else: 
                    if can_convert_to_int(txt):
                        real_x.append(int(txt))
                        img_x.append(round(0.5*(int(txt_x1[i]) + int(txt_x2[i]))))
                        print(txt, ': detected in x axis at pixel (col) ', round(0.5*(int(txt_x1[i]) + int(txt_x2[i]))) )
                        count_x+=1
    real_y=np.array(real_y)
    img_y=np.array(img_y)


    ### Show plot with detected x-tick labels
    gray_tmp = gray.copy()
    gray_tmp[:,img_x[0]] = 0
    gray_tmp[:,img_x[1]] = 0
    gray_tmp[:,x_shift] = 0
    plt.figure(figsize=(20,5))
    plt.imshow(gray_tmp)
    if mode == 'datetime':
        plt.text(img_x[0]-50,150,'Detected: '+str(real_x[0].date()))
        plt.text(img_x[1]-50,50,'Detected: '+str(real_x[1].date()))
        plt.text(x_shift-50,50,'Detected: data begins')
    else:
        plt.text(img_x[0]-50,150,'Detected: '+ str(real_x[0]))
        plt.text(img_x[1]-50,50,'Detected: '+ str(real_x[1]))
        plt.text(x_shift-50,50,'Detected: data begins')
    plt.show()


    x_ax_dummy = np.arange(0,len(data))

    if len(real_y) > 2:
        y_compress1 = (real_y[2] - real_y[1])/(img_y[2]-img_y[1])
        y_compress2 = (real_y[1] - real_y[0])/(img_y[1]-img_y[0])
        y_compress = 0.5*(y_compress1+y_compress2)
    else:
        y_compress = (real_y[1] - real_y[0])/(img_y[1]-img_y[0])

    if len(real_y) > 2:
        c1 = real_y[0] - y_compress*img_y[0]
        c2 = real_y[1] - y_compress*img_y[1]
        c3 = real_y[2] - y_compress*img_y[2]
        c = (c1+c2+c3)/3
    else:
        c1 = real_y[0] - y_compress*img_y[0]
        c2 = real_y[1] - y_compress*img_y[1]
        c = 0.5*(c1+c2)

    data_shift = data*y_compress + c


    if mode == 'datetime':
        days_p_pt1 = (real_x[2] - real_x[1]).days/(img_x[2]-img_x[1])
        days_p_pt2 = (real_x[1] - real_x[0]).days/(img_x[1]-img_x[0])
        days_p_pt = 0.5*(days_p_pt2 + days_p_pt1)
        start_date1 = real_x[1] - pd.Timedelta(days=days_p_pt* (img_x[1]-x_shift))
        start_date2 = real_x[2] - pd.Timedelta(days=days_p_pt* (img_x[2]-x_shift))
        start_date3 = real_x[0] - pd.Timedelta(days=days_p_pt* (img_x[0]-x_shift))
        start_date = pd.to_datetime((start_date1.timestamp() + start_date2.timestamp() +start_date3.timestamp())/3,unit='s')
        img_date_idx =start_date + pd.to_timedelta(x_ax_dummy*days_p_pt, unit='D')
        df_tmp = pd.DataFrame(data_shift, index=img_date_idx, columns=['A'])
        df = df_tmp.resample('D').median().interpolate()


    if mode == 'normal':
        x_compress1 = (real_x[2] - real_x[1])/(img_x[2]-img_x[1])
        x_compress2 = (real_x[1] - real_x[0])/(img_x[1]-img_x[0])
        x_compress = 0.5*(x_compress1+x_compress2)
        start_x1 = real_x[1] - x_compress*(img_x[1]-x_shift)
        start_x2 = real_x[2] - x_compress*(img_x[2]-x_shift)
        start_x3= real_x[0] -  x_compress*(img_x[0]-x_shift)
        start_x = (start_x1+start_x2+start_x3)/3
        x_ax = start_x + x_ax_dummy*x_compress




    if mode == 'normal':
        df = pd.DataFrame(data_shift, index=x_ax, columns=['Data'])


    plt.figure(figsize=(14,5))
    plt.plot(df)
    plt.show()
   
    return df

def can_convert_to_int(s):
    if s.startswith('-'):
        return s[1:].isdigit()
    return s.isdigit()


st.title("digitize your line plot image to data")

mode = st.selectbox("Select mode", ['datetime', 'normal'])
uploaded_file = st.file_uploader("Choose an image...", type="png")


if uploaded_file is not None:
    image = Image.open(uploaded_file)
    image = np.array(image)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    
    st.write("")
    st.write("Processing...")
    
    df = process_line_plot_image(image, mode)
    
    st.write("Output DataFrame:")
    st.write(df)

    st.line_chart(df)

#if __name__ == "__main__":
#    st.run()

