#!/usr/bin/env python
# coding: utf-8



import cv2
import numpy as np
import pytesseract
from matplotlib import pyplot as plt
import pandas as pd
import datetime as dt
from sklearn.cluster import KMeans

from scipy import stats
import streamlit as st
from io import BytesIO
from PIL import Image


def dominant_colors(image, num_colors=7):

    #pixels = image.reshape(-1, 3)
    pixels = image.view(image.dtype).reshape(-1, 3)
    # Use KMeans to cluster pixels
    kmeans = KMeans(n_clusters=num_colors)
    kmeans.fit(pixels)

    # Get the RGB values of the cluster centers
    dominant_colors = kmeans.cluster_centers_

    # Convert to integer values
    dominant_colors = dominant_colors.round(0).astype(int)

    return dominant_colors


# Function to convert RGB to hex
def rgb_to_hex(rgb):
    return '#{:02x}{:02x}{:02x}'.format(rgb[0], rgb[1], rgb[2])


def plot_colors(colors):
    # Create a pie chart to display the dominant colors
    plt.figure(figsize=(8, 6))
    plt.pie([1]*len(colors), colors=np.array(colors)/255, labels=colors, startangle=90)
    plt.axis('equal')
    plt.show()


# In[5]:


def filter_color(image, target_rgb, tolerance=15):

    #image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB

    # Define the lower and upper bounds for the target color with tolerance
    lower_bound = np.array([max(0, c - tolerance) for c in target_rgb])
    upper_bound = np.array([min(255, c + tolerance) for c in target_rgb])

    # Create a mask that only includes the target color within the tolerance range
    mask = cv2.inRange(image, lower_bound, upper_bound)

    # Apply the mask to keep only the target color
    filtered_image = cv2.bitwise_and(image,image, mask=mask)
    filtered_gray = cv2.cvtColor(filtered_image, cv2.COLOR_BGR2GRAY)
    _, filtered_binary = cv2.threshold(filtered_gray,50,255,cv2.THRESH_BINARY)

    return filtered_binary


# In[6]:


def extract_txt(binary, custom_config):
    strings = pytesseract.image_to_string(binary, config=custom_config)
    txt_list = strings.split('\n')
    boxes = pytesseract.image_to_boxes(binary,config=custom_config)
    box_list_tmp = boxes.strip().split('\n')
    while '' in txt_list: txt_list.remove('')
    while '~' in txt_list: txt_list.remove('')
    box_list = []
    for box in box_list_tmp:
        if '~' not in box:
            box_list.append(box)
    max_ln =  max(len(s) for s in txt_list)
    for i, txt in enumerate(txt_list):
        if len(txt) > 10:
            txt_n = txt.split(' ')
            txt_list[i:i+1] = txt_n
    return txt_list,box_list


# In[7]:


def find_txt_bnds(txt_list,box_list):
    counter = 0
    txt_x1 = []
    txt_x2 = []
    txt_y1 = []
    txt_y2 = []
    for txt in txt_list:
        
        c_counter=0
        for c in txt:
            #print(txt,c, counter, c_counter,box_list[counter].split(' '))
            if c == box_list[counter].split(' ')[0]:
                if c_counter == 0:
                    txt_x1.append(box_list[counter].split(' ')[1]) 
                    txt_y1.append(box_list[counter].split(' ')[2])
                counter+=1
            
                c_counter+=1
        txt_x2.append(box_list[counter-1].split(' ')[3]) 
        txt_y2.append(box_list[counter-1].split(' ')[4])
    return txt_x1,txt_x2,txt_y1,txt_y2


# In[8]:


def detect_axes_chars(binary,box_list,padding=2):
    #y_ax_xpts= []
    y_ax_ypts= []
    #x_ax_ypts= []
    x_ax_xpts= []
    im_ht = np.shape(binary)[0]-1
    im_ln = np.shape(binary)[1]-1
    for box in box_list:
        ch,x1b,y1b,x2b,y2b,_ = box.split(' ')
        if (max(int(x1b),int(x2b)) > 0.5*np.shape(binary)[1]) & (max(int(y1b),int(y2b)) < 0.5*np.shape(binary)[0]):
             if (abs(int(x1b)-int(x2b)) < 0.1*im_ln) & (abs(int(y1b)-int(y2b)) < 0.1*im_ht):
                #x_ax_ypts.extend([int(y1b),int(y2b)])
                x_ax_xpts.extend([int(x1b),int(x2b)])

        if (max(int(x1b),int(x2b)) < 0.5*np.shape(binary)[1]) & (max(int(y1b),int(y2b)) > 0.5*np.shape(binary)[0]):
            if (abs(int(x1b)-int(x2b)) < 0.1*im_ln) & (abs(int(y1b)-int(y2b)) < 0.1*im_ht):
                #y_ax_xpts.extend([int(x1b),int(x2b)])
                y_ax_ypts.extend([int(y1b),int(y2b)])
               
    return y_ax_ypts,x_ax_xpts


# In[9]:


def detect_xaxes_chars(binary,box_list,padding=2):
    x_ax_xpts= []
    im_ht = np.shape(binary)[0]-1
    im_ln = np.shape(binary)[1]-1
    for box in box_list:
        ch,x1b,y1b,x2b,y2b,_ = box.split(' ')
        if (max(int(x1b),int(x2b)) > 0.5*np.shape(binary)[1]) & (max(int(y1b),int(y2b)) < 0.5*np.shape(binary)[0]):
             if (abs(int(x1b)-int(x2b)) < 0.1*im_ln) & (abs(int(y1b)-int(y2b)) < 0.1*im_ht):
                x_ax_xpts.extend([int(x1b),int(x2b)])
               
    return x_ax_xpts


# In[10]:


def detect_xaxes_chars2(binary,box_list,padding=2):
    x_ax_xpts= []
    im_ht = np.shape(binary)[0]-1
    im_ln = np.shape(binary)[1]-1
    for box in box_list:
        ch,x1b,y1b,x2b,y2b,_ = box.split(' ')
        if (max(int(x1b),int(x2b)) > 0.3*np.shape(binary)[1]) & (max(int(y1b),int(y2b)) < 0.5*np.shape(binary)[0]):
             if (abs(int(x1b)-int(x2b)) < 0.1*im_ln) & (abs(int(y1b)-int(y2b)) < 0.1*im_ht):
                x_ax_xpts.extend([int(x1b),int(x2b)])
               
    return x_ax_xpts


# In[11]:


def detect_yaxes_chars(binary,box_list,padding=2):
    y_ax_ypts= []

    im_ht = np.shape(binary)[0]-1
    im_ln = np.shape(binary)[1]-1
    for box in box_list:
        ch,x1b,y1b,x2b,y2b,_ = box.split(' ')
        if (max(int(x1b),int(x2b)) < 0.5*np.shape(binary)[1]) & (max(int(y1b),int(y2b)) > 0.5*np.shape(binary)[0]):
            if (abs(int(x1b)-int(x2b)) < 0.1*im_ln) & (abs(int(y1b)-int(y2b)) < 0.1*im_ht):
                y_ax_ypts.extend([int(y1b),int(y2b)])
               
    return y_ax_ypts


# In[12]:


def detect_yaxes_chars2(binary,box_list,padding=2):
    y_ax_ypts= []

    im_ht = np.shape(binary)[0]-1
    im_ln = np.shape(binary)[1]-1
    for box in box_list:
        ch,x1b,y1b,x2b,y2b,_ = box.split(' ')
        if (max(int(x1b),int(x2b)) < 0.5*np.shape(binary)[1]) & (max(int(y1b),int(y2b)) > 0.3*np.shape(binary)[0]):
            if (abs(int(x1b)-int(x2b)) < 0.1*im_ln) & (abs(int(y1b)-int(y2b)) < 0.1*im_ht):
                y_ax_ypts.extend([int(y1b),int(y2b)])
               
    return y_ax_ypts


# In[13]:


def get_raw_data(filt_bin):
    binary_final = filt_bin
    pos= np.argmax(binary_final,axis=0)
    im_ht = np.shape(filt_bin)[0]-1
    pos_flip = im_ht - pos
    x_loc = pos_flip < im_ht
    data = pos_flip[x_loc]
    x_shift = np.argmax(x_loc)
    return x_shift,data


# In[14]:


def is_float(s):
    try:
        float(s)
        return True
    except ValueError:
        print(s, 'is not a float')
        return False


# In[15]:


def detect_x_ax_ticks(txt_list,box_list,
                    txt_x1,txt_x2,x_ax_xpts,
                    real_x,img_x,count_x):
    for i,txt in enumerate(txt_list):
        if int(txt_x1[i]) in x_ax_xpts:
            if count_x < 3:
                if mode == 'datetime':
                    try:
                        txt_tmp = pd.to_datetime(txt+ ' '+str(dt.date.today().year))
                    except Exception as e: 
                        print(e)
                    else:
                        if txt_tmp not in real_x:
                            real_x.append(txt_tmp)
                            img_x.append(round(0.5*(int(txt_x1[i]) + int(txt_x2[i]))))
                            print(txt, ': detected in x axis at pixel (col) ', round(0.5*(int(txt_x1[i]) + int(txt_x2[i]))) )
                            count_x+=1
                 
                else: 
                    if is_float(txt):
                        
                        real_x.append(float(txt))
                        print(real_x, 'real x array')
                        img_x.append(round(0.5*(int(txt_x1[i]) + int(txt_x2[i]))))
                        print(txt, ': detected in x axis at pixel (col) ', round(0.5*(int(txt_x1[i]) + int(txt_x2[i]))) )
                        count_x+=1
    return real_x,img_x,count_x


# In[16]:


def detect_x_ax_ticks2(txt_list,box_list,
                    txt_x1,txt_x2,x_ax_xpts,
                    real_x,img_x,count_x):
    for i,txt in enumerate(txt_list):
        if int(txt_x1[i]) in x_ax_xpts:
            if count_x < 3:
                if mode == 'datetime':
                    try:
                        txt_tmp = pd.to_datetime(txt)
                    except Exception as e0:
                        
                        try:
                            txt_tmp = pd.to_datetime(txt+ ' '+str(dt.date.today().year))
                        except Exception as e1: 
                            try:
                                txt_tmp = pd.to_datetime(txt+ ' Jan1')
                            except Exception as e2:
                                print(e2) 
                        
                    else:
                        if txt_tmp not in real_x:
                            real_x.append(txt_tmp)
                            img_x.append(round(0.5*(int(txt_x1[i]) + int(txt_x2[i]))))
                            print(txt, ': detected in x axis at pixel (col) ', round(0.5*(int(txt_x1[i]) + int(txt_x2[i]))) )
                            count_x+=1
                 
                else: 
                    if is_float(txt):
                        
                        real_x.append(float(txt))
                        print(real_x, 'real x array')
                        img_x.append(round(0.5*(int(txt_x1[i]) + int(txt_x2[i]))))
                        print(txt, ': detected in x axis at pixel (col) ', round(0.5*(int(txt_x1[i]) + int(txt_x2[i]))) )
                        count_x+=1
    return real_x,img_x,count_x


# In[17]:


def detect_x_ax_ticks3(txt_list,box_list,
                    txt_x1,txt_x2,x_ax_xpts,
                    real_x,img_x,count_x):
    for i,txt in enumerate(txt_list):
        if int(txt_x1[i]) in x_ax_xpts:
            if count_x < 3:
                
                    try:
                        txt_tmp = pd.to_datetime(txt)
                        mode = 'datetime'
                    except Exception as e0:
                        
                        try:
                            txt_tmp = pd.to_datetime(txt+ ' '+str(dt.date.today().year))
                            mode = 'datetime'
                        except Exception as e1: 
                            try:
                                txt_tmp = pd.to_datetime(txt+ ' Jan1')
                                mode = 'datetime'
                            except Exception as e2:
                                print(e2) 
                                if is_float(txt):
                                    real_x.append(float(txt))
                                    print(real_x, 'real x array')
                                    img_x.append(round(0.5*(int(txt_x1[i]) + int(txt_x2[i]))))
                                    print(txt, ': detected in x axis at pixel (col) ', round(0.5*(int(txt_x1[i]) + int(txt_x2[i]))) )
                                    count_x+=1
                        else:
                            if txt_tmp not in real_x:
                                real_x.append(txt_tmp)
                                img_x.append(round(0.5*(int(txt_x1[i]) + int(txt_x2[i]))))
                                print(txt, ': detected in x axis at pixel (col) ', round(0.5*(int(txt_x1[i]) + int(txt_x2[i]))) )
                                count_x+=1
                 
               
            
 
    return real_x,img_x,count_x,mode


# In[18]:


def detect_x_ax_ticks4(txt_list,box_list,
                    txt_x1,txt_x2,x_ax_xpts,
                    real_x,img_x,count_x):
    for i,txt in enumerate(txt_list):
        if int(txt_x1[i]) in x_ax_xpts:
            if count_x < 3:
                    print(real_x,count_x)
                    try:
                        txt_tmp = pd.to_datetime(txt)
                        mode = 'datetime'
                        if txt_tmp not in real_x:
                                real_x.append(txt_tmp)
                                img_x.append(round(0.5*(int(txt_x1[i]) + int(txt_x2[i]))))
                                print(txt, ': detected in x axis at pixel (col) ', round(0.5*(int(txt_x1[i]) + int(txt_x2[i]))) )
                                count_x+=1
                    except Exception as e0:
                        
                        try:
                            txt_tmp = pd.to_datetime(txt+ ' '+str(dt.date.today().year))
                            mode = 'datetime'
                            if txt_tmp not in real_x:
                                real_x.append(txt_tmp)
                                img_x.append(round(0.5*(int(txt_x1[i]) + int(txt_x2[i]))))
                                print(txt, ': detected in x axis at pixel (col) ', round(0.5*(int(txt_x1[i]) + int(txt_x2[i]))) )
                                count_x+=1
                        except Exception as e1: 
                            try:
                                txt_tmp = pd.to_datetime(txt+ ' Jan1')
                                mode = 'datetime'
                                if txt_tmp not in real_x:
                                    real_x.append(txt_tmp)
                                    img_x.append(round(0.5*(int(txt_x1[i]) + int(txt_x2[i]))))
                                    print(txt, ': detected in x axis at pixel (col) ', round(0.5*(int(txt_x1[i]) + int(txt_x2[i]))) )
                                    count_x+=1
                            except Exception as e2:
                                print(e2) 
                                if is_float(txt):
                                    mode='normal'
                                    real_x.append(float(txt))
                                    print(real_x, 'real x array')
                                    img_x.append(round(0.5*(int(txt_x1[i]) + int(txt_x2[i]))))
                                    print(txt, ': detected in x axis at pixel (col) ', round(0.5*(int(txt_x1[i]) + int(txt_x2[i]))) )
                                    count_x+=1
                        
                           
                 
               
            
 
    return real_x,img_x,count_x,mode


# In[19]:


def detect_y_ax_ticks(txt_list,box_list,
                    txt_y1,txt_y2,y_ax_ypts,
                    real_y,img_y,count_y):
    for i,txt in enumerate(txt_list):
        if int(txt_y1[i]) in y_ax_ypts:
            if count_y < 3:
                if is_float(txt):
                    if float(txt) not in real_y:
                        real_y.append(float(txt))
                        img_y.append(round(0.5*(int(txt_y1[i]) + int(txt_y2[i]))))
                        print(txt, ': detected in y axis at pixel (row) ', round(0.5*(int(txt_y1[i]) + int(txt_y2[i]))) )
                        count_y+=1
      
    return real_y,img_y,count_y


# In[20]:
st.title("Digitze your plot")

    
uploaded_file = st.file_uploader("Upload Image", type="png")
if uploaded_file is not None:
    st.write("File uploaded successfully. Click 'Run' to process the file.")
    #st.write("Using version: ",pytesseract._version_)
    # Display the Run button
    run_button = st.button("Run")
    if run_button:
    #image = Image.open(uploaded_file)
     #image = np.array(image)
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)
        st.image(image, caption='Uploaded Image.', use_column_width=True)

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
        #st.image(binary, caption='Binary Image.', use_column_width=True)
        detected_colors = dominant_colors(image)
        st.write("Detected colors:")

        color_boxes = ""
        for color in detected_colors:
            hex_color = rgb_to_hex(color)
            #color_boxes += f'<div style="width:50px; height:50px; background-color:{hex_color}; display:inline-block; margin-right:10px;"></div>'
            rgb_text = f"RGB: {color}"
            color_boxes += f'''
            <div style="display:inline-block; text-align:center; margin-right:20px;">
                <div style="width:50px; height:50px; background-color:{hex_color};"></div>
                <div style="margin-top:5px;">{rgb_text}</div>
            </div>
            '''
        st.markdown(color_boxes, unsafe_allow_html=True)


        max_len = 0
        max_color = [255,255,255]
        for colr in detected_colors:
            filt_bin = filter_color(image,colr)
            x_shift,data = get_raw_data(filt_bin)
            mode = stats.mode(data)
            data_tmp = data - mode[0]
            filt_data = data_tmp[data_tmp!=0]
            print(colr, len(filt_data),len(data))
            if len(filt_data) > max_len:
                max_len = len(filt_data)
                max_color = colr
                
        filt_bin = filter_color(image,max_color)
        x_shift,data = get_raw_data(filt_bin)



        #mode = 'datetime'


        custom_config0 = r'--psm 3 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz.-'
        custom_config1 = r'--psm 6 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz.-'
        custom_config2 = r'--psm 11 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz.-'
        custom_config3 = r''
        custom_config4 = r'--psm 6'
        custom_config5 = r'--psm 11'




        custom_configs = [custom_config0,custom_config1,custom_config2,custom_config3,custom_config4,custom_config5]
        counter_y =0
        while counter_y < 6:
            count_y = 0
            real_y, img_y = [],[]
            txt_list,box_list = extract_txt(binary, custom_configs[counter_y])
            st.write(count_y,txt_list,box_list)
            txt_x1,txt_x2,txt_y1,txt_y2 = find_txt_bnds(txt_list,box_list)
            y_ax_ypts = detect_yaxes_chars(binary,box_list,padding=2)
            real_y,img_y,count_y = detect_y_ax_ticks(txt_list,box_list,txt_y1,txt_y2,y_ax_ypts,real_y,img_y,count_y)
            if count_y >=2:
                break
            else:
                counter_y+=1
        counter_x = 0
        while counter_x < 6:
            count_x = 0
            real_x, img_x = [],[]
            txt_list,box_list = extract_txt(binary, custom_configs[counter_x])
            txt_x1,txt_x2,txt_y1,txt_y2 = find_txt_bnds(txt_list,box_list)
            x_ax_xpts = detect_xaxes_chars(binary,box_list,padding=2)
            real_x,img_x,count_x,mode = detect_x_ax_ticks4(txt_list,box_list,txt_x1,txt_x2,x_ax_xpts,real_x,img_x,count_x)
            if count_x >=2:
                break
            else:
                counter_x+=1

        #st.write(real_x,real_y)

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


        # In[ ]:


        x_ax = np.arange(0,len(data))


        # In[ ]:


        if mode == 'datetime':
            if len(real_x) > 2:
                days_p_pt1 = (real_x[2] - real_x[1]).days/(img_x[2]-img_x[1])
                days_p_pt2 = (real_x[1] - real_x[0]).days/(img_x[1]-img_x[0])
                days_p_pt = 0.5*(days_p_pt2 + days_p_pt1)
            else:
                days_p_pt = (real_x[1] - real_x[0]).days/(img_x[1]-img_x[0])


        # In[ ]:


        if mode == 'normal':
            if len(real_x) > 2:
                x_compress1 = (real_x[2] - real_x[1])/(img_x[2]-img_x[1])
                x_compress2 = (real_x[1] - real_x[0])/(img_x[1]-img_x[0])
                x_compress = 0.5*(x_compress1+x_compress2)
            else:
                x_compress = (real_x[1] - real_x[0])/(img_x[1]-img_x[0])


        # In[ ]:


        if mode == 'normal':
            if len(real_x) > 2:
                start_x1 = real_x[1] - x_compress*(img_x[1]-x_shift)
                start_x2 = real_x[2] - x_compress*(img_x[2]-x_shift)
                start_x3= real_x[0] -  x_compress*(img_x[0]-x_shift)
                start_x = (start_x1+start_x2+start_x3)/3
                fin_x_ax = start_x + x_ax*x_compress
            else:
                start_x1 = real_x[1] - x_compress*(img_x[1]-x_shift)
                start_x2= real_x[0] -  x_compress*(img_x[0]-x_shift)
                start_x = (start_x1+start_x2)/2
                fin_x_ax = start_x + x_ax*x_compress


        # In[ ]:


        if len(real_y) > 2:
            y_compress1 = (real_y[2] - real_y[1])/(img_y[2]-img_y[1])
            y_compress2 = (real_y[1] - real_y[0])/(img_y[1]-img_y[0])
            y_compress = 0.5*(y_compress1+y_compress2)
        else:
            y_compress = (real_y[1] - real_y[0])/(img_y[1]-img_y[0])


        # In[ ]:


        #real_y[0] - y_compress*img_y[0]
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



        if mode=='datetime':
            if len(real_x)>2:
                start_date1 = real_x[1] - pd.Timedelta(days=days_p_pt* (img_x[1]-x_shift))
                start_date2 = real_x[2] - pd.Timedelta(days=days_p_pt* (img_x[2]-x_shift))
                start_date3 = real_x[0] - pd.Timedelta(days=days_p_pt* (img_x[0]-x_shift))
                start_date = pd.to_datetime((start_date1.timestamp() + start_date2.timestamp() +start_date3.timestamp())/3,unit='s')
                img_date_idx =start_date + pd.to_timedelta(x_ax*days_p_pt, unit='D')
            else:
                print('here')
                start_date1 = real_x[1] - pd.Timedelta(days=days_p_pt* (img_x[1]-x_shift))
                start_date2 = real_x[0] - pd.Timedelta(days=days_p_pt* (img_x[0]-x_shift))
                start_date = pd.to_datetime((start_date1.timestamp() + start_date2.timestamp())/2,unit='s')
                img_date_idx =start_date + pd.to_timedelta(x_ax*days_p_pt, unit='D')


        if mode == 'datetime':
            df_tmp = pd.DataFrame(data_shift, index=img_date_idx, columns=['A'])
            df = df_tmp.resample('D').median().interpolate()



        if mode == 'normal':
            df = pd.DataFrame(data_shift, index=fin_x_ax, columns=['Data'])
            

        st.write("Output Data:")
        #st.dataframe(df)
        st.line_chart(df)
        st.download_button(
                        label="Download data as CSV",
                        data=df.to_csv(),
                        file_name='plot_data.csv',
                        mime='text/csv',
                    )





