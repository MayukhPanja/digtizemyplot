#!/usr/bin/env python
# coding: utf-8


import json
import pandas as pd
from scipy import stats
import datetime as dt
import io
from matplotlib import pyplot as plt
import numpy as np
import cv2
from google.cloud import vision
from google.oauth2 import service_account
from sklearn.cluster import KMeans

import streamlit as st


def is_float(s):
    try:
        float(s)
        return True
    except ValueError:
        print(s, 'is not a float')
        return False

def has_point(s):
    if '.' in s:
        return True
    else:
        return False

def ends_with_dash(s):
    if s.endswith('-'):
        return True
    else:
        return False

def try_dtime_1(txt):
    try:
        return pd.to_datetime(txt)
        
    except Exception as e: 
        print(e)
        return False
    
def try_dtime_2(txt):
    try:
        return  pd.to_datetime(txt + ' ' + str(dt.date.today().year))
        
    except Exception as e:
        print(e)
        return False

def check_space(str):
    if [str] == str.split(' '):
        return False
    else:
        return True




def dominant_colors(image, num_colors=7):

    pixels = image.reshape(-1, 3)

    # Use KMeans to cluster pixels
    kmeans = KMeans(n_clusters=num_colors)
    kmeans.fit(pixels)

    # Get the RGB values of the cluster centers
    dominant_colors = kmeans.cluster_centers_

    # Convert to integer values
    dominant_colors = dominant_colors.round(0).astype(int)

    return dominant_colors

def filter_color(image, target_rgb, tolerance=25): ###default tolerance 15

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


def plot_colors(colors):
    # Create a pie chart to display the dominant colors
    plt.figure(figsize=(8, 6))
    plt.pie([1]*len(colors), colors=np.array(colors)/255, labels=colors, startangle=90)
    plt.axis('equal')
    plt.show()




def get_raw_data(filt_bin):
    binary_final = filt_bin
    pos= np.argmax(binary_final,axis=0)
    im_ht = np.shape(filt_bin)[0]
    im_ln = np.shape(filt_bin)[1]
    pos_flip = im_ht - pos
    x_loc = pos_flip < im_ht
    data = pos_flip[x_loc]
    data_full = np.where(x_loc == True, pos_flip, np.nan)
    x_shift = np.argmax(x_loc)
    #x_end = im_ln -  np.argmax(np.flip(x_loc)) - 1
    return x_shift,data,data_full

# Function to convert RGB to hex
def rgb_to_hex(rgb):
    return '#{:02x}{:02x}{:02x}'.format(rgb[0], rgb[1], rgb[2])

def detect_ax_ticks(texts):
        txt_xmin = []
        txt_xmax = []
        txt_ymin = []
        txt_ymax = []
        txt_list = []
        counter = 1
        txt_list_org = texts[0].description.split('\n')
        for txt in txt_list_org:
            word_count = 0
            word_count2 = 0
            for word in txt.split(' '):
                #print(txt,word,texts[counter].description,counter)
                if word != texts[counter].description:
                    counter+=1
                if word == texts[counter].description:
                    
                    if word_count == 0:
                        txt_xmin.append(texts[counter].bounding_poly.vertices[0].x)
                        txt_ymin.append(texts[counter].bounding_poly.vertices[0].y)
                        print(txt, ' added min')
                    word_count2+=1
                counter+=1
                word_count+=1
                
        
            #print(counter-1)
            if word_count == word_count2:
                print(txt, ' added max',word_count,word_count2)
                txt_xmax.append(texts[counter-1].bounding_poly.vertices[1].x)
                txt_ymax.append(texts[counter-1].bounding_poly.vertices[2].y)
                txt_list.append(txt)
        
        ypos1 = np.where(np.diff(txt_ymin,n=1,append=txt_ymin[-1]) <=3,-1,0)
        ypos2 = np.where(np.diff(txt_ymax,n=1,append=txt_ymax[-1]) <=3,-1,0)
        xpos1 = np.where(np.diff(txt_xmin,n=1,prepend=txt_xmin[0]) <=3,1,0)
        xpos2 = np.where(np.diff(txt_xmax,n=1,prepend=txt_xmax[0]) <=3,1,0)
        
        ax_pos = ypos2+ypos1+xpos1+xpos2
        flag_ytick = np.where(ax_pos == 2,1,0)
        flag_xtick = np.where(ax_pos == -2,1,0)
        flag_tick = {'flag_xtick':flag_xtick,'flag_ytick':flag_ytick}
        txt_pos = { 'txt_xmin':txt_xmin, 'txt_xmax':txt_xmax,'txt_ymin':txt_ymin, 'txt_ymax':txt_ymax}
        return txt_list,flag_tick,txt_pos

def detect_ax_ticks_backup(texts):
    txt_xmin = []
    txt_xmax = []
    txt_ymin = []
    txt_ymax = []
    txt_list = []
    counter = 1
    txt_list_org = texts[0].description.split('\n')
    for txt in txt_list_org:
        word_count = 0
        for word in txt.split(' '):
            #print(txt,word,texts[counter].description,counter)
            if word != texts[counter].description:
                counter+=1
            if word == texts[counter].description:
                
                if word_count == 0:
                    txt_xmin.append(texts[counter].bounding_poly.vertices[0].x)
                    txt_ymin.append(texts[counter].bounding_poly.vertices[0].y)
                    txt_xmax.append(texts[counter].bounding_poly.vertices[1].x)
                    txt_ymax.append(texts[counter].bounding_poly.vertices[2].y)
                    txt_list.append(word)
                
                
            counter+=1
            word_count+=1
    ypos1 = np.where(np.diff(txt_ymin,n=1,append=txt_ymin[-1]) <=3,-1,0)
    ypos2 = np.where(np.diff(txt_ymax,n=1,append=txt_ymax[-1]) <=3,-1,0)
    xpos1 = np.where(np.diff(txt_xmin,n=1,prepend=txt_xmin[0]) <=3,1,0)
    xpos2 = np.where(np.diff(txt_xmax,n=1,prepend=txt_xmax[0]) <=3,1,0)

    ax_pos = ypos2+ypos1+xpos1+xpos2
    flag_ytick = np.where(ax_pos == 2,1,0)
    flag_xtick = np.where(ax_pos == -2,1,0)
    
    flag_tick = {'flag_xtick':flag_xtick,'flag_ytick':flag_ytick}
    txt_pos = { 'txt_xmin':txt_xmin, 'txt_xmax':txt_xmax,'txt_ymin':txt_ymin, 'txt_ymax':txt_ymax}
    return txt_list,flag_tick,txt_pos

def get_yticks(txt_list,flag_ytick,txt_ymin,txt_ymax):
    real_y,img_y = [],[]
    count_y = 0
    
    for i,txt in enumerate(txt_list):
        if flag_ytick[i] == 1:
            if is_float(txt):
                count_y+=1
                real_y.append(float(txt))
                img_y.append(round(0.4*txt_ymin[i] + 0.6*txt_ymax[i]))
            elif ends_with_dash(txt):
                if is_float(txt[:-1]):
                    count_y+=1
                    real_y.append(float(txt[:-1]))
                    img_y.append(round(0.4*txt_ymin[i] + 0.6*txt_ymax[i]))
                    
    img_y = image.shape[0] - np.array(img_y)
    return real_y,img_y

def get_xticks(txt_list,flag_xtick,txt_xmin,txt_xmax):
    real_x, img_x = [],[]
    count_x = 0
    for i,txt in sorted(enumerate(txt_list),reverse=True):
        if flag_xtick[i] == 1:
            print(txt)
            if try_dtime_1(txt) != False:
                print(txt,' Here:1')
                count_x+=1
                real_x.append(try_dtime_1(txt))
                img_x.append(round(0.5*(txt_xmax[i] + txt_xmin[i])))
                mode = 'dtime'
            elif try_dtime_2(txt) != False and is_float(txt) == False and has_point(txt) == False:
                print(txt,' Here:2')
                count_x+=1
                real_x.append(try_dtime_2(txt))
                img_x.append(round(0.5*(txt_xmax[i] + txt_xmin[i])))
                mode = 'dtime-daily'
                
            elif is_float(txt):
                print(txt,' Here:3')
                count_x_=1
                real_x.append(float(txt))
                img_x.append(round(0.5*(txt_xmax[i] + txt_xmin[i])))
                mode = 'normal'
            else:
                continue
    return real_x,img_x,mode

def detect_plot_color(detected_colors):
    df = pd.DataFrame(columns=['Color','Coherent length','Length', 'Brightness','IsGray'])
    for colr in detected_colors:
        filt_bin = filter_color(image,colr)
        x_shift,data,_ = get_raw_data(filt_bin)
        mode_st = stats.mode(data,keepdims=False)
        data_tmp = data - mode_st[0]
        filt_data = data_tmp[data_tmp!=0]
        bright = 0.299*colr[0] + 0.587*colr[1] + 0.114*colr[2]
        #plt.plot(filt_data)
        if len(filt_data) < image.shape[1]*0.2:
            continue
        if np.all(colr == colr[0]):
            isgray = 1
        else:
            isgray = 0
        df.loc[len(df.index)] = [colr, len(filt_data),len(data),bright,isgray]
        #if len(filt_data) > max_len:
        #    max_len = len(filt_data)
        #    max_color = colr
        #st.write(df)
    if df['IsGray'].mean() !=1:
        df_new = df[df['IsGray'] != 1].copy()
    else:
        df_new = df
    chosen_br = df_new.sort_values(by=['Coherent length'],ascending=False)[0:3]['Brightness'].min() 
    chosen_clr = df_new[df_new['Brightness'] == chosen_br]['Color'].iloc[0]
    return chosen_clr


st.title("DigMyPlot: Digitize your plot")
st.write("Automated digitization of images of plots")

uploaded_file = st.file_uploader("Upload Image", type="png")
if uploaded_file is not None:
    content = uploaded_file.read()
#with open('digmyplot_gvis_apicred.json') as f:
#    credentials_json = json.load(f)
# Access the credentials from st.secrets
    credentials_info = st.secrets["google_vision_credentials"]
    # Create credentials object from the dictionary
    credentials = service_account.Credentials.from_service_account_info(credentials_info)

    # Initialize the Vision API client with the credentials
    client = vision.ImageAnnotatorClient(credentials=credentials)




#with io.open('test_image1.png', 'rb') as image_file:
#        content = image_file.read()

    image = vision.Image(content=content)

    # Perform text detection
    response = client.text_detection(image=image)
    texts = response.text_annotations

    image_array = np.frombuffer(content, dtype=np.uint8)
    image = cv2.cvtColor(cv2.imdecode(image_array, cv2.IMREAD_COLOR),cv2.COLOR_BGR2RGB)
    #image = cv2.flip(image,0)
    st.image(image,caption="Uploaded Image")
    plt.show()
        
    txt_list,flag_tick,txt_pos = detect_ax_ticks(texts)
    real_y,img_y = get_yticks(txt_list,flag_tick['flag_ytick'],txt_pos['txt_ymin'],txt_pos['txt_ymax'])
    real_x,img_x,mode = get_xticks(txt_list,flag_tick['flag_xtick'],txt_pos['txt_xmin'],txt_pos['txt_xmax'])
    if len(real_x)<2:
        txt_list2,flag_tick2,txt_pos2 = detect_ax_ticks_backup(texts)
        real_x,img_x,mode = get_xticks(txt_list2,flag_tick2['flag_xtick'],txt_pos2['txt_xmin'],txt_pos2['txt_xmax'])
        st.write('Axis detection backup algorithm deployed.')
    if len(real_y)<2:
        txt_list2,flag_tick2,txt_pos2 = detect_ax_ticks_backup(texts)
        real_y,img_y = get_yticks(txt_list2,flag_tick2['flag_ytick'],txt_pos2['txt_ymin'],txt_pos2['txt_ymax'])
        st.write('Axis detection backup algorithm deployed.')
    detected_colors = dominant_colors(image)
    st.write("**Detecting all colors in image:**")

    
    color_boxes = ""
    for idx, color in enumerate(detected_colors, 1):
        if np.all(color == np.array([255,255,255])):
            continue
        hex_color = rgb_to_hex(color)
        #color_boxes += f'<div style="width:50px; height:50px; background-color:{hex_color}; display:inline-block; margin-right:10px;"></div>'
        #<p style="font-weight:bold;">{idx}</p> IF I WANT TO LATER ADD NUMBERS To DISPLAYED COLOR
        rgb_text = f"RGB: {color}"
        color_boxes += f'''
        
        <div style="display:inline-block; text-align:center; margin-right:20px;">
            <div style="width:50px; height:50px; background-color:{hex_color};"></div>
            
        </div>
        '''
    st.markdown(color_boxes, unsafe_allow_html=True)

    
    
    
    plot_color = detect_plot_color(detected_colors)
    hex_color = rgb_to_hex(plot_color)
    #rgb_text = f"RGB: {chosen_clr}"
    st.write("**Detecting color of plot:**")
    color_box = f'''
        <div style="display:inline-block; text-align:center; margin-right:20px;">
            <div style="width:50px; height:50px; background-color:{hex_color};"></div>
        
        </div>
        '''
    st.markdown(color_box, unsafe_allow_html=True)
    filt_bin = filter_color(image,plot_color)
    x_shift,data,data_full = get_raw_data(filt_bin)
    

    # In[11]:

    st.write('**Detecting axis ticks**')
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    gray = cv2.flip(gray,0)
    gray_tmp = gray.copy()
    g=1;h=0
    gray_tmp[:,img_x[g]] = 0
    gray_tmp[:,img_x[h]] = 0
    gray_tmp[:,x_shift] = 0
    gray_tmp[img_y[g],:] = 0
    gray_tmp[img_y[h],:] = 0
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(gray_tmp, cmap='gray', origin='lower')

    # Add text annotations based on the mode
    if mode == 'dtime' or mode == 'dtime-daily':
        ax.text(img_x[g]-50, 150, 'Detected: ' + str(real_x[g]),color='red')
        ax.text(img_x[h]-50, 50, 'Detected: ' + str(real_x[h]),color='red')
        #ax.text(x_shift-50, 50, 'Detected: data begins',color='red')
    else:
        ax.text(img_x[g]-50, 150, 'Detected: ' + str(real_x[g]),color='red')
        ax.text(img_x[h]-50, 50, 'Detected: ' + str(real_x[h]),color='red')
        #ax.text(x_shift-50, 50, 'Detected: data begins',color='red')

    ax.text(200, img_y[g]+10, 'Detected: ' + str(real_y[g]),color='red')
    ax.text(250, img_y[h]+10, 'Detected: ' + str(real_y[h]),color='red')
    st.pyplot(fig)

    st.write('**Detecting data points**')
    fig1, ax1 = plt.subplots(figsize=(10, 5))
    ax1.imshow(gray, cmap='gray', origin='lower')
    ax1.plot(data_full,'r*')
    st.pyplot(fig1)


    x_ax = np.arange(0,len(data_full))
    if mode == 'dtime' or mode == 'dtime-daily':
        if len(real_x) > 2:
            days_p_pt1 = (real_x[0] - real_x[1]).days/(img_x[0]-img_x[1])
            days_p_pt2 = (real_x[1] - real_x[2]).days/(img_x[1]-img_x[2])
            days_p_pt = 0.5*(days_p_pt2 + days_p_pt1)
            start_date1 = real_x[0] - pd.Timedelta(days=days_p_pt* (img_x[0]))
            start_date2 = real_x[1] - pd.Timedelta(days=days_p_pt* (img_x[1]))
            start_date3 = real_x[2] - pd.Timedelta(days=days_p_pt* (img_x[2]))
            start_date = pd.to_datetime((start_date1.timestamp() + start_date2.timestamp() +start_date3.timestamp())/3,unit='s')
            img_date_idx =start_date + pd.to_timedelta(x_ax*days_p_pt, unit='D')
        elif len(real_x) == 2:
            days_p_pt = (real_x[0] - real_x[1]).days/(img_x[0]-img_x[1])
            start_date1 = real_x[1] - pd.Timedelta(days=days_p_pt* (img_x[1]))
            start_date2 = real_x[0] - pd.Timedelta(days=days_p_pt* (img_x[0]))
            start_date = pd.to_datetime((start_date1.timestamp() + start_date2.timestamp())/2,unit='s')
            img_date_idx =start_date + pd.to_timedelta(x_ax*days_p_pt, unit='D')
        
    if mode == 'normal':
        if len(real_x) > 2:
            x_compress1 = (real_x[0] - real_x[1])/(img_x[0]-img_x[1])
            x_compress2 = (real_x[1] - real_x[2])/(img_x[1]-img_x[2])
            x_compress = 0.5*(x_compress1+x_compress2)
            start_x1 = real_x[1] - x_compress*(img_x[1])
            start_x2 = real_x[2] - x_compress*(img_x[2])
            start_x3= real_x[0] -  x_compress*(img_x[0])
            start_x = (start_x1+start_x2+start_x3)/3
            fin_x_ax = start_x + x_ax*x_compress
        elif len(real_x) == 2:
            x_compress = (real_x[0] - real_x[1])/(img_x[0]-img_x[1])
            start_x1 = real_x[1] - x_compress*(img_x[1])
            start_x2= real_x[0] -  x_compress*(img_x[0])
            start_x = (start_x1+start_x2)/2
            fin_x_ax = start_x + x_ax*x_compress

    if len(real_x) < 2:
        print('Automatic X-Tick Detection failed')
        


    # In[13]:


    if len(real_y) > 2:
        y_compress1 = (real_y[2] - real_y[1])/(img_y[2]-img_y[1])
        y_compress2 = (real_y[1] - real_y[0])/(img_y[1]-img_y[0])
        y_compress = 0.5*(y_compress1+y_compress2)
        c1 = real_y[0] - y_compress*img_y[0]
        c2 = real_y[1] - y_compress*img_y[1]
        c3 = real_y[2] - y_compress*img_y[2]
        c = (c1+c2+c3)/3
    elif len(real_y) == 2:
        y_compress = (real_y[1] - real_y[0])/(img_y[1]-img_y[0])
        c1 = real_y[0] - y_compress*img_y[0]
        c2 = real_y[1] - y_compress*img_y[1]
        c = 0.5*(c1+c2)


    # In[14]:


    data_shift = data_full*y_compress + c
    if mode == 'dtime':
        df = pd.DataFrame(data_shift, index=img_date_idx, columns=['Data'])
    if mode == 'dtime-daily':
        df_tmp = pd.DataFrame(data_shift, index=img_date_idx, columns=['Data'])
        df = df_tmp.resample('D').median()#.interpolate()
    if mode == 'normal':
        df = pd.DataFrame(data_shift, index=fin_x_ax, columns=['Data'])


    # In[15]:


     # Create a plot with customized y-axis limits
    fig3, ax3 = plt.subplots(figsize=(10,5))
    ax3.plot(df,'k')
    ax3.set_ylim([df['Data'].min()-0.5*df['Data'].std(), df['Data'].max()+0.5*df['Data'].std()])
    # Display the chart in Streamlit
    st.pyplot(fig3)

    st.markdown("""
        <style>
        .stDownloadButton > button {
            background-color: rgba(52, 201, 235);
            color: white;
        }
        </style>
        """, unsafe_allow_html=True)
    
    st.download_button(
                        label="Download data as CSV",
                        data=df.to_csv(),
                        file_name='digmyplot_data.csv',
                        mime='text/csv',
                    )

st.write("App designed by Mayukh Panja. For feedback write to mayukhpanja@gmail.com or @mayukh_panja on X.")




