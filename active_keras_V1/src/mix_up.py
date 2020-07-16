import numpy as np
from PIL import Image
import random
from task1_resnet101active import get_data_label_list
from keras.applications.imagenet_utils import preprocess_input
import pdb
def mix_data(datas,labels,i,j,sub_mean=False):
    image1=datas[i]
    label1=labels[i]
    image2=datas[j]
    label2=labels[j]

    r = np.array(random.random())

    if sub_mean:
        g1 = np.std(image1)
        g2 = np.std(image2)
        p = 1.0 / (1 + g1 / g2 * (1 - r) / r)
        image = ((image1 * p + image2 * (1 - p)) / np.sqrt(p ** 2 + (1 - p) ** 2)).astype(np.float32)
    else:
        image = (image1 * r + image2 * (1 - r)).astype(np.float32)

            # Mix two labels
    eye = np.eye(2)
    label = (eye[label1] * r + eye[label2] * (1 - r)).astype(np.float32)
    return image, label
'''
def get_mixed_pair(data_dir,sub_mean):
    data_lists = []
    label_lists = []
    data_label_lists,_ = get_data_label_list(data_dir)
    mix_data_lists=[]
    mix_label_lists=[]
    
    #image1,label1=data_label_lists[random.randint(0,len(data_label_lists)-1)]['file_path']
    #label1=data_label_lists[random.randint(0,len(data_label_lists)-1)]['label']
    for i in range(int(len(data_label_lists))):
        #for i in range(batch_size):
        data_label = data_label_lists[i]
        img_RGB=Image.open(data_label['file_path'])
        img_RGB_resize= img_RGB.resize((224,224),Image.ANTIALIAS)
        img = np.asarray(img_RGB_resize,'f')#change to numpy
        # img.flags.writeable = True
        img_preprocess = preprocess_input(img)
        data_lists.append(img)
        label_lists.append(data_label['label'])
    datas = np.array(data_lists)
    labels = np.array(label_lists)
    p_list=np.where(labels==1)[0].tolist()
    n_list=np.where(labels==0)[0].tolist()
    random_selected_n_list=random.sample(n_list,len(p_list))
    j=0
    for i in range(len(datas)):
   
        if labels[i]==1:
            image,label=mix_data(datas,labels,i,random_selected_n_list[j],sub_mean=sub_mean)
            j+=1
         
        else:
            
            image,label=mix_data(datas,labels,i,p_list[random.randint(0,len(p_list)-1)],sub_mean=sub_mean)
        mix_data_lists.append(image)
        mix_label_lists.append(label)
    #datas=np.array(mix_data_lists)
    #labels=np.array(mix_label_lists)
    return mix_data_lists,mix_label_lists
'''
def get_mix_data_label(data_dir,batch_size,sub_mean=False):
    data_lists = []
    label_lists = []
    mix_data,mix_label=get_mixed_pair(data_dir,sub_mean)
    for epoch in range (int(len(mix_data)//batch_size)):
        for i in range(batch_size):
            label=mix_label[epoch*batch_size+i]
            img=mix_data[epoch*batch_size+i]
            data_lists.append(img)
            label_lists.append(label)
    datas = np.array(data_lists)
    labels = np.array(label_lists)
    return datas,labels

def normalize(mean, std):
    def f(image):
        return (image - mean[:, None, None]) / std[:, None, None]

    return f


def get_mixed_pair(select_data_label_list,sub_mean):
    ## mix 1/4 data
    data_lists = []
    label_lists = []
    data_dir = '../../../../2017_ISBI_suply/224data/task2/train'
    data_label_lists,_ = get_data_label_list(data_dir)
    
    mix_data_lists=[]
    mix_label_lists=[]
    for i in range(int(len(data_label_lists))):
        #for i in range(batch_size):
        data_label = data_label_lists[i]
        img_RGB=Image.open(data_label['file_path'])
        img_RGB_resize= img_RGB.resize((224,224),Image.ANTIALIAS)
        img = np.asarray(img_RGB_resize)#change to numpy
        data_lists.append(img)
        label_lists.append(data_label['label'])
    datas = np.array(data_lists)
    labels = np.array(label_lists)
    p_list=np.where(labels==1)[0].tolist()
    n_list=np.where(labels==0)[0].tolist()
    
    random_selected_p_list=random.sample(p_list,int(len(p_list)/4))
    random_selected_n_list=random.sample(n_list,int(len(n_list)/4))
    random_selected_n_list_2=random.sample(random_selected_n_list,len(random_selected_p_list))
    j=0
    for i in range(len(datas)):
        if i in random_selected_p_list:
            image,label=mix_data(datas,labels,i,random_selected_n_list_2[j],sub_mean=False)
            j+=1
            mix_data_lists.append(image)
            mix_label_lists.append(label)
         
        elif i in random_selected_n_list:
            
            image,label=mix_data(datas,labels,i, random_selected_p_list[random.randint(0,len( random_selected_p_list)-1)],sub_mean=False)
   
            
            
            mix_data_lists.append(image)
            mix_label_lists.append(label)

    return mix_data_lists,mix_label_lists