#!/usr/bin/env python
# coding: utf-8

# In[1]:


import soundfile as sf    
import librosa            
import os                 
import glob               
import pickle             
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier 
from sklearn.metrics import accuracy_score 


# In[2]:


def feature_extraction(fileName,mfcc,chroma,mel):
    with sf.SoundFile(fileName) as file:
        sound = file.read(dtype='float32')
        sample_rate = file.samplerate    
        if chroma:          
            stft = np.abs(librosa.stft(sound))
        feature = np.array([])                
        if mfcc:
            mfcc = np.mean(librosa.feature.mfcc(y=sound,sr=sample_rate,n_mfcc=40).T,axis=0)
            feature =np.hstack((feature,mfcc))
        if chroma:
            chroma =  np.mean(librosa.feature.chroma_stft(S=stft,sr=sample_rate).T,axis=0)
            feature = np.hstack((feature,chroma))
        if mel:
            mel = np.mean(librosa.feature.melspectrogram(y=sound,sr=sample_rate).T,axis=0)
            feature =np.hstack((feature,mel))
        return feature  


# In[3]:


int_emotion = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised"
}
#Emotions we want to observe
EMOTIONS = {"happy","sad","neutral","angry","calm","fearful","disgust","surprised"}


# In[4]:


def train_test_data(test_size=0.3):
    features, emotions = [],[] #intializing features and emotions
    for file in glob.glob("data/Actor_/.wav"):
        fileName = os.path.basename(file)   #obtaining the file name
        emotion = int_emotion[fileName.split("-")[2]] #getting the emotion
        if emotion not in EMOTIONS:
            continue
        feature=feature_extraction(file,mfcc=True,chroma=True,mel=True,) #extracting feature from audio
        features.append(feature)
        emotions.append(emotion)
    return train_test_split(np.array(features),emotions, test_size=test_size, random_state=7) #returning the data splited into train and test set
we are obtaining train and test data from train_test_data(). Here, the test size is 30%.


# In[5]:


X_train,X_test,y_train,y_test=train_test_data(test_size=0.3)
print("Total number of training sample: ",X_train.shape[0])
print("Total number of testing example: ",X_test.shape[0])
print("Feature extracted",X_train.shape[1])


# In[6]:


model=MLPClassifier(alpha=0.01, batch_size=256, epsilon=1e-08, hidden_layer_sizes=(400,), learning_rate='adaptive', max_iter=1000)


# In[7]:


print("___Training the model___")
model.fit(X_train,y_train)


# In[8]:


y_pred = model.predict(X_test)


# In[9]:


accuracy = accuracy_score(y_true=y_test,y_pred=y_pred)
accuracy*=100
print("accuracy: {:.4f}%".format(accuracy))


# In[10]:


if not os.path.isdir("model"): 
   os.mkdir("model") 
pickle.dump(model, open("model/mlp_classifier.model", "wb"))


# In[ ]:




