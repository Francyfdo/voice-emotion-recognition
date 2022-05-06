#!/usr/bin/env python
# coding: utf-8

# In[13]:


conda install -c numba numba
install -c conda-forge librosa
conda install numpy,pyaudio,scikit-learn==0.19
conda install -c conda-forge pysoundfile


# In[14]:


import soundfile
import numpy as np 
import librosa  
import glob 
import os # to use operating system dependent functionality
from sklearn.model_selection import train_test_split # for splitting training and testing 
from sklearn.neural_network import MLPClassifier # multi-layer perceptron model 
from sklearn.metrics import accuracy_score # to measure how good we are


# In[15]:


def get_feature(file_name,mfccs,mel,chroma,contrast):
    
        data, sample_rate = librosa.load(file_name)
        stft = np.abs(librosa.stft(data))
        mfccs = np.mean(librosa.feature.mfcc(y=data, sr=sample_rate, n_mfcc=40).T, axis=0)
        mel = np.mean(librosa.feature.melspectrogram(data, sr=sample_rate).T,axis=0)
        chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
        contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T,axis=0)
        
        
    
        return mfccs,mel,chroma,contrast


# In[16]:


# emotions in dataset
list_emotion = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised"
}

classify_emotions = {
    "sad",
    "happy",
    "surprised"
    "disgust"
    "fearful"
    "angry"
    "neutral"
    "calm"
    
    
}


# In[17]:


def load_data(test_size=0.2):
    feature, y = [], []
    for file in glob.glob("C:\\Users\\Documents\\ravdess data\\Actor_\\.wav"):
        basename = os.path.basename(file)  # get the base name of the audio file
       
        emotion = list_emotion[basename.split("-")[2]]   # get the emotion label
       
        if emotion not in classify_emotions:    # we allow only classify_emotions we set
            try:
                mfccs,mel,chroma,contrast = get_feature(file)
            except Exception as e:
                print ("Error encountered while parsing file: ", file)
                continue
            ext_features = np.hstack([mfccs,mel,chroma,contrast])
            feature.append(ext_features)
            y.append(emotion)
        
 
    return train_test_split(np.array(feature), y, test_size=test_size, random_state=9)


# In[18]:


feature_train, feature_test, y_train, y_test = load_data(test_size=0.25)


# In[19]:


print("Number of samples in training data:", feature_train.shape[0])
print("Number of samples in testing data:", feature_test.shape[0])


# In[20]:


print("Training the model.....")
clf=MLPClassifier(alpha=0.01, batch_size=256, epsilon=1e-08, hidden_layer_sizes=(300,), learning_rate='adaptive', max_iter=500).fit(feature_train, y_train)


# In[21]:


y_pred = clf.predict(feature_test)
# calculate the accuracy
accuracy = accuracy_score(y_true=y_test, y_pred=y_pred)
print("Accuracy is: {:.2f}%".format(accuracy*100))


# In[22]:


print("Number of features:", feature_train.shape[1])


# In[ ]:





# In[ ]:





# In[ ]:




