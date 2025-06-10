#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install librosa')


# In[2]:


import librosa
import os
import numpy as np
from keras.models import Model
from keras.layers import Input, Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, GlobalAveragePooling1D
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder


# In[3]:


def extract_features(file_path):
    audio_data, sampling_rate = librosa.load(file_path, sr=None)
    features = []
    mfccs = librosa.feature.mfcc(y=audio_data, sr= sampling_rate, n_mfcc=13)
    features.append(np.mean(mfccs, axis=1))
    chroma = librosa.feature.chroma_stft(y=audio_data, sr=sampling_rate)
    features.append(np.mean(chroma,axis=1))
    mel = librosa.feature.melspectrogram(y=audio_data, sr=sampling_rate)
    features.append(np.mean(mel,axis=1))
    return np.concatenate(features)


# In[4]:


def load_audio_file(data_dir):
    X=[]
    for file in os.listdir(data_dir):
        if file.endswith(".wav"):
            file_path = os.path.join(data_dir, file)
            features = extract_features(file_path)
            X.append(features)
    return np.array(X)


# In[5]:


def training_fault(data_dir):
    y=[]
    for file in os.listdir(data_dir):
        if file.endswith(".wav"):
            if file[4]=="N":
                label=file[4]
            else:
                label = file[4:7]
            y.append(label)
    return np.array(y)

def training_model_name(data_dir):
    y=[]
    for file in os.listdir(data_dir):
        if file.endswith(".wav"):             
            label = file[0:1]
            y.append(label)
    return np.array(y)

def training_man_direction(data_dir):
    y=[]
    for file in os.listdir(data_dir):
        if file.endswith(".wav"):
            label = file[2:3]
            y.append(label)
    return np.array(y)


# In[66]:


def training_model(X_input, main_output, auxilary_output1, auxilary_output2):
    
    X_input_reshaped = X_input.reshape(X_input.shape[0], X_input.shape[1], 1)
    
    num_main_classes = main_output.shape[1]
    num_aux_classes_fault = auxilary_output1.shape[1]
    num_aux_classes_model = auxilary_output2.shape[1]

    input_shape = (X_input_reshaped.shape[1], X_input_reshaped.shape[2])

    input_layer = Input(shape=input_shape, name='main_input')

    shared_conv1 = Conv1D(filters=64, kernel_size=3, activation='relu')(input_layer)
    shared_pool1 = MaxPooling1D(pool_size=2)(shared_conv1)
    shared_conv2 = Conv1D(filters=128, kernel_size=3, activation='relu')(shared_pool1)
    shared_pool2 = MaxPooling1D(pool_size=2)(shared_conv2)
    shared_conv3 = Conv1D(filters=256, kernel_size=3, activation='relu')(shared_pool2)
    shared_pool3 = MaxPooling1D(pool_size=2)(shared_conv3)
    shared_flat = Flatten()(shared_pool3)

    main_dense1 = Dense(512, activation='relu')(shared_flat)
    main_dropout1 = Dropout(0.5)(main_dense1)
    main_dense2 = Dense(256, activation='relu')(main_dropout1)
    main_dropout2 = Dropout(0.5)(main_dense2)
    main_output_layer = Dense(num_main_classes, activation='softmax', name='main_output')(main_dropout2)

    aux_dense_fault1 = Dense(512, activation='relu')(shared_flat)
    aux_dropout_fault1 = Dropout(0.5)(aux_dense_fault1)
    aux_dense_fault2 = Dense(256, activation='relu')(aux_dropout_fault1)
    aux_dropout_fault2 = Dropout(0.5)(aux_dense_fault2)
    aux_output_fault = Dense(num_aux_classes_fault, activation='softmax', name='aux_output_fault')(aux_dropout_fault2)

    aux_dense_model1 = Dense(512, activation='relu')(shared_flat)
    aux_dropout_model1 = Dropout(0.5)(aux_dense_model1)
    aux_dense_model2 = Dense(256, activation='relu')(aux_dropout_model1)
    aux_dropout_model2 = Dropout(0.5)(aux_dense_model2)
    aux_output_model = Dense(num_aux_classes_model, activation='softmax', name='aux_output_model')(aux_dropout_model2)

    model = Model(inputs=input_layer, outputs=[main_output_layer, aux_output_fault, aux_output_model])
    
    model.compile(optimizer='adam',
                  loss={'main_output': 'categorical_crossentropy', 
                        'aux_output_fault': 'categorical_crossentropy',
                        'aux_output_model': 'categorical_crossentropy'},
                  metrics={'main_output': 'accuracy', 
                           'aux_output_fault': 'accuracy',
                           'aux_output_model': 'accuracy'})
    
    model.fit(X_input_reshaped,
              [main_output, auxilary_output1, auxilary_output2],
              epochs=20,
              batch_size=16)
    
    return model


# In[ ]:


drones = ['A', 'B', 'C']
mics = ['mic1', 'mic2']


# In[1]:


X_train = []
y_train_fault = []
y_train_model_name = []
y_train_man_dir = []

for drone in drones:
    for mic in mics:
        data_dir = f"/kaggle/input/sound-based-drone-fault-classification-using-multi/Dataset/Dataset/drone_{drone}/{drone}/train/{mic}"
        
        # Load audio features
        X_train.append(load_audio_file(data_dir))
        y_train_fault.append(training_fault(data_dir))
        
        # Load labels
        y_train_model_name.append(training_model_name(data_dir))
        y_train_man_dir.append(training_man_direction(data_dir))

X_train = np.concatenate(X_train)
y_train_fault = np.concatenate(y_train_fault)
y_train_model_name = np.concatenate(y_train_model_name)
y_train_man_dir = np.concatenate(y_train_man_dir)


# In[28]:


label_encoder = LabelEncoder()


# In[29]:


y_train_man_dir_encoded = label_encoder.fit_transform(y_train_man_dir)
y_train_fault_encoded = label_encoder.fit_transform(y_train_fault)
y_train_model_name_encoded = label_encoder.fit_transform(y_train_model_name)


# In[30]:


X_train.shape , y_train_man_dir_encoded.shape, y_train_fault_encoded.shape, y_train_model_name_encoded.shape


# In[31]:


y_train_man_dir_encoded = to_categorical(y_train_man_dir_encoded.reshape(-1, 1), num_classes=len(set(y_train_man_dir)))
y_train_fault_encoded = to_categorical(y_train_fault_encoded.reshape(-1, 1), num_classes=len(set(y_train_fault)))
y_train_model_name_encoded = to_categorical(y_train_model_name_encoded.reshape(-1, 1), num_classes=len(set(y_train_model_name)))


# In[2]:


X_test = []
y_test_fault = []
y_test_model_name = []
y_test_man_dir = []

for drone in drones:
    for mic in mics:
        data_dir = f"/kaggle/input/sound-based-drone-fault-classification-using-multi/Dataset/Dataset/drone_{drone}/{drone}/test/{mic}"
        
        # Load audio and labels
        X_test.append(load_audio_file(data_dir))
        y_test_fault.append(training_fault(data_dir))
        y_test_model_name.append(training_model_name(data_dir))
        y_test_man_dir.append(training_man_direction(data_dir))

# Concatenate test data
X_test = np.concatenate(X_test)
y_test_fault = np.concatenate(y_test_fault)
y_test_model_name = np.concatenate(y_test_model_name)
y_test_man_dir = np.concatenate(y_test_man_dir)


# In[57]:


y_test_man_dir_encoded = label_encoder.fit_transform(y_test_man_dir)
y_test_fault_encoded = label_encoder.fit_transform(y_test_fault)
y_test_model_name_encoded = label_encoder.fit_transform(y_test_model_name)


# In[58]:


y_test_man_dir_encoded = to_categorical(y_test_man_dir_encoded.reshape(-1, 1), num_classes=len(set(y_test_man_dir)))
y_test_fault_encoded = to_categorical(y_test_fault_encoded.reshape(-1, 1), num_classes=len(set(y_test_fault)))
y_test_model_name_encoded = to_categorical(y_test_model_name_encoded.reshape(-1, 1), num_classes=len(set(y_test_model_name)))


# In[97]:


model = training_model(X_train, y_train_man_dir_encoded, y_train_fault_encoded, y_train_model_name_encoded)


# In[98]:


X_test_reshaped = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)


test_loss, main_output_test_accuracy, aux_output_fault_test_accuracy, aux_output_model_test_accuracy = model.evaluate(X_test_reshaped, 
                                                                                                                             [y_test_man_dir_encoded, y_test_fault_encoded, y_test_model_name_encoded])

print(f'Main Output Test Accuracy: {main_output_test_accuracy}')
print(f'Auxiliary Output 1 Test Accuracy: {aux_output_fault_test_accuracy}')
print(f'Auxiliary Output 2 Test Accuracy: {aux_output_model_test_accuracy}')


# In[99]:


X_test_reshaped = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

predictions = model.predict(X_test_reshaped)

predictions_main_output = predictions[0]

predictions_aux_output_fault = predictions[1]

predictions_aux_output_model_name = predictions[2]


# In[100]:


predictions_main_output


# In[101]:


all_unique_labels = np.unique(y_train_man_dir)
label_encoder.fit(all_unique_labels)
predicted_labels = np.argmax(predictions_main_output, axis=1)
predicted_man_dir_inverse = label_encoder.inverse_transform(predicted_labels)


# In[102]:


all_unique_labels = np.unique(y_train_fault)
label_encoder.fit(all_unique_labels)
predicted_labels_fault = np.argmax(predictions_aux_output_fault, axis=1)
predicted_fault_inverse = label_encoder.inverse_transform(predicted_labels_fault)


# In[103]:


all_unique_labels_model_name = np.unique(y_train_model_name)
label_encoder.fit(all_unique_labels_model_name)
predicted_labels_model_name = np.argmax(predictions_aux_output_model_name, axis=1)
predicted_model_name_inverse = label_encoder.inverse_transform(predicted_labels_model_name)


# In[128]:


mic1_paths = [
    "/kaggle/input/sound-based-drone-fault-classification-using-multi/Dataset/Dataset/drone_A/A/test/mic1",
    "/kaggle/input/sound-based-drone-fault-classification-using-multi/Dataset/Dataset/drone_B/B/test/mic1",
    "/kaggle/input/sound-based-drone-fault-classification-using-multi/Dataset/Dataset/drone_C/C/test/mic1"
]

mic2_paths = [
    "/kaggle/input/sound-based-drone-fault-classification-using-multi/Dataset/Dataset/drone_A/A/test/mic2",
    "/kaggle/input/sound-based-drone-fault-classification-using-multi/Dataset/Dataset/drone_B/B/test/mic2",
    "/kaggle/input/sound-based-drone-fault-classification-using-multi/Dataset/Dataset/drone_C/C/test/mic2"
]

all_labels = []

for test_dir in mic1_paths:
    mic_info = "mic1"
    for filename in os.listdir(test_dir):
        if filename.endswith(".wav"):
            id_, extension = os.path.splitext(filename)
            parts = id_.split("_")
            label_parts = parts[:-1]
            full_label = "_".join(label_parts) + extension
            all_labels.append(full_label)

for test_dir in mic2_paths:
    mic_info = "mic2"
    for filename in os.listdir(test_dir):
        if filename.endswith(".wav"):
            id_, extension = os.path.splitext(filename)
            parts = id_.split("_")
            label_parts = parts[:-1]
            full_label = "_".join(label_parts) + extension
            all_labels.append(full_label)



# In[129]:


import pandas as pd


# In[130]:


submission_df = pd.DataFrame({
    "ID": all_labels,
    "model_type": predicted_model_name_inverse,
    "maneuvering_direction": predicted_man_dir_inverse,
    "fault": predicted_fault_inverse
})


# In[133]:


submission_df.to_csv('submission1.csv', index=False)


# In[134]:


print(submission_df)


# In[ ]:




