from pathlib import Path
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import matplotlib.pyplot as plt
import librosa
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

BASE_DIR=Path(__file__).resolve().parent
DATA_DIR=BASE_DIR/'data'/'ravdess'
OUTPUT_DIR=BASE_DIR/'outputs'; OUTPUT_DIR.mkdir(exist_ok=True)
SAMPLE_RATE=16000; DURATION=3.0; N_MFCC=40; TEST_SIZE=.20; RANDOM_STATE=42; EPOCHS=25; BATCH_SIZE=32
EMOTIONS={'01':'neutral','02':'calm','03':'happy','04':'sad','05':'angry','06':'fearful','07':'disgust','08':'surprised'}

def emotion_from_filename(path):
    parts=path.stem.split('-')
    return EMOTIONS.get(parts[2]) if len(parts)==7 else None

def extract_features(path):
    audio,sr=librosa.load(path,sr=SAMPLE_RATE,duration=DURATION,mono=True)
    target=int(SAMPLE_RATE*DURATION)
    if len(audio)<target: audio=np.pad(audio,(0,target-len(audio)))
    else: audio=audio[:target]
    mfcc=librosa.feature.mfcc(y=audio,sr=sr,n_mfcc=N_MFCC,n_fft=1024,hop_length=512)
    delta=librosa.feature.delta(mfcc); delta2=librosa.feature.delta(mfcc,order=2)
    x=np.stack([mfcc,delta,delta2],axis=-1)
    x=(x-x.mean(axis=(0,1),keepdims=True))/(x.std(axis=(0,1),keepdims=True)+1e-8)
    return x.astype('float32')

def build_model(shape,nclasses):
    model=tf.keras.Sequential([
        tf.keras.layers.Input(shape=shape),
        tf.keras.layers.Conv2D(32,(3,3),activation='relu',padding='same'),
        tf.keras.layers.BatchNormalization(),tf.keras.layers.MaxPooling2D((2,2)),tf.keras.layers.Dropout(.25),
        tf.keras.layers.Conv2D(64,(3,3),activation='relu',padding='same'),
        tf.keras.layers.BatchNormalization(),tf.keras.layers.MaxPooling2D((2,2)),tf.keras.layers.Dropout(.25),
        tf.keras.layers.Conv2D(128,(3,3),activation='relu',padding='same'),tf.keras.layers.BatchNormalization(),
        tf.keras.layers.GlobalAveragePooling2D(),tf.keras.layers.Dense(128,activation='relu'),tf.keras.layers.Dropout(.35),
        tf.keras.layers.Dense(nclasses,activation='softmax')])
    model.compile(optimizer=tf.keras.optimizers.Adam(.001),loss='sparse_categorical_crossentropy',metrics=['accuracy'])
    return model

def main():
    files=sorted(DATA_DIR.rglob('*.wav'))
    if not files: raise FileNotFoundError('No WAV files found. Put extracted RAVDESS speech WAV files inside data/ravdess/.')
    X=[]; labels=[]; used=[]
    print(f'Found {len(files)} WAV files.')
    for i,p in enumerate(files,1):
        emotion=emotion_from_filename(p)
        if emotion is None: continue
        try: X.append(extract_features(p)); labels.append(emotion); used.append(str(p.relative_to(BASE_DIR)))
        except Exception as e: print('Skipped',p.name,':',e)
        if i%100==0: print(f'Processed {i}/{len(files)}')
    if len(X)<20: raise ValueError('Too few valid RAVDESS files found.')
    X=np.asarray(X,dtype='float32'); enc=LabelEncoder(); y=enc.fit_transform(labels)
    Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=TEST_SIZE,random_state=RANDOM_STATE,stratify=y)
    print('Feature shape:',X.shape); print('Classes:',list(enc.classes_))
    model=build_model(Xtr.shape[1:],len(enc.classes_)); model.summary()
    callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss',patience=5,restore_best_weights=True),tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss',factor=.5,patience=2,min_lr=1e-6)]
    history=model.fit(Xtr,ytr,validation_split=.15,epochs=EPOCHS,batch_size=BATCH_SIZE,callbacks=callbacks,verbose=1)
    prob=model.predict(Xte,verbose=0); pred=np.argmax(prob,axis=1)
    acc=accuracy_score(yte,pred); pre=precision_score(yte,pred,average='weighted',zero_division=0); rec=recall_score(yte,pred,average='weighted',zero_division=0); f1=f1_score(yte,pred,average='weighted',zero_division=0)
    print(f'Accuracy: {acc:.4f}\nPrecision: {pre:.4f}\nRecall: {rec:.4f}\nF1-Score: {f1:.4f}')
    (OUTPUT_DIR/'model_metrics.txt').write_text(f'Accuracy: {acc:.4f}\nPrecision: {pre:.4f}\nRecall: {rec:.4f}\nF1-Score: {f1:.4f}\n')
    (OUTPUT_DIR/'classification_report.txt').write_text(classification_report(yte,pred,target_names=enc.classes_,zero_division=0))
    fig,ax=plt.subplots(figsize=(9,8)); ConfusionMatrixDisplay(confusion_matrix(yte,pred),display_labels=enc.classes_).plot(ax=ax,cmap='Blues',xticks_rotation=45,colorbar=False); ax.set_title('Speech Emotion Recognition - Confusion Matrix'); plt.tight_layout(); plt.savefig(OUTPUT_DIR/'01_confusion_matrix.png',dpi=180); plt.close()
    fig,ax=plt.subplots(figsize=(8,5)); ax.plot(history.history['accuracy'],label='Training Accuracy'); ax.plot(history.history['val_accuracy'],label='Validation Accuracy'); ax.set_title('CNN Training and Validation Accuracy'); ax.set_xlabel('Epoch'); ax.set_ylabel('Accuracy'); ax.legend(); plt.tight_layout(); plt.savefig(OUTPUT_DIR/'02_accuracy_curve.png',dpi=180); plt.close()
    fig,ax=plt.subplots(figsize=(8,5)); ax.plot(history.history['loss'],label='Training Loss'); ax.plot(history.history['val_loss'],label='Validation Loss'); ax.set_title('CNN Training and Validation Loss'); ax.set_xlabel('Epoch'); ax.set_ylabel('Loss'); ax.legend(); plt.tight_layout(); plt.savefig(OUTPUT_DIR/'03_loss_curve.png',dpi=180); plt.close()
    model.save(OUTPUT_DIR/'speech_emotion_cnn.keras'); np.save(OUTPUT_DIR/'class_names.npy',enc.classes_); (OUTPUT_DIR/'used_audio_files.txt').write_text('\n'.join(used))
    print('SUCCESS. Results saved in outputs/.')

if __name__=='__main__':
    try: main()
    except Exception as e: print('\nERROR:',type(e).__name__,e)
