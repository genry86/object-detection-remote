# 📦 Computer Vision - Object detection - tv remote

Demo of Machine Learning to recognize 
tv-remote on the realtime iphone camera
and presenting red rect for recognized object(ig score is more than 0.5).  
4 models are trained with 4 different tools: Apple CreateML, Ultralytics, Pytorch, Tensorflow.

Note: some parts are not yet refactored and pythons' scripts for Pytorch, Tensorflow aren't fully finished. 

---

## 🚀 Features

- Pytorch. "pt" folder.
- Tensorflow. "tf" folder.
- Ultralytics tool.
- Create Ml app from Apple.
- Fully custom dataset. *Small for that kind of training.  
- Pillow. Working with images
- Numpy. To work with objects and arrays.
- Json. 
- Matplotlib. to visualize results 
- tqdm. to show progress bar for training process.
- CNN.
- Checkpoints. Used to restore last epoch if something happened within train process.
- ReduceLROnPlateau. Used to dynamically change Learning_rate.
- stop_counter. Used to stop training if training-metrics isn't improving.
- test_models.py. Tests and compare two models from TF and PT.
- convert_to_coreml.py. Converts both TF and PT models to CoreML(to use on the ios, osx)
- iOS UKKit apps used to work with converted CoreML and TFLite models.

---

## 🖼 Screenshots

###  iOS test 1

![Screen1](images/images_1.jpg)

### iOS test 2

![Screen2](images/images_2.jpg)

### iOS test 2

![Screen2](images/images_3.jpg)

---

## 🛠 Setup

```
## Python
git clone https://github.com/genry86/Classification_Face_Parts.git
cd Classification_Face_Parts
pip3 install -r requirements.txt

python data_download.py # to download CalebA dataset from kagglehub
python dataset_prepare.py # clean-up csv file and copy/split all photos to tran/val/test folders.

Use "pt" folder to work with pytorch. "train.py"
User "tf" to work with tensorflow. "train.py"

"test_models.py" file is used to show visual results of both generated models.
"convert_to_coreml.py" is used to convert both models to CoreML format, in order to be used on iOS phones.

 ## iOS
cd iOS
pod install
run ImageClassification.xcworkspace
# iOS will use real-time front camera to get image of your face to detect landmarks. Result photos will go to through trained models(Pytorch and Tensorflow). Results will be showed on the screen. 
