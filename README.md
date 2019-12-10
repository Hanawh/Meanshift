# Meanshift
Implement meanshift method for video tracking
## Dataset
You can download the OTB datasets from [here](http://cvlab.hanyang.ac.kr/tracker_benchmark/datasets.html). FaceOcc1 is used in my code. Of course I tried other datasets, but there are some problems with it.
## Show results
```
cd <YOUR_CODE_DIR>
python meanshift.py --mode show
```
## Train
You must ensure your datasets are in `data`
```
python meanshift.py --data_dir /data/<YOUR_DATA_DIR>
```

## Result
![image1](https://github.com/Hanawh/Meanshift/tree/master/result/FaceOcc1/00050.jpg)
![image2](https://github.com/Hanawh/Meanshift/tree/master/result/FaceOcc1/000100.jpg)
![image3](https://github.com/Hanawh/Meanshift/tree/master/result/FaceOcc1/000800.jpg)