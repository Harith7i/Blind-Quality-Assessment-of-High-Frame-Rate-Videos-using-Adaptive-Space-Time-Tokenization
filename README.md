# Blind Quality Assessment of High Frame Rate Videos using Adaptive Space-Time Tokenization

[![License](https://img.shields.io/github/license/mashape/apistatus.svg?maxAge=2592000)](LICENSE)

Blind Quality Assessment of High Frame Rate Videos using Adaptive Space-Time Tokenization


<p align="center">
  <img src="https://github.com/Harith7i/Blind-Quality-Assessment-of-High-Frame-Rate-Videos-using-Adaptive-Space-Time-Tokenization/blob/main/Figures/End_To_End_Model.png">
</p>


This repository contains the code of our project [Blind Quality Assessment of High Frame Rate Videos using Adaptive Space-Time Tokenization]. 



  * [Requirements](#requirements)
  * [Model Training](#model-training)
  * [Test](#test)
      * [On LIVE HFR VQA](#a-on-LIVE-HFR-VQA)
      * [On BVI HFR VQA](#b-on-BVI-HFR-VQA)
      * [Cross Validation onHFR datasets](#b-Cross-Validation-on-HFR-datasets)
  * [Demo](#demo)
  * [Evaluate](#evaluate)
  * [Performance Benchmark](#performance-benchmark)
  * [References](#references)
    
<!-- /code_chunk_output -->



## Requirements
```python
pip install -r requirements.txt
```


## Model Training (optional):

In order to traing the model, you need to run the folowing command specify the model parameters and datasets paths.
The datasets have been treated with care to avoid any content overlapping.

To train your own model:



```python
add the python command                                         
```                              


## Test: 


In the following section, we provide two tables to resume the results on two dataset

#### a-LIVE-HFR-VQA:


|    Methods   |SROCC            | PLCC               | RMSE |
|:------------:|:---------------------:|:-------------------:|:------------:|
| L2VQA     | 0.9187         | 0.9171           | 5.37  |  


<p align="center">
  <img src="https://github.com/Harith7i/Blind-Quality-Assessment-of-High-Frame-Rate-Videos-using-Adaptive-Space-Time-Tokenization/blob/main/Figures/scatter%20for%20LVYTHFRcc.png">
</p>

#### b-On LIVE_VQC: 


|    Methods   |SROCC            | PLCC               | RMSE |
|:------------:|:---------------------:|:--------------------:|:------------:|
| L2VQA   | 0.9288  | 0.9312      | 6.83 |


<p align="center">
  <img src="https://github.com/Harith7i/Blind-Quality-Assessment-of-High-Frame-Rate-Videos-using-Adaptive-Space-Time-Tokenization/blob/main/Figures/scatterbvi.png">
</p>



## Demo:

To predict the quality of your own dataset using pre-trained model:

```python

--add demo here

```

## Evaluate:

To evaluate the model:

Please note that your csv file should have two columns: 'Mos' and 'Predicted'.

```python
python evaluate.py  --mos_pred konvid.csv
```



## Performance Benchmark:


###### LIVE HFR VQA[1]:

<p align="center">
  <img src="https://github.com/Harith7i/Blind-Quality-Assessment-of-High-Frame-Rate-Videos-using-Adaptive-Space-Time-Tokenization/blob/main/Figures/LIVE_results.png">
</p>


###### BVI HFR VQA [2]:

<p align="center">
  <img src="https://github.com/Harith7i/Blind-Quality-Assessment-of-High-Frame-Rate-Videos-using-Adaptive-Space-Time-Tokenization/blob/main/Figures/BVI_results.png">
</p>

###### Cross Validation on HFR datasets [2]:
<p align="center">
  <img src="https://github.com/Harith7i/Blind-Quality-Assessment-of-High-Frame-Rate-Videos-using-Adaptive-Space-Time-Tokenization/blob/main/Figures/Cross_Validation.png">
</p>







## References


```
[1] V. Hosu, F. Hahn, M. Jenadeleh, H. Lin, H. Men, T. Szirányi, S. Li,and D. Saupe, “The konstanz natural video database (konvid-1k),” in2017 Ninth international conference on quality of multimedia experience(QoMEX).  IEEE, 2017, pp. 1–6.

[2] Z. Sinno and A. C. Bovik, “Large-scale study of perceptual videoquality,”IEEE Transactions on Image Processing, vol. 28, no. 2, pp.612–627, 2018.

[3] Y. Wang, S. Inguva, and B. Adsumilli, “Youtube ugc dataset for videocompression research,” in2019 IEEE 21st International Workshop onMultimedia Signal Processing (MMSP).  IEEE, 2019, pp. 1–5.
```



