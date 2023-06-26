# Blind Quality Assessment of High Frame Rate Videos using Adaptive Space-Time Tokenization

[![License](https://img.shields.io/github/license/mashape/apistatus.svg?maxAge=2592000)](LICENSE)

Blind Quality Assessment of High Frame Rate Videos using Adaptive Space-Time Tokenization


<p align="center">
  <img src="https://github.com/Harith7i/Blind-Quality-Assessment-of-High-Frame-Rate-Videos-using-Adaptive-Space-Time-Tokenization/blob/main/Figures/End_To_End_Model.png">
</p>


This repository contains the code of our project [Blind Quality Assessment of High Frame Rate Videos using Adaptive Space-Time Tokenization]. 



  * [Requirements](#requirements)
  * [Demo](#demo)
  * [Test](#test)
      * [On LIVE HFR VQA](#a-on-LIVE-HFR-VQA)
      * [On BVI HFR VQA](#b-on-BVI-HFR-VQA)
      * [Cross Validation onHFR datasets](#b-Cross-Validation-on-HFR-datasets)
  * [Performance Benchmark](#performance-benchmark)
  * [References](#references)
    
<!-- /code_chunk_output -->



## Requirements
```python
pip install -r requirements.txt
```


## Demo:

In order to traing the model, you need to run the folowing command specify the model parameters and datasets paths.
The datasets have been treated with care to avoid any content overlapping.

To train your own model:



```python
python3 main.py [-h] [-epochs E] [-dataset_type TYPE] [-dataset_csv_path CSV]
             [-video_path VP] [-spatial_pooling SP] [-embedding_size EMB_DIM]
             [-weights_folder WF] [-best_score_weights BSW]

optional arguments:
  -h, --help            show this help message and exit
  -epochs E, --e E      The total number of training epochs
  -dataset_type TYPE, --type TYPE
                        Set the variable to LIVE or BVI depending on your
                        choice or training dataset
  -dataset_csv_path CSV, --csv CSV
                        Set the path to dataset CSV
  -video_path VP, --vp VP
                        Set the path to the videos directory
  -spatial_pooling SP, --sp SP
                        Set aptial pooling to lstm, rnn, or gru
  -embedding_size EMB_DIM, --emb_dim EMB_DIM
                        Set the embedding sitz (the size of the transformer
                        token)
  -weights_folder WF, --wf WF
                        Set the path to the directory containing all training
                        weights
  -best_score_weights BSW, --bsw BSW
                        Set the path to the .pth containtnig the best weights
                                        
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



