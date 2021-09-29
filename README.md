# Robust Shadow Detection by Exploring Effective Shadow Contexts

This repository contains Pytorch code for the paper titled "Robust Shadow Detection by Exploring Effective Shadow Contexts" by  Xianyong Fang, et al. at ACM Multimedia 2021.

### Architecture

Attached below is the architecture diagram as given in the paper.
![network](img/pipeline.jpg)

### Requirements

- Pytorch
- Python3.X
- numpy
- cv2
- PIL

### Usage

- You can search and download the datasets from the Internet.
- ResNext101 has been adopted，and you can download the ResNet101's settings from [here](https://drive.google.com/drive/folders/1qBivnosrTb1PUnB2i89t27oKmSbmDaqP?usp=sharing)，you can put it in the `./` directory.

### Training

```python
python train.py
```

### Testing

```python
python test.py
```



### Results
![results](img/results.jpg)
(Left to right: Input, ground truth, detection result)

More results can be downloaded [here](https://drive.google.com/drive/folders/1OCs8usYDHB2oqNtsZqR5Q8qDXXNjaYWy?usp=sharing).

### Trained model

You can download from [here](https://drive.google.com/drive/folders/1uQmR-Gg16kEKvf-qFcH0syHOlBJKAQgY?usp=sharing).

### References

- Xianyong Fang, Xiaohao He, Linbo Wang, Jianbing Shen, Robust Shadow Detection by Exploring Effective Shadow Contexts, ACM Multimedia 2021. 
