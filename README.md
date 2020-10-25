## Space-time Memory Networks for Video Object Segmentation with User Guidance
### Seoung Wug Oh, Joon-Young Lee, Ning Xu, Seon Joo Kim
### PAMI 
[[pami paper]](https://www.dropbox.com/s/z4vfeb5e68catwi/PAMI_STM_Final.pdf?dl=1)
[[iccv19 paper]](http://openaccess.thecvf.com/content_ICCV_2019/html/Oh_Video_Object_Segmentation_Using_Space-Time_Memory_Networks_ICCV_2019_paper.html)



### - Requirements
- python 3.6
- pytorch 1.0.1.post2
- numpy, opencv, pillow, matplotlib
- PyQt5
- qdarkstyle
- davisinteractive
- etc

### - How to Use
#### Download weights
##### Place it the same folder with demo scripts
```
wget -O e120.pth "https://www.dropbox.com/s/nvor7d4d1wk8nte/e120.pth?dl=1"
```

#### Run
``` 
python gui.py 
```


### - Quantitative Evaluation
This repository contains a software only for demonstration. For the quantitative evaluation, we used the [DAVIS framework](https://interactive.davischallenge.org/).
For the comparison with our model used for DAVIS Interactive VOS Challenge 2019 (https://davischallenge.org/challenge2019/interactive.html), please use evaluation summary obtained from the [DAVIS framework](https://interactive.davischallenge.org/). 
[[Download link (DAVIS-17-val)]](https://www.dropbox.com/s/owoms3rtalg52wn/STM_Interactive_summary_DAVIS17_val.json?dl=1).
The timing in the paper is measured using a single 2080Ti GPU.


### - Reference 
If you find our paper and repo useful, please cite our paper. Thanks!
``` 
Space-time Memory Networks for Video Object Segmentation with User Guidance
Seoung Wug Oh, Joon-Young Lee, Ning Xu, Seon Joo Kim
PAMI
```
``` 
Video Object Segmentation using Space-Time Memory Networks
Seoung Wug Oh, Joon-Young Lee, Ning Xu, Seon Joo Kim
ICCV 2019
```


### - Related Project
``` 
Fast User-Guided Video Object Segmentation by Interaction-and-Propagation Networks
Seoung Wug Oh, Joon-Young Lee, Ning Xu, Seon Joo Kim
CVPR 2019
```
[[paper]](https://openaccess.thecvf.com/content_CVPR_2019/papers/Oh_Fast_User-Guided_Video_Object_Segmentation_by_Interaction-And-Propagation_Networks_CVPR_2019_paper.pdf)
[[github]](https://github.com/seoungwugoh/ivs-demo)




### - Terms of Use
This software is for non-commercial use only.
The source code is released under the Attribution-NonCommercial-ShareAlike (CC BY-NC-SA) Licence
(see [this](https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode) for details)
