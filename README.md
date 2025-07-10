# uwpose
Repository for TMR4930: Marine Technology, Master's Thesis at NTNU

This works compares the performance of two methods: DeepURL(Joshi et al.) and Onepose++(He et al.) on the DFKI docking station dataset. 

To install the pipelines, please follow their respective repositories.

1. DeepURL: https://github.com/joshi-bharat/deep_underwater_localization
2. Onepose++: https://github.com/zju3dv/OnePose_Plus_Plus

The Docking Station Dataset is available on the following link. 
https://zenodo.org/records/13928144

To avoid version conflict, it is recommend to use Docker environment for DeepURL. The Docker file for the environment is available in this repository.(WIP)

To reproduce the result on the thesis, please follow these steps.
1. Install the pipelines and dataset 
2. Set up the Docker environment for DeepURL.
3. Inference the Onepose++ pipeline on the docking staion dataset. Save the estimated pose. This work uses the original pretrained weights. 
4. in the Docker enviroment created in 2., run 
```
python eval_oneposeplus.py
```
6. DOwnload the pretrained DeepURL weights at: https://drive.google.com/file/d/1Lwp61foQ7bFwnPYg5n74rJTpiuv4pBqo/view?usp=sharing
5. To run the inference of DeepURL, run 
```
python test_image_list_docking_station.py --image_list path_to_test_txt_file --checkpoint_dir path_to_weight_dir 
```
This will evaluate the model and report the metrices. An Inference plot on each image can be obtained by uncommenting the respective lines. 

6. Save the video of the inference by executing vdo_oneposeplus.ipynb

# Integration of SIFT point Clouds into Onepose++

WIP

# References 
Please cite the following works if this repository is reproduced for your research. 

Onepose++
```
@inproceedings{
    he2022oneposeplusplus,
    title={OnePose++: Keypoint-Free One-Shot Object Pose Estimation without {CAD} Models},
    author={Xingyi He and Jiaming Sun and Yuang Wang and Di Huang and Hujun Bao and Xiaowei Zhou},
    booktitle={Advances in Neural Information Processing Systems},
    year={2022}
}
```
DeepURL
```
@misc{joshi2020deepurl,
    title={DeepURL: Deep Pose Estimation Framework for Underwater Relative Localization},
    author={Bharat Joshi and Md Modasshir and Travis Manderson and Hunter Damron and Marios Xanthidis and Alberto Quattrini Li and Ioannis Rekleitis and Gregory Dudek},
    year={2020},
    archivePrefix={arXiv}
}
```
COLMAP(SIFT Point Clouds)
```
@INPROCEEDINGS{7780814,
  author={Schönberger, Johannes L. and Frahm, Jan-Michael},
  booktitle={2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR)}, 
  title={Structure-from-Motion Revisited}, 
  year={2016},
  volume={},
  number={},
  pages={4104-4113},
  keywords={Image reconstruction;Robustness;Cameras;Internet;Image registration;Transmission line matrix methods;Pipelines},
  doi={10.1109/CVPR.2016.445}}
  
```