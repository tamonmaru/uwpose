# uwpose
Repository for TMR4930: Marine Technology, Master's Thesis at NTNU

This works compares the performance of two methods: DeepURL(Joshi et al.) and Onepose++(He et al.) on the DFKI docking station dataset. 

To install the pipelines, please follow their respective repositories.
1.https://github.com/joshi-bharat/deep_underwater_localization
2.https://github.com/zju3dv/OnePose_Plus_Plus

The Docking Station Dataset is available on the following link. 
https://zenodo.org/records/13928144

To avoid version conflict, it is recommend to use Docker environment for DeepURL. The Docker file for the environment is available in this repository.(WIP)

To reproduce the result on the thesis, please follow these steps.
1. Install the pipelines and dataset 
2. Set up the Docker environment for DeepURL.
3. Inference the Onepose++ pipeline on the docking staion dataset. Save the estimated pose 
4. in the Docker enviroment created in 2., run 
```
python eval_oneposeplus.py
```
5. To run the inference of DeepURL, run 
```
python test_image_list_docking_station.py --image_list path_to_test_txt_file --checkpoint_dir path_to_checkpoint_dir 
```