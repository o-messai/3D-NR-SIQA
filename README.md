# 3D-NR-SIQA
3D Saliency guided Deep Quality predictor for No-Reference Stereoscopic Images

# 3D Saliency guided Deep Quality predictor for No-Reference Stereoscopic Images
Paper title: "3D Saliency guided Deep Quality predictor for No-Reference Stereoscopic Images"
This code is the implementation of the proposed method called 3D-NR-SIQA Saliency.

# Abtract :

The use of 3D technologies is growing rapidly, and stereoscopic imaging is usually used to display the 3D contents. However, compression, transmission and other necessary treatments may reduce the quality of these images. Stereo Image Quality Assessment (SIQA) has attracted more attention to ensure good viewing experience for the users and thus several methods have been proposed in the literature with a clear improvement for deep learning-based methods. This paper introduces a new deep learning-based no-reference SIQA using cyclopean view hypothesis and human visual attention. First, the cyclopean image is constructed considering the presence of binocular rivalry that covers the asymmetric distortion case. Second, the saliency map is computed considering the depth information. The latter aims to extract patches on the most perceptual relevant regions. Finally, a modified version of the pre-trained Convolutional Neural Network (CNN) is fine-tuned and used to predict the quality score through the selected patches. Five distinct pre-trained models were analyzed and compared in term of results. The performance of the proposed metric has been evaluated on four commonly used datasets (3D LIVE phase I and phase II databases as well as Waterloo IVC 3D Phase 1 and Phase 2). Compared with the state-of-the-art metrics, the proposed method gives better outcomes. The implementation code will be made accessible to the public at: https://github.com/o-messai/3D-NR-SIQA

Publisher URL: https://www.sciencedirect.com/science/article/pii/S0925231222000029

DOI: 10.1016/j.image.2019.115772

## IQA datasets:

We have reported experimental results on different IQA datasets including LIVE 3D Phase I, LIVE 3D Phase II, Waterloo 3D Phase 1 and Phase 2.

# Citation :

If you use this code, we kindly ask you to cite the paper :
```
Messai, Oussama, Aladine Chetouani, Fella Hachouf, and Zianou Ahmed Seghir. "3D Saliency guided Deep Quality predictor for No-Reference Stereoscopic Images." Neurocomputing (2022).
```
# BibTex :
```
@article{messai20223d,
  title={3D Saliency guided Deep Quality predictor for No-Reference Stereoscopic Images},
  author={Messai, Oussama and Chetouani, Aladine and Hachouf, Fella and Seghir, Zianou Ahmed},
  journal={Neurocomputing},
  year={2022},
  publisher={Elsevier}
}
```




