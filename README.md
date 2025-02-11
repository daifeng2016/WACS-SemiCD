This repository is for a simple-yet-effective semi-supervised change detection method [Perturbation Matters: A Novel Approach for Semi-Supervised Remote Sensing Imagery Change Detection](https://www.mdpi.com/2072-4292/17/4/576) published in RS.
# Introduction
Due to the challenge of acquiring abundant labeled samples, semi-supervised change detection (SSCD) approaches are becoming increasingly popular in tackling CD tasks with limited labeled data. Despite their success, these methods tend to come with complex network architectures or cumbersome training procedures, which also ignore the domain gap between the labeled data and unlabeled data. Differently, we hypothesize that diverse perturbations are more favorable to exploit the potential of unlabeled data. In light of this spirit, we propose a novel SSCD approach based on Weak–strong Augmentation and Class-balanced Sampling (WACS-SemiCD). Specifically, we adopt a simple mean-teacher architecture to deal with labeled branch and unlabeled branch separately, where supervised learning is conducted on the labeled branch, while weak–strong consistency learning (e.g., sample perturbations’ consistency and feature perturbations’ consistency) is imposed for the unlabeled. To improve domain generalization capacity, an adaptive CutMix augmentation is proposed to inject the knowledge from the labeled data into the unlabeled data. A class-balanced sampling strategy is further introduced to mitigate class imbalance issues in CD. Particularly, our proposed WACS-SemiCD achieves competitive SSCD performance on three publicly available CD datasets under different labeled settings. Comprehensive experimental results and systematic analysis underscore the advantages and effectiveness of our proposed WACS-SemiCD.
# Main Framework

![GA](https://github.com/user-attachments/assets/03522aac-70f0-46cb-8982-cd850ee78bce)

# Main results 
![image](https://github.com/user-attachments/assets/6dce7298-62d4-4684-a187-a58747b3b213)
![image](https://github.com/user-attachments/assets/b60acf41-ec25-4ded-b233-e688b2c805d2)
![image](https://github.com/user-attachments/assets/15e99f74-62f0-4925-9871-1b4b48b4dbcc)
![image](https://github.com/user-attachments/assets/ece03faa-e970-4a31-aa8a-7375acd4ef21)
![image](https://github.com/user-attachments/assets/4c7df81a-de20-4abb-a39b-f824ddecf1c7)
![image](https://github.com/user-attachments/assets/611c4b92-3471-4b21-8ca2-acdf35f4913c)
![image](https://github.com/user-attachments/assets/12731fb9-807b-4191-ac67-b666af60b303)
![image](https://github.com/user-attachments/assets/42100f8c-69eb-465a-89c7-023f8a071f10)




# Citation
Please cite our paper if you find it is useful for your research.
```
@article{peng2025perturbation,
  title={Perturbation Matters: A Novel Approach for Semi-Supervised Remote Sensing Imagery Change Detection},
  author={Peng, Daifeng and Liu, Min and Guan, Haiyan},
  journal={Remote Sensing},
  volume={17},
  number={4},
  pages={576},
  year={2025},
  publisher={MDPI}
}
```
