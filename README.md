# Wearable-based Pain Assessment in Patients with Adhesive Capsulitis Using Machine Learning
> This project aimed to provide real-time and personal shoulder pain assessment using wearable devices and machine learning.
> full text [_pdf_](https://drive.google.com/file/d/1XXQIcL8pW0Peai1j7CgW7yS3x-9oJSYx/view?usp=share_link). <!-- If you have the project hosted somewhere, include the link here. -->

## Table of Contents
* [General Info](#general-information)
* [Technologies Used](#technologies-used)
* [Features](#features)
* [Acknowledgements](#acknowledgements)
* [Contact](#contact)
<!-- * [License](#license) -->


## General Information
Reliable shoulder function and pain assessment tools are important for the medical management in patients with adhesive capsulitis (AC). Several works have developed objective evaluation tools for shoulder function, but few of them focused on the pain. Currently, the pain assessment for AC still relies on the self-report by the patients. This pilot study proposed objective pain assessment tools for AC using inertial measurement units (IMUs) and machine learning (ML) approaches. The experiment recruited 23 AC patients who performed 5 shoulder tasks and reported pain scores based on the shoulder pain and disability index. Two IMUs were placed on the wrist and arm while performing shoulder tasks. We analyzed the correlations between pain scores and two IMU feature categories (e.g., smoothness, power, and speed). Moreover, we built pain prediction models with all extracted IMU features and different ML approaches. The results showed that the smoothness-related features have higher Spearman correlations with patient-reported pain scores than the category of power and speed features. ML- based pain prediction model using Gaussian process regression had strong and significant Spearman correlations (0.795,p<0.01), 5.680 mean absolute error, and 6.663 root mean square error.


## Technologies Used
- Python 3.9 

## Features
- using inertial measurement units (IMUs) to identify key demographic features.
- performed five shoulder tasks based on the SPADI, including placing an object on a high shelf (POH), washing hair (WH), washing the lower back (WLB), removing an object from the back pocket (ROP), and washing the upper back (WUB).
- several typical supervised regression and neural network (NN) approaches were applied to build a pain score predictor, including BRR, SVR with the radial basis kernel (stopping criteria= 0.001, 𝐶 = 1 ), GPR with the kernel function mixing dot product and white kernel (noise = 0.5), and MLP.


## Acknowledgements
- The results further suggested that WUB may be included as the main shoulder task for pain assessment.
- The ML-based pain prediction model using GPR presented a moderate-to-strong and significant Spearman correlation (0.795, 𝑝 < 0.01), MAE of 5.680, and RMSE of 6.663.
- In future studies, an increased number of patients with AC needs to be recruited to further validate the reliability of IMU-based pain assessment tools and ML-based pain prediction models.


## Contact
Created by [@chhsch](https://chhsch.github.io/projects/) - feel free to contact me!


<!-- Optional -->
<!-- ## License -->
<!-- This project is open source and available under the [... License](). -->

<!-- You don't have to include all sections - just the one's relevant to your project -->
