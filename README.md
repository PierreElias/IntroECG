# IntroECG: A full-process library for deep learning on 12-lead electrocardiograms

This repository is meant to be a useful resource library for getting started with deep learning work using electrocardiograms. <br>

## 1-Waveform Extraction
<img align="right" width="60%" src="./Intro_ECG_ValveNet_Model_Full.png">

Scripts and tutorial for extracting raw ECG waveforms from GE Muse or PDFs of ECGs. It also includes examples of how to display and review your ECG data. 

## 2-Generating Synthetic ECG Data
Generate your own synthetic electrocardiograms. Comes with the ability to alter many different aspects of the waveform to test different hypotheses.

## 3-Preprocessing
Key preprocessing steps for cleaning and normalizing ECG data. 

## 4-Models
Different example models we've built to showcase approaches that work for electrocardiograms, in pytorch and tensorflow/keras.

## 5-Training with Ignite and Optuna
A framework built on PyTorch Ignite using Optuna to allow for rapid experimentation and displaying your results using Tensorboard

## 6-Putting it into Practice: The ValveNet Model
See the notebooks we used for generating our figures and key results on our valvular heart disease model published in JACC.

For more details, see the accompanying paper,

> [**Deep Learning Electrocardiographic Analysis for Detection of Left-Sided Valvular Heart Disease**](https://www.jacc.org/doi/10.1016/j.jacc.2022.05.029?utm_medium=email_newsletter&utm_source=jacc&utm_campaign=toc&utm_content=20220801#mmc1)<br/>
  Pierre Elias, Timothy J. Poterucha, Vijay Rajaram, Luca Matos Moller, Victor Rodriguez, Shreyas Bhave, Rebecca T. Hahn, Geoffrey Tison, Sean A. Abreau, Joshua Barrios, Jessica Nicole Torres, J. Weston Hughes, Marco V. Perez, Joshua Finer, Susheel Kodali, Omar Khalique, Nadira Hamid, Allan Schwartz, Shunichi Homma, Deepa Kumaraiah, David J. Cohen, Mathew S. Maurer, Andrew J. Einstein, Tamim Nazif, Martin B. Leon, and Adler J. Perotte. <b>Journal of the American College of Cardiology</b>, August 1, 2022. https://www.jacc.org/doi/10.1016/j.jacc.2022.05.029?utm_medium=email_newsletter&utm_source=jacc&utm_campaign=toc&utm_content=20220801#mmc1


## 7-Running Inference: The EchoNext Minimodel

To run EchoNext inference, follow the steps here: [README.md](7-EchoNext%20Minimodel/README.md)

For more details, see the accompanying paper,
> [**Detecting structural heart disease from electrocardiograms using AI**](https://www.nature.com/articles/s41586-025-09227-0)<br/>
Timothy J. Poterucha, Linyuan Jing, Ramon Pimentel Ricart, Michael Adjei-Mosi, Joshua Finer, Dustin Hartzel, Christopher Kelsey, Aaron Long, Daniel Rocha, Jeffrey A. Ruhl, David vanMaanen, Marc A. Probst, Brock Daniels, Shalmali D. Joshi, Olivier Tastet, Denis Corbin, Robert Avram, Joshua P. Barrios, Geoffrey H. Tison, I-Min Chiu, David Ouyang, Alexander Volodarskiy, Michelle Castillo, Francisco A. Roedan Oliver, Paloma P. Malta, Siqin Ye, Gregg F. Rosner, Jose M. Dizon, Shah R. Ali, Qi Liu, Corey K. Bradley, Prashant Vaishnava, Carol A. Waksmonski, Ersilia M. DeFilippis, Vratika Agarwal, Mark Lebehn, Polydoros N. Kampaktsis, Sofia Shames, Ashley N. Beecy, Deepa Kumaraiah, Shunichi Homma, Allan Schwartz, Rebecca T. Hahn, Martin Leon, Andrew J. Einstein, Mathew S. Maurer, Heidi S. Hartman, John Weston Hughes, Christopher M. Haggerty & Pierre Elias. <b>Nature</b>, July 16, 2025. https://doi.org/10.1038/s41586-025-09227-0


> PhysioNet dataset coming soon

## Development Team
Lead Developers:<br>
[-Pierre Elias](https://twitter.com/PierreEliasMD)<br>
[-Adler Perotte](https://twitter.com/aperotte)<br>

Contributors:<br>
-Vijay Rajaram<br>
-Shengqing Xia<br>
-Alex Wan<br>
-Junyang Jiang<br>
-Yuge Shen<br>
-Han Wang<br>
-Joshua Finer
