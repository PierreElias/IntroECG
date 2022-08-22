
[GE_Muse_Waveform_Extraction_Tutorial.ipynb](https://github.com/perottelab/IntroECG/blob/main/1-Waveform%20Extraction/GE_Muse_Waveform_Extraction_Tutorial.ipynb) shows how to extract waveform data from the commonly used software GE Muse. ECGs must be printed as .xml files from within the program, and then are stored as a nested dictionary with the raw waveform data embedded in base-64 encoding. We provide an example .xml output and show how to extract the waveforms. 

[muse_xml_to_array.py](https://github.com/perottelab/IntroECG/blob/main/1-Waveform%20Extraction/muse_xml_to_array.py) is a script that automates that process to run on batches of EKGs. 

[ecg_pdf_to_dataframe.py](https://github.com/perottelab/IntroECG/blob/main/1-Waveform%20Extraction/ecg_pdf_to_dataframe.py) shows how to extract vector waveforms from EKGs saved as PDFs (often times the only way people may have access to it), extracting waveforms as SVGs and converting them to .npy arrays. 

[Visualizing_ECGs.ipynb](Visualizing_ECGs.ipynb) shows you how to plot and look at the ECGs themselves.

[Create XML EKG.docx](https://github.com/PierreElias/IntroECG/blob/master/1-Waveform%20Extraction/Create%20XML%20EKG.docx) is a word document that explains how to set up extraction via XML as well automating export for future ECGs. 
