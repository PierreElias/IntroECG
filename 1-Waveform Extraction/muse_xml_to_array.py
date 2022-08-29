import pandas as pd 
import numpy as np 
import xmltodict
import base64
import struct
import argparse
import os
import sys



def file_path(path):
    filepath = path
    for dirName, subdirList, fileList in os.walk(filepath):
        for filename in fileList:
            if ".xml" in filename.lower():
                ekg_file_list.append(os.path.join(dirName, filename))
    
#need to update this function to check the output directory for the output file and then only on newly added EKGs
#add timestamp to start file string 
#this is annoying because the XML file name is a random timestamp and the output file is the UniqueECGID


if not os.path.exists(os.getcwd() + '/ekg_waveforms_output/'):
    os.mkdir(os.getcwd() + '/ekg_waveforms_output/')

# parser = argparse.ArgumentParser(description='Input and outputs for XML EKG parsing')
# parser.add_argument('input', type=str)
# parser.set_defaults(output=os.getcwd() + '/ekg_waveforms_output/') #ensure this directory already exists

# args = parser.parse_args()



def decode_ekg_muse(raw_wave):
    """
    Ingest the base64 encoded waveforms and transform to numeric
    """
    # covert the waveform from base64 to byte array
    arr = base64.b64decode(bytes(raw_wave, 'utf-8'))

    # unpack every 2 bytes, little endian (16 bit encoding)
    unpack_symbols = ''.join([char*int(len(arr)/2) for char in 'h'])
    byte_array = struct.unpack(unpack_symbols,  arr)
    return byte_array


def decode_ekg_muse_to_array(raw_wave, downsample = 1):
    """
    Ingest the base64 encoded waveforms and transform to numeric

    downsample: 0.5 takes every other value in the array. Muse samples at 500/s and the sample model requires 250/s. So take every other.
    """
    try:
        dwnsmpl = int(1//downsample)
    except ZeroDivisionError:
        print("You must downsample by more than 0")
    # covert the waveform from base64 to byte array
    arr = base64.b64decode(bytes(raw_wave, 'utf-8'))

    # unpack every 2 bytes, little endian (16 bit encoding)
    unpack_symbols = ''.join([char*int(len(arr)/2) for char in 'h'])
    byte_array = struct.unpack(unpack_symbols,  arr)
    return np.array(byte_array)[::dwnsmpl]



def xml_to_np_array_file(path_to_xml, path_to_output = os.getcwd()):

    with open(path_to_xml, 'rb') as fd:
        dic = xmltodict.parse(fd.read().decode('utf8'))

    """

    Upload the ECG as numpy array with shape=[2500,12,1] ([time, leads, 1]).

    The voltage unit should be in 1 mv/unit and the sampling rate should be 250/second (total 10 second).

    The leads should be ordered as follow I, II, III, aVR, aVL, aVF, V1, V2, V3, V4, V5, V6.

    """
    try:
        pt_id = dic['RestingECG']['PatientDemographics']['PatientID']
    except:
        print("no PatientID")
        pt_id = "none"
    try:
        PharmaUniqueECGID = dic['RestingECG']['PharmaData']['PharmaUniqueECGID']
    except:
        print("no PharmaUniqueECGID")
        PharmaUniqueECGID = "none"
    try:
        AcquisitionDateTime = dic['RestingECG']['TestDemographics']['AcquisitionDate'] + "_" + dic['RestingECG']['TestDemographics']['AcquisitionTime'].replace(":","-")
    except:
        print("no AcquisitionDateTime")
        AcquisitionDateTime = "none"    
        

    # try:
    #     requisition_number = dic['RestingECG']['Order']['RequisitionNumber']
    # except:
    #     print("no requisition_number")
    #     requisition_number = "none"

    #need to instantiate leads in the proper order for the model
    lead_order = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']

    """
    Each EKG will have this data structure:
    lead_data = {
        'I': np.array
    }
    """

    lead_data =  dict.fromkeys(lead_order)
    #lead_data = {leadid: None for k in lead_order}

#     for all_lead_data in dic['RestingECG']['Waveform']:
#         for single_lead_data in lead['LeadData']:
#             leadname =  single_lead_data['LeadID']
#             if leadname in (lead_order):

    for lead in dic['RestingECG']['Waveform']:
        for leadid in range(len(lead['LeadData'])):
                sample_length = len(decode_ekg_muse_to_array(lead['LeadData'][leadid]['WaveFormData']))
                #sample_length is equivalent to dic['RestingECG']['Waveform']['LeadData']['LeadSampleCountTotal']
                if sample_length == 5000:
                    lead_data[lead['LeadData'][leadid]['LeadID']] = decode_ekg_muse_to_array(lead['LeadData'][leadid]['WaveFormData'], downsample = 0.5)
                elif sample_length == 2500:
                    lead_data[lead['LeadData'][leadid]['LeadID']] = decode_ekg_muse_to_array(lead['LeadData'][leadid]['WaveFormData'], downsample = 1)
                else:
                    continue
            #ensures all leads have 2500 samples and also passes over the 3 second waveform

    lead_data['III'] = (np.array(lead_data["II"]) - np.array(lead_data["I"]))
    lead_data['aVR'] = -(np.array(lead_data["I"]) + np.array(lead_data["II"]))/2
    lead_data['aVF'] = (np.array(lead_data["II"]) + np.array(lead_data["III"]))/2
    lead_data['aVL'] = (np.array(lead_data["I"]) - np.array(lead_data["III"]))/2
    
    lead_data = {k: lead_data[k] for k in lead_order}
    # drops V3R, V4R, and V7 if it was a 15-lead ECG

    # now construct and reshape the array
    # converting the dictionary to an np.array
    temp = []
    for key,value in lead_data.items():
        temp.append(value)

    #transpose to be [time, leads, ]
    ekg_array = np.array(temp).T

    #expand dims to [time, leads, 1]
    ekg_array = np.expand_dims(ekg_array,  axis=-1)

    # Here is a check to make sure all the model inputs are the right shape
#     assert ekg_array.shape == (2500, 12, 1), "ekg_array is shape {} not (2500, 12, 1)".format(ekg_array.shape )

    # filename = '/ekg_waveform_{}_{}.npy'.format(pt_id, requisition_number)
    filename = '{}_{}_{}.npy'.format(pt_id, AcquisitionDateTime,PharmaUniqueECGID)

    path_to_output += filename
    # print(path_to_output)
    with open(path_to_output, 'wb') as f:
        np.save(f, ekg_array)
        

def ekg_batch_run(ekg_list):
    i = 0
    x = 0
    for file in ekg_list:
        try:
            xml_to_np_array_file(file, output_dir)
            i+=1
        except Exception as e:
            # print("file failed: ", file)
            print(file, e)
            x+=1
        if i % 10000 == 0:
            print(f"Succesfully converted {i} EKGs, failed converting {x} EKGs")

output_dir = os.getcwd() + '/ekg_waveforms_output/'
print("args", sys.argv)
ekg_file_list = []
file_path(sys.argv[1])  #if you want input to be a directory
print("Number of EKGs found: ", len(ekg_file_list))

ekg_batch_run(ekg_file_list)


# To reconstruct the 12 lead ecg from the array
# test1 = np.load('waveform_output_example.npy')
# lead_order = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
# plt.rcParams["figure.figsize"] = [16,9]
# fig, axs = plt.subplots(len(lead_data))
# for i in range(0,12):
#     axs[i].plot(test1[:,i])
#     axs[i].set(ylabel=str(lead_order[i]))



# To find paced EKGs will use below, but work in progress
# dx_txt = [] 
# for line in dic['RestingECG']['Diagnosis']['DiagnosisStatement']:
#     dx_txt.append(line['StmtText'])
# print(dx_txt)
