# Example code of how to downsample or upsample the ECG as some are measured at 250 Hz and others at 500 Hz

def upsample_5k_single(input_array,output_array):
    assert input_array.shape == (2500, 12, 1), f"input_array is shape {input_array.shape} not (2500, 12, 1)"
    processed_data = cv2.resize(input_array.squeeze(),(input_array.shape[1],input_array.shape[0]*2),interpolation=cv2.INTER_LINEAR)
    output_array.append(processed_data)


def upsample_5k_batch(input_array,output_array):
    a=0
    time1 = time.time()
    assert input_array.shape[1:] == (2500, 12, 1), f"input_array is shape {input_array.shape[1:]} not (2500, 12, 1)"
    for ecg in input_array:
        a+=1
        processed_data = cv2.resize(ecg.squeeze(),(ecg.shape[1],ecg.shape[0]*2),interpolation=cv2.INTER_LINEAR)
        output_array.append(processed_data)
        if a%1000==0:
            print(a,'ECGs converted')
            print(time.time()-time1,"seconds since start")
    output_array = np.expand_dims(output_array, axis=3)
    print(output_array.shape)


def downsample_5k_batch(input_array,output_array):
    a=0
    time1 = time.time()
    assert input_array.shape[1:] == (5000, 12, 1), f"input_array is shape {input_array.shape[1:]} not (5000, 12, 1)"
    for ecg in input_array:
        a+=1
        processed_data = cv2.resize(ecg.squeeze(),(ecg.shape[1],ecg.shape[0]*0.5),interpolation=cv2.INTER_LINEAR)
        output_array.append(processed_data)
        if a%1000==0:
            print(a,'ECGs converted')
            print(time.time()-time1,"seconds since start")
    output_array = np.expand_dims(output_array, axis=3)
    print(output_array.shape)
