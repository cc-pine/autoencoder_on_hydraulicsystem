import numpy as np
from sklearn import preprocessing 

def pre_processing(Normalized=True):
    variables=["Index", "PS1", "PS2", "PS3", "PS4", "PS5", "PS6", "EPS1",
            "FS1", "FS2", "TS1", "TS2", "TS3", "TS4", "VS1", "CE", "CP",
            "SE"]
    #PS1~EPS1 100Hz FS 10Hz else 1Hz

    paths = [x+".txt" for x in variables]
    raw_data = {}

    for i,path in enumerate(paths[1:]):
        with open(path) as f:
            data = []
            for line in f:
                l = line.split()
                l = [float(x) for x in l]
                data.append(l)
        raw_data[variables[i+1]] = np.array(data)


    processed_data={}
    scaler = preprocessing.StandardScaler()
    if Normalized:
        for variable, data in raw_data.items():
            if data.shape[1]==6000:
                processed_data[variable] = scaler.fit_transform(data)

            elif data.shape[1]==600:
                processed = np.zeros((2205, 6000))
                data = scaler.fit_transform(data)
                for i, module in enumerate(data):
                    for j, value in enumerate(module):
                        processed[i, 10*j:10*j+10] = value
                processed_data[variable] = processed

            elif data.shape[1]==60:
                processed = np.zeros((2205, 6000))
                data = scaler.fit_transform(data)
                for i, module in enumerate(data):
                    for j, value in enumerate(module):
                        processed[i, 100*j:100*j+100]=value
                processed_data[variable] = processed

            else:
                print("shape error")
                break

    else:
        for variable, data in raw_data.items():
            if data.shape[1]==6000:
                processed_data[variable] = data

            elif data.shape[1]==600:
                processed = np.zeros((2205, 6000))
                for i, module in enumerate(data):
                    for j, value in enumerate(module):
                        processed[i, 10*j:10*j+10] = value
                processed_data[variable] = processed

            elif data.shape[1]==60:
                processed = np.zeros((2205, 6000))
                for i, module in enumerate(data):
                    for j, value in enumerate(module):
                        processed[i, 100*j:100*j+100]=value
                processed_data[variable] = processed

            else:
                print("shape error")
                break

    return processed_data


def get_profile():
    with open("profile.txt") as f:
        profile = []
        for line in f:
            l = line.split()
            l = [int(x) for x in l]
            profile.append(l)

    return np.array(profile)

def get_variable_profile_set(Normalized=True):
    variable_data = pre_processing(Normalized=Normalized)
    profile = get_profile()
    variable_profile_set = []
    for i in range(2205):
        v = []
        p = profile[i]
        for _, data in variable_data.items():
            v.append(data[i])
        variable_profile_set.append((np.array(v),p))

    return variable_profile_set


