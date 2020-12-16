import numpy as np
import os
def imu_feature_concate():
        feature_path = '/home/backup/data_cyr/assemble/features_imu_lr/'
        left = os.listdir(feature_path+'0')
        for file in left:
            file_name = file.split('.')[0][:-2]
            l = np.load(feature_path+'0/' + file_name+'_0.npy')
            r = np.load(feature_path+'1/' + file_name+'_1.npy')
            try:
                f = np.concatenate([l,r])
            except:
                print(file)
                continue
            np.save('/home/backup/data_cyr/assemble/features/'+file_name+'.npy', f)

def imu_video_concate():
    root = '/home/backup/data_cyr/assemble/'
    imu = os.listdir(root + 'features_imu')
    for file in imu:
        a = np.load(root + 'features_imu/' + file)
        v = np.load(root + 'features_video/' + file)
        try:
            f = np.concatenate([v, a])
        except:
            print(file)
            continue
        np.save('/home/backup/data_cyr/assemble/features/' + file, f)

if __name__ == '__main__':
    # imu_feature_concate()
    imu_video_concate()