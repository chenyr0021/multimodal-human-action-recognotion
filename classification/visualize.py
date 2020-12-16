import cv2

video_file = './assemble_ori/20201102-211559.mp4'
fusion_result_file = open('./results/assemble/split_1/20201102-211559')
imu_result_file = open('./results/assemble_imu/split_1/20201102-211559')
video_result_file = open('./results/assemble_video/split_1/20201102-211559')

gt_file = open('./results/assemble/groundTruth/20201102-211559.txt')
fusion_result = fusion_result_file.read().split('\n')[-1].split(' ')
imu_result = imu_result_file.read().split('\n')[-1].split(' ')
video_result = video_result_file.read().split('\n')[-1].split(' ')

gt = gt_file.read().split('\n')
i=0
cap = cv2.VideoCapture(video_file)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
out = cv2.VideoWriter('out4.mp4',cv2.CAP_FFMPEG, fourcc=fourcc, fps=fps, frameSize=(640, 480),params=None)
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    # cv2.rectangle(frame, (0,0), (30,30), color=0)
    imu_pred = imu_result[min(i, len(imu_result)-1)]
    video_pred = video_result[min(i, len(video_result)-1)]
    fusion_pred = fusion_result[min(i, len(fusion_result)-1)]
    imu_color = (0, 255, 0) if imu_pred == gt[i] else (0, 0, 255)
    video_color = (0, 255, 0) if video_pred == gt[i] else (0, 0, 255)
    fusion_color = (0, 255, 0) if fusion_pred == gt[i] else (0, 0, 255)

    font = cv2.QT_FONT_NORMAL
    key1 = 'ground_truth: ' + gt[i]
    key2 = 'imu_pred:     ' + imu_pred
    key3 = 'video_pred:   ' + video_pred
    key4 = 'fusion_pred:  ' + fusion_pred

    cv2.putText(frame, key1, (0,25), font, 1, (255, 0, 0), 2)
    cv2.putText(frame, key2, (0,60), font, 1, imu_color, 2)
    cv2.putText(frame, key3, (0,90), font, 1, video_color, 2)
    cv2.putText(frame, key4, (0,120), font, 1, fusion_color, 2)
    cv2.imshow('video', frame)
    out.write(frame)
    i += 1
    if cv2.waitKey(10) == ord('q'):
        break
fusion_result_file.close()
imu_result_file.close()
video_result_file.close()
cv2.destroyAllWindows()
cap.release()