import cv2
cv2.setNumThreads(6)



def visualize(frame, face):

    for pt_num, (x,y,c) in enumerate(face.lms):
        x = int(x + 0.5)
        y = int(y + 0.5)
        frame = cv2.putText(frame, str(pt_num), (int(y), int(x)), cv2.FONT_HERSHEY_SIMPLEX, 0.25, (255,255,0))
        color = (0, 255, 0)
        if pt_num >= 66:
            color = (255, 255, 0)
        if not (x < 0 or y < 0 or x >= height or y >= width):
            cv2.circle(frame, (y, x), 1, color, -1)
    cv2.imshow("test",cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    cv2.waitKey(1)
