import cv2 as cv
import numpy as np

#corner detection
feature_params = dict(maxCorners = 200,
                      qualityLevel = 0.2,
                      minDistance = 7,
                      blockSize = 7,
                      useHarrisDetector = False,
                      k=0.04
                      )

#lk optical flow
lk_params = dict(maxLevel = 2,
                 criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))

capture = cv.VideoCapture(0)

if not capture.isOpened():
    raise RuntimeError("Could not open camera")

ret, frame = capture.read()
assert ret, "couldnt grab inital frame"
prev_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

#detect initial corners
p0 = cv.goodFeaturesToTrack(prev_gray, mask=None, **feature_params)
print("Inital corners: ", 0 if p0 is None else len(p0))


#creating mask for drawing
mask = np.zeros_like(frame)
colors = np.random.randint(0, 255, (500, 3), dtype=np.uint8)

while True:
    ret, frame = capture.read()
    if not ret:
        break
    
    gray = cv.GaussianBlur(cv.cvtColor(frame, cv.COLOR_BGR2GRAY), (0, 0), 1.2)

    if p0 is None or len(p0) == 0: #reseed if points lost
        p0 = cv.goodFeaturesToTrack(gray, mask=None, **feature_params)
    
        prev_gray = gray.copy()

    #calculate optical flow
    p1, st, err = cv.calcOpticalFlowPyrLK(prev_gray, gray, p0, None, **lk_params)

    #pick good points
    if p1 is not None:
        good_new = p1[st==1]
        good_old = p0[st==1]
    
    else:
        good_new = good_old = np.empty((0, 2), dtype=np.float32)

    mask = (mask * 0.92).astype(np.uint8)

    if good_new.size:
        v = good_new - good_old
        mag = np.linalg.norm(v, axis=1)
        m = mag > 0.6
        
    #overlay
    if np.any(m):
        mean_up = float(-np.mean(v[m, 1]))
        mean_side = float(np.mean(v[m, 0]))
        
        text = f"tracks: {good_new.shape[0]:3d} up:{mean_up:+.2f} side:{mean_side:+.2f}"
        cv.putText(frame, text, (10, 30),
                    cv.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 0, 0), 3, cv.LINE_AA)
        cv.putText(frame, text, (10, 30),
                    cv.FONT_HERSHEY_SIMPLEX, 0.7,
                    (255, 255, 255), 1, cv.LINE_AA)


        for i, (new, old) in enumerate(zip(good_new, good_old)):
            a, b = new.ravel().astype(int)
            c, d = old.ravel().astype(int)
            col = colors[i % len(colors)].tolist()
            cv.line(mask, (c, d), (a, b), col, 3)
            cv.circle(frame, (a, b), 3, col, 2)

    img = cv.addWeighted(frame, 1.0, mask, 0.7, 0.0)
    cv.imshow("Flow Ball", img)

    prev_gray = gray
    if good_new.size:
        p0 = good_new.reshape(-1, 1, 2).astype(np.float32)
    else:
        p0 = None

    key = cv.waitKey(1) & 0xFF
    if key in (ord('q'), 27):
        break

capture.release()
cv.destroyAllWindows()

