import time
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

t = 0
#creating mask for drawing
mask = np.zeros_like(frame)
colors = np.random.randint(0, 255, (500, 3), dtype=np.uint8)

#ball 
x, y = 320.0, 120.0 #starting position
vx,vy = 0.0, 0.0 #velocity
R = 24 #radius
ROI_MARGIN = 36 #flow region for extra pixels around ball

g = 100.0 #gravity
drag = 0.995 #velocity damping
bounce = 0.75 #energy kept on vertical bounce

t_prev = time.time()

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

    #ROI flow
    mean_up_roi, mean_side_roi = 0.0, 0.0
    h, w = gray.shape[:2]

    if good_new.size:
        v = good_new - good_old
        mag = np.linalg.norm(v, axis=1)
        m = mag > 0.6
        
        #overlay
        if np.any(m):

            #ROI box around ball
            x0 = max(0, int(x - R - ROI_MARGIN))
            y0 = max(0, int(y - R - ROI_MARGIN))
            x1 = min(w, int(x + R + ROI_MARGIN))
            y1 = min(h, int(y + R + ROI_MARGIN))

            pts = good_new[m]
            inside = (pts[:,0] >= x0) & (pts[:,0] < x1) & (pts[:,1] >= y0) &(pts[:,1] < y1)
            if np.any(inside):
                v_roi = v[m][inside]
                mean_up_roi = float(-np.mean(v_roi[:,1]))
                mean_side_roi = float(np.mean(v_roi[:,0]))
                cv.rectangle(frame, (x0, y0), (x1, y1), (255, 200, 64), 1)
            
            #physics integration
            t_now = time.time()
            dt = max(1e-3, t_now - t_prev)
            t_prev = t_now
            
            #gravity
            vy += g * dt

            #impulses from ROI 
            if mean_up_roi > 0.35: #threshold
                vy -= 900.0 * mean_up_roi * dt
            if abs(mean_side_roi) > 0.2:
                vx += 400.0 *mean_side_roi * dt

            #integrate position
            x += vx * dt
            y += vy * dt

            #collision with window bounds
            if y > h - R:
                y = h - R
                vy = -vy * bounce
            if y < R:
                y = R
                vy = -vy * bounce
            if x > w - R:
                x = w - R
                vx = -vx * 0.9
            if x < R:
                x = R
                vx = -vx * 0.9


            #drag
            vx *= drag
            vy *= drag

            # draw ball
            cv.circle(frame, (int(x), int(y)), R, (60,180,255), -1)

            # optional HUD
            hud = f"upROI:{mean_up_roi:+.2f} sideROI:{mean_side_roi:+.2f}"
            cv.putText(frame, hud, (10, 55), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 3, cv.LINE_AA)
            cv.putText(frame, hud, (10, 55), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1, cv.LINE_AA)


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
            cv.circle(frame, (a, b), 3, col, -1)

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

