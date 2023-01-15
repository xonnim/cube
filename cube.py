import cv2
import numpy as np
import math


SIZE = 14
GAP = 2
OFFSET = 20
STICKER_AREA_TILE_SIZE = 30
STICKER_AREA_TILE_GAP = 4
STICKER_AREA_OFFSET = 20
norm_area =0
default_color = 0,0,0

wh = [[default_color for _ in range(3)] for _ in range(3)]
re = [[default_color for _ in range(3)] for _ in range(3)]
ye = [[default_color for _ in range(3)] for _ in range(3)]
ora = [[default_color for _ in range(3)] for _ in range(3)]
gr = [[default_color for _ in range(3)] for _ in range(3)]
bl = [[default_color for _ in range(3)] for _ in range(3)]

def find_contour(frame):

    grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurredFrame = cv2.blur(grayFrame, (3, 3))
    cannyFrame = cv2.Canny(blurredFrame, 0, 60, 3)
    kernel = np.ones((4,4), np.uint16)
    dilatedFrame = cv2.dilate(cannyFrame, kernel, iterations=2)
    kernel = np.ones((5, 5), np.uint8)
    dilatedFrame = cv2.erode(dilatedFrame, kernel, iterations = 1)
    contours, hierarchy = cv2.findContours(dilatedFrame, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    first_contours = []
    final_contours = []
    first_moments = []
    final_moments = []
    global norm_area, cent

    for contour in contours:

        epsilon = 0.06*cv2.arcLength(contour,True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        #cv2.drawContours(bgrcap, [approx], 0, (0, 255, 0), 3)
        if len (approx) == 4:

            area = cv2.contourArea(contour)

            if 100000 >= area >= 1000:    

                (x, y, w, h) = cv2.boundingRect(approx)

                ratio = w / float(h)

                if ratio >= 0.9 and ratio <= 1.1 and 1.2> area / (w * h) > 0.4 :
                    first_contours.append((x, y, w, h))
                    m = cv2.moments(approx)
                    first_moments.append((int(m['m10'] / m['m00']), int(m['m01'] / m['m00'])))

    found = False
    contour_neighbors = {}

    for index, contour in enumerate(first_contours):

        (x, y, w, h) = contour
        contour_neighbors[index] = []
        center_x, center_y = first_moments[index]
        radius = 1.5

        if w*h <= 10000:

            neighbor_positions = [
                [(center_x - w * radius), (center_y - h * radius)],
                [center_x, (center_y - h * radius)],
                [(center_x + w * radius), (center_y - h * radius)],
                [(center_x - w * radius), center_y],
                [center_x, center_y],
                [(center_x + w * radius), center_y],
                [(center_x - w * radius), (center_y + h * radius)],
                [center_x, (center_y + h * radius)],
                [(center_x + w * radius), (center_y + h * radius)],
            ]

            for index2 in range(index, len(first_contours)):
                (x2, y2, w2, h2) = first_contours[index2]
                for (x3, y3) in neighbor_positions:
                    if (x2 < x3 and y2 < y3) and (x2 + w2 > x3 and y2 + h2 > y3):
                        final_contours.append(contour)
                        final_moments.append(first_moments[index])
                        norm_area += w*h

        cent = final_moments

        if len(list(set(final_contours))) >= 6:
            found = True
            norm_area = norm_area // len(final_contours)
            cent = list(set(cent))

    if not found:
        return []
    
    final_contours = list(set(final_contours))
    y_sorted = sorted(final_contours, key=lambda item: item[1])

    top_row = sorted(y_sorted[0:3], key=lambda item: item[0])
    middle_row = sorted(y_sorted[3:6], key=lambda item: item[0])
    bottom_row = sorted(y_sorted[6:9], key=lambda item: item[0])

    sorted_contours = top_row + middle_row + bottom_row

    return sorted_contours

def draw_contour(contours):
    
    for index, (x, y, w, h) in enumerate(contours):
        #if 1.5 >= (w*h / norm_area) >= 0.5 :
        cv2.rectangle(bgrcap, (int(x), int(y)), (int(x) + int(w), int(y) + int(h)), (255, 0, 0), 2)
    norm_area = 0

def find_cent(cent) :

    y_list = []
    x_list = []

    for i in cent :
        x_list.append(i[0])
        y_list.append(i[1])
    
    y_list.sort()
    x_list.sort()

    if len(y_list) >=2 and len(x_list)>=2 :

        y_cent = (y_list[-1] + y_list[0])//2
        x_cent = (x_list[-1] + x_list[0])//2
        moment = (x_cent, y_cent)
        return moment

    else :
        return 0

def find_rest(contours, cent_rect) :

    global cent

    (center_x, center_y) = cent_rect
    
    all = contours

    '''neighbor_positions = [
                [(center_x - w * radius), (center_y - h * radius)],
                [center_x, (center_y - h * radius)],
                [(center_x + w * radius), (center_y - h * radius)],
                [(center_x - w * radius), center_y],
                [center_x, center_y],
                [(center_x + w * radius), center_y],
                [(center_x - w * radius), (center_y + h * radius)],
                [center_x, (center_y + h * radius)],
                [(center_x + w * radius), (center_y + h * radius)],
            ]'''

    area = 0

    for index, contour in enumerate(contours):
        (x, y, w, h) = contour
        area += w*h
    w = round(math.sqrt((area // len(contours))), 2)

    n_positions = [
                [(center_x - w ), (center_y - w)],
                [center_x, (center_y - w)],
                [(center_x + w ), (center_y - w)],
                [(center_x - w ), center_y],
                [center_x, center_y],
                [(center_x + w ), center_y],
                [(center_x - w ), (center_y + w )],
                [center_x, (center_y + w )],
                [(center_x + w ), (center_y + w)],
            ]
    positions = [
                [(center_x - w ), (center_y - w)],
                [center_x, (center_y - w)],
                [(center_x + w ), (center_y - w)],
                [(center_x - w ), center_y],
                [center_x, center_y],
                [(center_x + w ), center_y],
                [(center_x - w ), (center_y + w )],
                [center_x, (center_y + w )],
                [(center_x + w ), (center_y + w)],
            ]

    for index in range(len(contours)):
        (x2, y2, w2, h2) =contours[index]
        for (x3, y3) in n_positions:
            if (x2 < x3 and y2 < y3) and (x2 + w2 > x3 and y2 + h2 > y3):
                if [x3, y3] in positions:
                    positions.remove([x3,y3])
            else :
                pass

    for (x,y) in positions :
        all.append((x-w//2,y-w//2,w,w))
        cent.append((x,y))
    return all

def getcolor(i):
    sz = 10
    p = np.array(i)
    p = p.astype(np.uint16)
    rect = hsv[p[1] - sz:p[1] + sz, p[0] - sz:p[0] + sz]
    median = np.sum(rect, axis=(0, 1)) / sz / sz / 4
    mh, ms, mv = median
    if ms <= cube_num.sat_W and mv >= cube_num.val_W:
        return median, 'white'
    elif cube_num.orange_L <= mh < cube_num.orange_H:
        return median, 'orange'
    elif cube_num.orange_H <= mh < cube_num.yellow_H:
        return median, 'yellow'
    elif cube_num.yellow_H <= mh < cube_num.green_H:
        if ms < 150:
            return median, 'white'  # green saturation is always higher
        else:
            return median, 'green'
    elif cube_num.green_H <= mh < cube_num.blue_H:
        if ms < 150:
            return median, 'white'  # blue saturation is always higher
        else:
            return median, 'blue'
    else:
        return median, 'red'

def draw_sticker() :
    global re, ora, bl, gr, wh, ye
    SIZE = 14
    GAP = 2
    OFFSET = 20

    grid = {
                'white' : [1, 0],
                'orange': [0, 1],
                'green' : [1, 1],
                'red'   : [2, 1],
                'blue'  : [3, 1],
                'yellow': [1, 2],
            }

    side_offset = GAP * 3
    side_size = SIZE * 3 + GAP * 2

    width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
    offset_x = width - (side_size * 4) - (side_offset * 3) - OFFSET
    offset_y = height - (side_size * 3) - (side_offset * 2) - OFFSET

    for side, (grid_x, grid_y) in grid.items():
        index = -1
        for row in range(3):
            for col in range(3):
                index += 1
                x1 = int((offset_x + SIZE * col) +(GAP * col) +((side_size + side_offset) * grid_x))
                y1 = int((offset_y + SIZE * row) +(GAP * row) +((side_size + side_offset) * grid_y))
                x2 = int(x1 + SIZE)
                y2 = int(y1 + SIZE)
                if side == 'red':
                    color = 0, 0, 255
                    if row == col == 1 :
                        cv2.rectangle(bgrcap,(x1, y1),(x2, y2),color,-1)
                    else:    
                        cv2.rectangle(bgrcap,(x1, y1),(x2, y2),re[row][col],-1)
                if side == 'orange':
                    color = 0, 165, 255
                    if row == col == 1 :
                        cv2.rectangle(bgrcap,(x1, y1),(x2, y2),color,-1)
                    else:    
                        cv2.rectangle(bgrcap,(x1, y1),(x2, y2),ora[row][col],-1)
                if side == 'blue':
                    color = 255, 0, 0
                    if row == col == 1 :
                        cv2.rectangle(bgrcap,(x1, y1),(x2, y2),color,-1)
                    else:    
                        cv2.rectangle(bgrcap,(x1, y1),(x2, y2),bl[row][col],-1)
                if side == 'green':
                    color = 0, 255, 0
                    if row == col == 1 :
                        cv2.rectangle(bgrcap,(x1, y1),(x2, y2),color,-1)
                    else:    
                        cv2.rectangle(bgrcap,(x1, y1),(x2, y2),gr[row][col],-1)
                if side == 'white':
                    color = 255,255,255
                    if row == col == 1 :
                        cv2.rectangle(bgrcap,(x1, y1),(x2, y2),color,-1)
                    else:    
                        cv2.rectangle(bgrcap,(x1, y1),(x2, y2),wh[row][col],-1)
                if side == 'yellow':
                    color = 0,255,255
                    if row == col == 1 :
                        cv2.rectangle(bgrcap,(x1, y1),(x2, y2),color,-1)
                    else:    
                        cv2.rectangle(bgrcap,(x1, y1),(x2, y2),ye[row][col],-1)

                '''if row == col == 1 :
                    cv2.rectangle(bgrcap,(x1, y1),(x2, y2),color,-1)
                else:    
                    cv2.rectangle(bgrcap,(x1, y1),(x2, y2),(0, 0, 0),-1)'''

def change (col):

    if col == 'red':
        col = 0, 0, 255
    if col == 'orange' :
        col = 0, 165, 255
    if col == 'blue' :
        col = 255, 0, 0
    if col == 'green' :
        col = 0, 255, 0
    if col == 'yellow' :
        col = 0,255,255
    if col == 'white' :
        col = 255,255,255

    return col

def update_color(t,m,b,cent_color) :
    global re, ora, gr, bl, wh, ye
    color = []
    t_c=[]
    b_c = []
    m_c =[]

    for i in t :
        _,col = getcolor(i)
        col = change(col)
        t_c.append(col)
    for i in m :
        _,col = getcolor(i)
        col = change(col)
        m_c.append(col)
    for i in b :
        _,col = getcolor(i)
        col = change(col)
        b_c.append(col)
    color.append(t_c)
    color.append(m_c)
    color.append(b_c)
    print(color)   
    

    if cent_color == 'red':
        re = color
    if cent_color == 'orange' :
        ora = color
    if cent_color == 'blue' :
        bl = color
    if cent_color == 'green' :
        gr = color
    if cent_color == 'yellow' :
        ye = color
    if cent_color == 'white' :
        wh = color

cam=cv2.VideoCapture(0)
_, bgrcap = cam.read()

if not cam.isOpened():
    print("Could not open webcam")
    exit()

height, width = bgrcap.shape[:2]

while cam.isOpened():    

    while True:

        status, frame = cam.read()
        key = cv2.waitKey(10) & 0xff
        fps = cam.get(cv2.CAP_PROP_FPS)
        fourcc = cv2.VideoWriter_fourcc(*'DIVX')
        delay = round(1000/fps)

        if key == 27:
            break

        bgrcap = cv2.blur(frame, (5, 5))
        hsv = cv2.cvtColor(bgrcap, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        h_mask = cv2.inRange(h, 0, 160)
        h = cv2.bitwise_and(h, h, mask=h_mask)
        hsv = cv2.merge((h, s, v)).astype(float)

        color_mask = cv2.inRange(bgrcap, np.array([1, 1, 1]), np.array([0, 0, 0]))  
        white_mask = cv2.inRange(bgrcap, np.array([1, 1, 1]), np.array([0, 0, 0]))  

        cent = []  

        contours = find_contour(bgrcap)
        #draw_contour(contours)
        if len(contours) >= 5 :
            moment = find_cent(cent)
            #cv2.line(bgrcap, moment, moment, (0,0,0), 5)
            found = False
            for index, contour in enumerate(contours):
                (x, y, w, h) = contour
                if (moment[0] < x + w and moment[1] < y + h) and (moment[0] > x and moment[1] > y):
                    found = True
                    cent_rect = (x,y,w,h)
                    #cv2.rectangle(bgrcap, (x, y), (x + w, y + h), (255, 255, 0), 2)
            if found == False :
                cent_rect = (moment[0]-30, moment[1]-30, 60, 60)
            
            contours = find_rest(contours, moment)
            draw_contour(contours)
        
            y_sorted = sorted(cent, key=lambda item: item[1])

            top_row = sorted(y_sorted[0:3], key=lambda item: item[0])
            middle_row = sorted(y_sorted[3:6], key=lambda item: item[0])
            bottom_row = sorted(y_sorted[6:9], key=lambda item: item[0])

            cent = top_row + middle_row + bottom_row 
            _, cet_color = getcolor(middle_row[1])
            update_color(top_row,middle_row,bottom_row,cet_color)
            '''for i in cent :
                _, col = getcolor(i)
                print(col)'''

        draw_sticker()
        cv2.imshow('Webcam', bgrcap)
        out = cv2.VideoWriter('output.avi', fourcc, fps, (width, height))


    cam.release()
    cv2.destroyAllWindows()