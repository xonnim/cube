import cv2
import numpy as np
import math
import cube_num
import time
from tkinter import *
import PIL.ImageGrab as ImageGrab

GAP = 2
OFFSET = 20

STICKER_AREA_TILE_SIZE = 30
STICKER_AREA_TILE_GAP = 4
STICKER_AREA_OFFSET = 20

norm_area =0
default_color = 0,0,0

current_col = None
current = [[default_color for _ in range(3)] for _ in range(3)]
current_str = wh_str = [["black" for _ in range(3)] for _ in range(3)]

wh = [[default_color for _ in range(3)] for _ in range(3)]
wh_str = [["black" for _ in range(3)] for _ in range(3)]
re = [[default_color for _ in range(3)] for _ in range(3)]
re_str = [["black" for _ in range(3)] for _ in range(3)]
ye = [[default_color for _ in range(3)] for _ in range(3)]
ye_str = [["black" for _ in range(3)] for _ in range(3)]
ora = [[default_color for _ in range(3)] for _ in range(3)]
ora_str = [["black" for _ in range(3)] for _ in range(3)]
gr = [[default_color for _ in range(3)] for _ in range(3)]
gr_str =[["black" for _ in range(3)] for _ in range(3)]
bl = [[default_color for _ in range(3)] for _ in range(3)]
bl_str = [["black" for _ in range(3)] for _ in range(3)]

col_text = { 1 : "Show the orange side",
             2 : 'Turn right to show the green side' ,
             3 : 'Turn right to show the red side',
             4 : 'Turn right to show the blue side',
             5 : 'Turn up to show the yellow side',
             0 : 'Turn up two times to show the white side' }

col_index = { 1 : 'orange',
              2 : 'green' ,
              3 : 'red',
              4 : 'blue',
              5 : 'yellow',
              0 : 'white' }
'''
fin = False

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
            return median, 'white' 
        else:
            return median, 'green'
    elif cube_num.green_H <= mh < cube_num.blue_H:
        if ms < 150:
            return median, 'white' 
        else:
            return median, 'blue'
    else:
        return median, 'red'

def draw_sticker(bgrcap, size) :

    global re, ora, bl, gr, wh, ye
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
    side_size = size * 3 + GAP * 2

    width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
    offset_x = width - (side_size * 4) - (side_offset * 3) - OFFSET
    offset_y = height - (side_size * 3) - (side_offset * 2) - OFFSET

    for side, (grid_x, grid_y) in grid.items():
        index = -1
        for row in range(3):
            for col in range(3):
                index += 1
                x1 = int((offset_x + size * col) +(GAP * col) +((side_size + side_offset) * grid_x))
                y1 = int((offset_y + size * row) +(GAP * row) +((side_size + side_offset) * grid_y))
                x2 = int(x1 + size)
                y2 = int(y1 + size)
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
    global re_str, ora_str, gr_str, bl_str, wh_str, ye_str
    color = []
    color_str = []
    t_c=[]
    t_s = []
    b_c = []
    b_s =[]
    m_c =[]
    m_s =[]

    for i in t :
        _,col = getcolor(i)
        t_s.append(col)
        col = change(col)
        t_c.append(col)
    for i in m :
        _,col = getcolor(i)
        m_s.append(col)
        col = change(col)
        m_c.append(col)
    for i in b :
        _,col = getcolor(i)
        b_s.append(col)
        col = change(col)
        b_c.append(col)

    color.append(t_c)
    color.append(m_c)
    color.append(b_c)

    color_str.append(t_s)
    color_str.append(m_s)
    color_str.append(b_s)
    

    if cent_color == 'red':
        re = color
        re_str = color_str
    if cent_color == 'orange' :
        ora = color
        ora_str = color_str
    if cent_color == 'blue' :
        bl = color
        bl_str = color_str
    if cent_color == 'green' :
        gr = color
        gr_str = color_str
    if cent_color == 'yellow' :
        ye = color
        ye_str = color_str
    if cent_color == 'white' :
        wh = color
        wh_str = color_str

def update_text(color_index) :

    color = col_text[color_index]
    textsize = cv2.getTextSize(color, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
    cv2.putText(bgrcap,color,((width - textsize[0]) // 2, height//6),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)

    te = 'Press M to go back'
    cv2.putText(bgrcap,te,(50, 450),cv2.FONT_HERSHEY_SIMPLEX,0.4,(255,255,255),2)

cam=cv2.VideoCapture(0)
_, bgrcap = cam.read()
fourcc = cv2.VideoWriter_fourcc(*'DIVX')
fps = cam.get(cv2.CAP_PROP_FPS)
height, width = bgrcap.shape[:2]
out = cv2.VideoWriter('output.avi', fourcc, fps, (width, height))
color_index = 1
start = time.time()

if not cam.isOpened():
    print("Could not open webcam")
    exit()

while cam.isOpened():    

    while True:

        status, frame = cam.read()
        key = cv2.waitKey(10) 
        delay = round(1000/fps)

        if key == ord('q'):
            break

        if key == ord('m'):

            if 1 <= color_index <= 5 :
                color_index -= 1
            elif color_index ==0 :
                color_index +=5

            if current_col == 'red':
                re = current
                re_str = current_str
            if current_col == 'orange' :
                ora = current
                ora_str = current_str
            if current_col == 'blue' :
                bl = current
                bl_str = current_str
            if current_col == 'green' :
                gr = current
                gr_str = current_str
            if current_col == 'yellow' :
                ye = current
                ye_str = current_str
            if current_col == 'white' :
                wh = current
                wh_str = current_str

        bgrcap = cv2.blur(frame, (5, 5))
        hsv = cv2.cvtColor(bgrcap, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        h_mask = cv2.inRange(h, 0, 160)
        h = cv2.bitwise_and(h, h, mask=h_mask)
        hsv = cv2.merge((h, s, v)).astype(float)
        
        if color_index<= 5 :
            update_text(color_index)

        color_mask = cv2.inRange(bgrcap, np.array([1, 1, 1]), np.array([0, 0, 0]))  
        white_mask = cv2.inRange(bgrcap, np.array([1, 1, 1]), np.array([0, 0, 0]))  
        
        now = time.time()
        tm = round(now - start)

        if tm <= 20 :
            cent = []  
            contours = find_contour(bgrcap)

            if len(contours) >= 5 :

                moment = find_cent(cent)
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
                current_col = cet_color

                if col_index[color_index] == cet_color and color_index<=4:
                    update_color(top_row,middle_row,bottom_row,cet_color)
                    if col_index[color_index] == 'white':
                        fin = True
                    start = time.time()
                    color_index += 1
                if col_index[color_index] == cet_color and color_index == 5:
                    update_color(top_row,middle_row,bottom_row,cet_color)
                    start = time.time()
                    color_index -= 5
            
        else :

            text = "Place the cube on the line and press K"
            textsize = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
            cv2.putText(bgrcap,text,((width - textsize[0]) // 2, height//6 + 20),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)
            cv2.line(bgrcap, (200, 120), (440, 120), (255, 255, 255), 4)
            cv2.line(bgrcap, (440, 120), (440, 360), (255, 255, 255), 4)
            cv2.line(bgrcap, (200, 360), (440, 360), (255, 255, 255), 4)
            cv2.line(bgrcap, (200, 120), (200, 360), (255, 255, 255), 4)
            if key == ord('k') :
                top_row = [(240, 160), (320, 160), (400, 160)]
                middle_row = [(240, 240), (320, 240), (400, 240)]
                bottom_row = [(240, 320), (320, 320), (400, 320)]

                _, cet_color = getcolor(middle_row[1])
                current_col = cet_color

                if col_index[color_index] == cet_color and color_index<=4:
                    update_color(top_row,middle_row,bottom_row,cet_color)
                    if col_index[color_index] == 'white':
                        fin = True
                    start = time.time()
                    color_index += 1
                if col_index[color_index] == cet_color and color_index == 5:
                    update_color(top_row,middle_row,bottom_row,cet_color)
                    start = time.time()
                    color_index -= 5

        if fin == True :
            break

        draw_sticker(bgrcap, 14)
        cv2.imshow('Webcam', bgrcap)
        out.write(bgrcap)


    cam.release()
    out.release()
    cv2.destroyAllWindows()
'''

######################################################

width = 60  
facelet_id = [[[0 for col in range(3)] for row in range(3)] for fc in range(6)]
colorpick_id = [0 for i in range(6)]
curcol = None
t = ("U", "R", "F", "D", "L", "B")
cols = ("yellow", "orange", "green", "red", "blue","white")

def click(_event):
    global curcol
    idlist = canvas.find_withtag("current")
    if len(idlist) > 0:
        if idlist[0] in colorpick_id:
            curcol = canvas.itemcget("current", "fill")
            for i in range(6):
                canvas.itemconfig(colorpick_id[i], width=1)
            canvas.itemconfig("current", width=5)
        else:
            canvas.itemconfig("current", fill=curcol)

def create_facelet_rects(a):
    offset = ((1 ,2), (0, 1), (1, 1), (2, 1), (3, 1), (1, 0))
    for f in range(6):
        for row in range(3):
            y = 10 + offset[f][1] * 3 * a + row * a
            for col in range(3):
                x = 10 + offset[f][0] * 3 * a + col * a
                facelet_id[f][row][col] = canvas.create_rectangle(x, y, x + a, y + a, fill="grey")
                if row == 1 and col == 1:
                    canvas.create_text(x + width // 2, y + width // 2, font=("", 14), text="", state=DISABLED)
    for f in range(6):
        canvas.itemconfig(facelet_id[f][1][1], fill=cols[f])

def create_colorpick_rects(a):
    global curcol
    global cols
    for i in range(6):
        x = (i % 3) * (a + 5) + 7 * a
        y = (i // 3) * (a + 5) + 7 * a
        colorpick_id[i] = canvas.create_rectangle(x, y, x + a, y + a, fill=cols[i])
        canvas.itemconfig(colorpick_id[0], width=4)
        curcol = cols[0]

def transfer():

    for index in range(6) :
        centercol = col_index[index]
        dc = {}
        for i in range(3):
            for j in range(3):
                if centercol == 'red':
                    canvas.itemconfig(facelet_id[index][i][j], fill=re_str[i][j])
                if centercol == 'orange':
                    canvas.itemconfig(facelet_id[index][i][j], fill=ora_str[i][j])
                if centercol == 'blue':
                    canvas.itemconfig(facelet_id[index][i][j], fill=bl_str[i][j])
                if centercol == 'green':
                    canvas.itemconfig(facelet_id[index][i][j], fill=gr_str[i][j])
                if centercol == 'white':
                    canvas.itemconfig(facelet_id[index][i][j], fill=wh_str[i][j])
                if centercol == 'yellow':
                    canvas.itemconfig(facelet_id[index][i][j], fill=ye_str[i][j])

def save():
    x = root.winfo_rootx()
    y = root.winfo_rooty()
    img = ImageGrab.grab(bbox=(x,y,x+740,y+560)).save("result.jpg")
    
    
root = Tk()
root.wm_title("Result")
canvas = Canvas(root, width=12 * width + 20, height=9 * width + 80)
canvas.pack()
canvas.bind("<Button-1>", click)
create_facelet_rects(width)
create_colorpick_rects(width)
btransfer = Button(text="Webcam import", height=2, width=13, relief=RAISED, command=transfer)
canvas.create_window(10 + 8.5 * width, 10 + 9.2 * width, anchor=NW, window=btransfer)
'''bsolve = Button(text="Solve", height=2, width=10, relief=RAISED, command=solve)
bsolve_window = canvas.create_window(10 + 10.5 * width, 10 + 6.5 * width, anchor=NW, window=bsolve)'''
bsave = Button(text="SAVE", height=1, width=10, relief=RAISED, command=save)
bsave_window = canvas.create_window(10 + 10.5 * width, 10 + 9.2 * width, anchor=NW, window=bsave)

root.mainloop()