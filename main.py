import cv2

car = 'cars.xml'
bus = 'Bus_front.xml'
pdt = 'pedestrian.xml'
twh = 'two_wheeler.xml'

cap = cv2.VideoCapture('TrafficCamera.mp4')

car_cascade = cv2.CascadeClassifier(car)
bus_cascade = cv2.CascadeClassifier(bus)
pedestrian_cascade = cv2.CascadeClassifier(pdt)
twowheeler_cascade = cv2.CascadeClassifier(twh)

scale_percent = 30


while True:
    ret, img = cap.read()
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    resizeimg = cv2.resize(img, dim)
    grayimg = cv2.cvtColor(resizeimg, cv2.COLOR_BGR2GRAY)

    cars = car_cascade.detectMultiScale(grayimg, 1.1, 3)
    buses = bus_cascade.detectMultiScale(grayimg, 1.1, 5)
    pedestrians = pedestrian_cascade.detectMultiScale(grayimg, 1.1, 5)
    twowheelers = twowheeler_cascade.detectMultiScale(grayimg, 1.1, 5)

    for (x,y,w,h) in cars:
        cv2.rectangle(resizeimg,(x,y),(x+w,y+h),(255,0,0),2)
        cv2.putText(resizeimg, 'car', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255,0,0),2)
    for (x,y,w,h) in buses:
        cv2.rectangle(resizeimg,(x,y),(x+w,y+h),(0,255,0),2)
        cv2.putText(resizeimg, 'bus', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,255,0),2)
    for (x,y,w,h) in pedestrians:
        cv2.rectangle(resizeimg,(x,y),(x+w,y+h),(0,0,255),2)
        cv2.putText(resizeimg, 'pedestrian', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,255),2)
    for (x,y,w,h) in twowheelers:
        cv2.rectangle(resizeimg,(x,y),(x+w,y+h),(255,0,255),2)
        cv2.putText(resizeimg, 'motobike', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255,0,255),2)

    cv2.imshow('video', resizeimg)
    cv2.waitKey(1)
    if cv2.waitKey(1) & 0xff== ord('q'):
        break

cv2.destroyAllWindows()