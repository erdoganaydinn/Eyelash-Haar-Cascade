import cv2


eyelash_cascade = cv2.CascadeClassifier("eyelash.xml")

capt = cv2.VideoCapture(0)
while True:
    isTrue,video = capt.read()

    if isTrue == False:
        break

    grayVideo = cv2.cvtColor(video,cv2.COLOR_BGR2GRAY)
    eyelashs = eyelash_cascade.detectMultiScale(grayVideo,1.3,5)
    for (x,y,w,h) in eyelashs:
        cv2.rectangle(video,(x,y),(x+w,y+h),(0,0,255),3)
    

    cv2.imshow("eyelash-haar-cascade",video)
    esc = cv2.waitKey(30) & 0xff

    if esc == 27:
        break

    
capt.release()
cv2.destroyAllWindows()

    
