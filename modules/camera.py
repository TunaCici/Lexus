import cv2

class Camera:
    
    img = 0
    ret_val = 0

    def start(self):
        cam = cv2.VideoCapture(0)
        while True:
            selfret_val, self.img = cam.read()
            if True: 
                self.img = cv2.rotate(self.img, cv2.ROTATE_180)
            cv2.imshow('my webcam', self.img)
            if cv2.waitKey(1) == 27: 
                break  # esc to quit
        cv2.destroyAllWindows()

    def get_image(self):
        return self.img

if __name__ == "__main__":
    x = Camera()
    x.start()