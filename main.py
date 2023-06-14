import time
import cv2

def image_processing():
    img = cv2.imread('./images/variant-3.jpeg')
    if img is None:
        print("Failed to load the image")
        exit()
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)  # Convert from RGB to HSV
    cv2.imshow('original', img)
    cv2.imshow('HSV', hsv_img)

    h, w = img.shape[:2]
    center_x, center_y = w // 2, h // 2
    square_size = 200

    # Draw a square in the center of the image
    top_left = (center_x - square_size // 2, center_y - square_size // 2)
    bottom_right = (center_x + square_size // 2, center_y + square_size // 2)
    cv2.rectangle(img, top_left, bottom_right, (0, 255, 0), 2)

    cv2.imshow('labeled_image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def video_processing():
    cap = cv2.VideoCapture(0)
    down_points = (640, 480)
    i = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, down_points, interpolation=cv2.INTER_LINEAR)
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)  # Convert from RGB to HSV

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)
        ret, thresh = cv2.threshold(gray, 110, 255, cv2.THRESH_BINARY_INV)

        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if len(contours) > 0:
            c = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(c)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            if i % 5 == 0:
                a = x + (w // 2)
                b = y + (h // 2)
                print(a, b)

        cv2.imshow('frame', frame)
        cv2.imshow('HSV_frame', hsv_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        time.sleep(0.1)
        i += 1

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    image_processing()
    # video_processing()
