import cv2


def start_camera():
    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        print("Camera could not be opened!")
        return

    print("Camera started successfully!")

    while True:
        ret, frame = camera.read()

        if not ret:
            print("Could not read camera frame!")
            break

        cv2.imshow("Smart Bus System - Camera", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    start_camera()