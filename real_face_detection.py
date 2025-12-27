import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# ۱. تنظیمات مدل (همانند قبل)
model_path = 'blaze_face_short_range.tflite'
base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.FaceDetectorOptions(base_options=base_options)
detector = vision.FaceDetector.create_from_options(options)

# ۲. باز کردن دوربین (عدد 0 معمولاً دوربین لپ‌تاپ است)
cap = cv2.VideoCapture(0)

print("برنامه در حال اجراست. برای خروج کلید 'q' را فشار دهید.")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("خطا در خواندن دوربین!")
        break

    # ۳. آماده‌سازی تصویر
    # تبدیل BGR به RGB برای هوش مصنوعی
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # ایجاد فرمت مخصوص MediaPipe
    image_mp = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)

    # ۴. تشخیص چهره
    detection_result = detector.detect(image_mp)

    # ۵. رسم نتایج روی فریم زنده
    if detection_result.detections:
        for detection in detection_result.detections:
            # الف) رسم مستطیل دور صورت
            box = detection.bounding_box
            start_point = (int(box.origin_x), int(box.origin_y))
            end_point = (int(box.origin_x + box.width), int(box.origin_y + box.height))
            cv2.rectangle(frame, start_point, end_point, (255, 0, 0), 2)

            # ب) رسم نقاط کلیدی (چشم، بینی، گوش)
            if detection.keypoints:
                for keypoint in detection.keypoints:
                    # تبدیل مختصات درصدی به پیکسل
                    x = int(keypoint.x * frame.shape[1])
                    y = int(keypoint.y * frame.shape[0])
                    cv2.circle(frame, (x, y), 4, (0, 255, 0), -1)

    # ۶. نمایش خروجی
    cv2.imshow('Live Face Detection', frame)

    # ۷. کلید خروج (q)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ۸. آزاد کردن منابع
cap.release()
cv2.destroyAllWindows()