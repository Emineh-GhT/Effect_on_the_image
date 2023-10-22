import numpy as np
import cv2

face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
smile_detector = cv2.CascadeClassifier('haarcascade_smile.xml')
eye_detector = cv2.CascadeClassifier('haarcascade_eye.xml')
img = cv2.imread('images/pic.jpg' , 0)
img = cv2.resize(img , (500,500))
video_cap1 = cv2.VideoCapture(0)#be cam laptop vasl mishe
video_cap2 = cv2.VideoCapture('images/obama.mp4')
emoji = cv2.imread('images/ez.png' , 0)
# محاسبه مقدار بیشینه پیکسل‌ها
max_pixel_value1 = np.max(emoji)
# نگاتیو کردن تصویر
emoji = max_pixel_value1 - emoji
eye_emoji = cv2.imread('images/hearth.jpg' , 0)
# محاسبه مقدار بیشینه پیکسل‌ها
max_pixel_value2 = np.max(eye_emoji)
# نگاتیو کردن تصویر
eye_emoji = max_pixel_value2 - eye_emoji
lips_emoji = cv2.imread('images/lips.jpg' , 0)
# محاسبه مقدار بیشینه پیکسل‌ها
max_pixel_value3 = np.max(lips_emoji)
# نگاتیو کردن تصویر
lips_emoji = max_pixel_value3 - lips_emoji
show_effect1 = False  # وضعیت نمایش ایموجی
show_effect2 = False
show_effect3 = False
show_effect4 = False

# تابع برای رسم دایره روی تصویر
def draw_circle(image, center, radius, color, thickness):
    cv2.circle(image, center, radius, color, thickness)
# تابع برای اعمال ماسک روی تصویر
def apply_mask(image, mask):
    masked_image = cv2.bitwise_and(image, image, mask=mask)
    return masked_image

# faces = face_detector.detectMultiScale(img , scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)) #img , mizan hasasiat و ...
# eyes = eye_detector.detectMultiScale(img , scaleFactor=1.1, minNeighbors=100, minSize=(50, 50)) #img , mizan hasasiat و ...
# smiles = smile_detector.detectMultiScale(img , scaleFactor=1.1, minNeighbors=100, minSize=(100, 100)) #img , mizan hasasiat و ...

# cv2.imshow('output' , img)
# key = cv2.waitKey()
# # اگر کلید "1" فشرده شده بود، تغییر وضعیت نمایش ایموجی
# if key == ord('1'):
#     show_effect1 = not show_effect1
#     for face in faces:
#         x, y, w, h = face
#         sticker_resized = cv2.resize(emoji,(w,h)) 
#         # مرکز دایره
#         center = (x + w // 2, y + h // 2)
#         # شعاع دایره (نصف عرض یا ارتفاع چهره)
#         radius = min(w, h) // 2
#         # رسم دایره با رنگ آبی وضخامت 2
#         draw_circle(img, center, radius, (255, 0, 0), 2)
#         # ایجاد ماسک دایره
#         mask = np.zeros_like(sticker_resized)
#         # رسم دایره در ماسک با رنگ سفید وضخامت -1 (پر شده)
#         draw_circle(mask, (w // 2, h // 2), radius, (255, 255, 255), -1)
#         # اعمال ماسک بر روی تصویر چهره
#         masked_image = apply_mask(sticker_resized, mask)
#         # ادغام تصویر چهره و ایموجی
#         img[y:y+h, x:x+w] = cv2.add(img[y:y+h, x:x+w], masked_image)
#     cv2.imwrite('output1.jpg' , img)
# if key == ord('2'):
#     show_effect2 = not show_effect2
#     for eye in eyes:
#         x, y, w, h = eye
#         sticker_eye_resized = cv2.resize(eye_emoji,(w,h))
#         # مرکز دایره
#         center = (x + w // 2, y + h // 2)
#         # شعاع دایره (نصف عرض یا ارتفاع چهره)
#         radius = min(w, h) // 2
#         # رسم دایره با رنگ آبی وضخامت 2
#         draw_circle(img, center, radius, (255, 0, 0), 2)
#         # ایجاد ماسک دایره
#         mask_eye = np.zeros_like(sticker_eye_resized)
#         # رسم دایره در ماسک با رنگ سفید وضخامت -1 (پر شده)
#         draw_circle(mask_eye, (w // 2, h // 2), radius, (255, 255, 255), -1)
#         # اعمال ماسک بر روی تصویر چهره
#         masked_eye_image = apply_mask(sticker_eye_resized, mask_eye)
#         # ادغام تصویر چهره و ایموجی
#         img[y:y+h, x:x+w] = cv2.add(img[y:y+h, x:x+w], masked_eye_image)
#     for smile in smiles:
#         x, y, w, h = smile
#         sticker_smile_resized = cv2.resize(lips_emoji,(w,h))
#         # مرکز دایره
#         center = (x + w // 2, y + h // 2)
#         # شعاع دایره (نصف عرض یا ارتفاع چهره)
#         radius = min(w, h) // 2
#         # رسم دایره با رنگ آبی وضخامت 2
#         draw_circle(img, center, radius, (255, 0, 0), 2)
#         # ایجاد ماسک دایره
#         mask_smile = np.zeros_like(sticker_smile_resized)
#         # رسم دایره در ماسک با رنگ سفید وضخامت -1 (پر شده)
#         draw_circle(mask_smile, (w // 2, h // 2), radius, (255, 255, 255), -1)
#         # اعمال ماسک بر روی تصویر چهره
#         masked_smile_image = apply_mask(sticker_smile_resized, mask_smile)
#         # ادغام تصویر چهره و ایموجی
#         img[y:y+h, x:x+w] = cv2.add(img[y:y+h, x:x+w], masked_smile_image)
#     cv2.imwrite('output2.jpg' , img)
# if key == ord('3'):
#     show_effect3 = not show_effect3
#     for (x,y,w,h) in faces:
#         temp = cv2.resize(img[y:y+h, x:x+w],(20,20))
#         img[y:y+h, x:x+w] = cv2.resize(temp,(w,h),interpolation=cv2.INTER_AREA)
#     cv2.imwrite('output3.jpg' , img)
# if key == ord('4'):
#     img = cv2.flip(img , 1)
#     cv2.imwrite('output4.jpg' , img)
# cv2.imshow('output' , img)
# cv2.waitKey()


def effect1_on_face():
    faces = face_detector.detectMultiScale(frame_gray , scaleFactor=1.3, minNeighbors=5, minSize=(50, 50)) #img , mizan hasasiat و ...
    for face in faces:
        x, y, w, h = face
        sticker_resized = cv2.resize(emoji,(w,h))
        # مرکز دایره
        center = (x + w // 2, y + h // 2)
        # شعاع دایره (نصف عرض یا ارتفاع چهره)
        radius = min(w, h) // 2
        # رسم دایره با رنگ آبی وضخامت 2
        draw_circle(frame_gray, center, radius, (255, 0, 0), 2)
        # ایجاد ماسک دایره
        mask = np.zeros_like(sticker_resized)
        # رسم دایره در ماسک با رنگ سفید وضخامت -1 (پر شده)
        draw_circle(mask, (w // 2, h // 2), radius, (255, 255, 255), -1)
        # اعمال ماسک بر روی تصویر چهره
        masked_image = apply_mask(sticker_resized, mask)
        # ادغام تصویر چهره و ایموجی
        frame_gray[y:y+h, x:x+w] = cv2.add(frame_gray[y:y+h, x:x+w], masked_image)
def effect2_on_face():
    eyes = eye_detector.detectMultiScale(frame_gray , scaleFactor=1.3, minNeighbors=5, minSize=(10, 10)) #img , mizan hasasiat و ...
    smiles = smile_detector.detectMultiScale(frame_gray , scaleFactor=1.3, minNeighbors=5, minSize=(20, 20)) #img , mizan hasasiat و ...
    for eye in eyes:
        x, y, w, h = eye
        sticker_eye_resized = cv2.resize(eye_emoji,(w,h))
        # مرکز دایره
        center = (x + w // 2, y + h // 2)
        # شعاع دایره (نصف عرض یا ارتفاع چهره)
        radius = min(w, h) // 2
        # رسم دایره با رنگ آبی وضخامت 2
        draw_circle(frame_gray, center, radius, (255, 0, 0), 2)
        # ایجاد ماسک دایره
        mask_eye = np.zeros_like(sticker_eye_resized)
        # رسم دایره در ماسک با رنگ سفید وضخامت -1 (پر شده)
        draw_circle(mask_eye, (w // 2, h // 2), radius, (255, 255, 255), -1)
        # اعمال ماسک بر روی تصویر چهره
        masked_eye_image = apply_mask(sticker_eye_resized, mask_eye)
        # ادغام تصویر چهره و ایموجی
        frame_gray[y:y+h, x:x+w] = cv2.add(frame_gray[y:y+h, x:x+w], masked_eye_image)
    for smile in smiles:
        x, y, w, h = smile
        sticker_smile_resized = cv2.resize(lips_emoji,(w,h))
        # مرکز دایره
        center = (x + w // 2, y + h // 2)
        # شعاع دایره (نصف عرض یا ارتفاع چهره)
        radius = min(w, h) // 2
        # رسم دایره با رنگ آبی وضخامت 2
        draw_circle(frame_gray, center, radius, (255, 0, 0), 2)
        # ایجاد ماسک دایره
        mask_smile = np.zeros_like(sticker_smile_resized)
        # رسم دایره در ماسک با رنگ سفید وضخامت -1 (پر شده)
        draw_circle(mask_smile, (w // 2, h // 2), radius, (255, 255, 255), -1)
        # اعمال ماسک بر روی تصویر چهره
        masked_smile_image = apply_mask(sticker_smile_resized, mask_smile)
        # ادغام تصویر چهره و ایموجی
        frame_gray[y:y+h, x:x+w] = cv2.add(frame_gray[y:y+h, x:x+w], masked_smile_image)
def effect3_on_face():
    faces = face_detector.detectMultiScale(frame_gray , scaleFactor=1.3, minNeighbors=5, minSize=(50, 50)) #img , mizan hasasiat و ...
    for face in faces:
        x, y, w, h = face
        temp = cv2.resize(frame_gray[y:y+h, x:x+w],(20,20))
        frame_gray[y:y+h, x:x+w] = cv2.resize(temp,(w,h),interpolation=cv2.INTER_AREA)
        
while True:
    # ret, frame = video_cap2.read()
    ret, frame = video_cap2.read()
    # انتظار برای فشردن کلید
    key = cv2.waitKey(1) #1ms
    # اگر کلید "q" فشرده شد، خروج از حلقه while  
    if ret == False or key == ord('q'):
        break
    frame = cv2.resize(frame,(500,500))
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # انتظار برای فشردن کلید
    key = cv2.waitKey(1) #1ms
    # اگر کلید "1" فشرده شده بود، تغییر وضعیت نمایش ایموجی
    if key == ord('1'):
        show_effect1 = not show_effect1
    if show_effect1:
        effect1_on_face() 
    # اگر کاربر دکمه‌ای را فشار داد، وقفه در صورت نمایش ایموجی ایجاد می‌شود
    if cv2.waitKey(1) != -1:
        show_effect1 = False
    # انتظار برای فشردن کلید
    key = cv2.waitKey(1) #1ms
    # اگر کلید "2" فشرده شده بود، تغییر وضعیت نمایش ایموجی
    if key == ord('2'):
        show_effect2 = not show_effect2
    if show_effect2:
        effect2_on_face()   
    # اگر کاربر دکمه‌ای را فشار داد، وقفه در صورت نمایش ایموجی ایجاد می‌شود
    if cv2.waitKey(1) != -1:
        show_effect2 = False
    # انتظار برای فشردن کلید
    key = cv2.waitKey(1) #1ms
    # اگر کلید "3" فشرده شده بود، تغییر وضعیت نمایش ایموجی
    if key == ord('3'):
        show_effect3 = not show_effect3
    if show_effect3:
        effect3_on_face()   
    # اگر کاربر دکمه‌ای را فشار داد، وقفه در صورت نمایش ایموجی ایجاد می‌شود
    if cv2.waitKey(1) != -1:
        show_effect3 = False
    # انتظار برای فشردن کلید
    key = cv2.waitKey(1) #1ms
    # اگر کلید "4" فشرده شده بود، تغییر وضعیت نمایش ایموجی
    if key == ord('4'):
        show_effect4 = not show_effect4
    if show_effect4:
        frame_gray = cv2.flip(frame_gray , 1)   
    # اگر کاربر دکمه‌ای را فشار داد، وقفه در صورت نمایش ایموجی ایجاد می‌شود
    if cv2.waitKey(1) != -1:
        show_effect3 = False

    cv2.imshow('output' , frame_gray)

# آزاد کردن منابع و بستن پنجره‌های باز
video_cap2.release()
cv2.destroyAllWindows()


