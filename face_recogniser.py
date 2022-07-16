import cv2
import face_recognition as fr
img1 = fr.load_image_file('msd.jpg')

# converting to rgb format because face recognition loads image in bgr format
img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)

img1test = fr.load_image_file('msd2.jpg')

img1test = cv2.cvtColor(img1test, cv2.COLOR_BGR2RGB)

# finding face location in image.
face=fr.face_locations(img1)[0]
# face_locations function returns a tuple which have coordinates of the face in the image.
# print(face)

# encoding image will give us a tuple which have features of face in the form of numeric values
encodedimage = fr.face_encodings(img1)[0]
# print(encodedimage)

cv2.rectangle(img1, (face[3], face[0]), (face[1], face[2]), (255, 0, 0), 4)
# finding face location in test image
facetest = fr.face_locations(img1test)[0]
encodeTestImage = fr.face_encodings(img1test)[0]
cv2.rectangle(img1test, (facetest[3], facetest[0]), (facetest[1], facetest[2]), (255, 0, 0), 4)

res = fr.compare_faces ([encodedimage], encodeTestImage)
print(res)
face_dis = fr.face_distance([encodedimage],encodeTestImage)
print(face_dis)


cv2.imshow("Ms dhoni", img1)
cv2.imshow("MS dhoni2", img1test)
cv2.waitKey()
cv2.destroyAllWindows()