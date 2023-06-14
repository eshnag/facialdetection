Background
	Current methods to measure facial neuromotor disorders, asymmetry, and synkinesis are difficult to implement, expensive, and require subjective analysis from experts. This study is based on Emotrics- a software model that enables automatic facial landmark localization with the use of Machine Learning techniques. Emotrics place facial landmarks and measures on a front-facing image. However, this tool requires user verification of landmarks and is also prone to error, as it has been primarily tested on symmetrical faces. 
  Our project aims to identify facial paralysis and synkinesis conditions, using a similar yet more robust facial mapping technique. FaceMesh is a ML model for detecting key features from images. It is a 3D model of the face, providing around 468 points in the 3D coordinate plane. It plots these points on dynamic and static images. Using OpenCV, we process these images to detect key facial features and plot the FaceMesh. 
Hypothesis
  The use of FaceMesh will allow more accurate facial landmark-based measurement systems, to measure the degree and severity of facial paralysis, without subjective evaluation. 
  We aim to make such a model convenient for clinical use.
Materials & Methodology

  1. We gather a diverse sample set of images with both symmetrical and asymmetrical faces.
  2. Test our program on this sample set, and ensure the measurements are accurate for various types of facial features and abnormalities
  3. Analyze results
  4. Grade the photos with EFACE with the MEEI dataset
  5. Send data back to the clinic (Keck School of Medicine)

Materials:
  OpenCV Python library: enable image processing through computer vision tasks: object detection, facial recognition, real time tracking
  FaceMesh ML model 
  Mediapipe: supports facemesh model and face detection (Real-time supportive model)

References:
  https://github.com/google/mediapipe/wiki/MediaPipe-Face-Mesh
