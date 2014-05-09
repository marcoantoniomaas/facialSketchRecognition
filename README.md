facialSketchRecognition
=======================

Methods of facial recognition through sketches.

List of Methods and Descriptors
-----------

1. Descriptors

* HOG
* HAOG
* SIFT
* LBP
* MLBP
* GS
* LRBP

2. Methods

* Local Feature-based Discriminant Analysis
* Heterogeneous Prototype Framework

Requeriments
-----------

* OpenCV >= 2.4.2 (apply the lda.patch on the OpenCV modules/contrib/src/lda.cpp)
* VLfeat >= 0.9.17 (Temporally)
* CMake

Download and Installation
-----------

    cd $PATH
    git clone https://github.com/estranho/facialSketchRecognition.git
    mkdir $PATH/facialSketcheRecognition/build
    cd facialSketcheRecognition/build
    cmake ..
    make

Contributing
------------

1. Fork it.
2. Create a branch (`git checkout -b my_facialSketchRecognition`)
3. Commit your changes (`git commit -am "Added Something"`)
4. Push to the branch (`git push origin my_facialSketchRecognition`)
5. Open a [Pull Request][1]

[1]: https://github.com/estranho/facialSketchRecognition/pulls


References
------------

Déniz, O., Bueno, G., Salido, J. and De la Torre, F.: 2011, Face recognition using
histograms of oriented gradients, Pattern Recognition Letters 32(12), 1598–1603.

Kiani Galoogahi, H. and Sim, T.: 2012a, Face photo retrieval by sketch example, Proce-
edings of the 20th ACM international conference on Multimedia, ACM, pp. 949–952.

Kiani Galoogahi, H. and Sim, T.: 2012b, Face sketch recognition by local radon binary
pattern: Lrbp, Image Processing (ICIP), 2012 19th IEEE International Conference
on, IEEE, pp. 1837–1840.

Kiani Galoogahi, H. and Sim, T.: 2012c, Inter-modality face sketch recognition, Multi-
media and Expo (ICME), 2012 IEEE International Conference on, IEEE, pp. 224–229.

Klare, B., Li, Z. and Jain, A.: 2011, Matching forensic sketches to mug shot photos,
Pattern Analysis and Machine Intelligence, IEEE Transactions on 33(3), 639–646.

Klare, B. and Jain, A. K.: 2013, Heterogeneous face recognition using kernel prototype
similarities, Pattern Analysis and Machine Intelligence, IEEE Transactions on 2, 6.

