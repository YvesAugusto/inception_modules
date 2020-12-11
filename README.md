# inception_modules
Implements inception modules based on GoogLeNet convolutional layers.
Inception module combines the output of multiple convolutional layers outputing these tensors concatenated. It was noticed that this kind of combination spend three times less computation than the traditional convolutional method.

### Inception module scheme
![inception_module](https://user-images.githubusercontent.com/53539227/101893816-0aeda380-3b84-11eb-8d3a-4a5cf1f0cd91.png)

### Paper
C. Szegedy et al., "Going deeper with convolutions," 2015 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), Boston, MA, 2015, pp. 1-9, doi: 10.1109/CVPR.2015.7298594.
