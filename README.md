# two_WL_NTK

This Repositroy contains the code used to run the experiments for the Masterthesis "Neural Tangent Kernel for a Higher-order Graph
Convolutional Network".

To get results, run:
```
python prepare_for_dataloader.py
```
To download the datsets and prepare them for futher processing.
Then you can run:
```
python gradient_Descent_training.py
```
To get gradient descent training results.
Or you can run:
```
python calculate_NTK_gram_matrix.py
python Kernel_SVM.py
```
To calculate the M-2-WL and the GCN NTKs and run a SVM classifier using the calculated NTKs.