# bone-segmentation
group model training for femur segmentation

This project implements a deep learning pipeline for segmenting knee X-ray images using a UNet-based architecture (UNetLext). The dataset is split into training, validation, and test sets, where training and validation images include corresponding masks. The model is trained using a combination of Binary Cross-Entropy and Dice loss to optimize segmentation performance. During training, the validation Dice score is monitored, and the best-performing model weights are automatically saved to ensure reliable inference on unseen images.

For evaluation, a separate test set of X-ray images without masks is used. The saved best model is loaded, and predictions are generated for each test image, producing segmentation masks that can be saved and visualized.