## Domain Generalization via Ensemble Stacking for Face Presentation Attack Detection
#### Authors: Usman Muhammad, Jorma Laaksonen, Djamila Romaissa Beddiar and Mourad Oussalah


##

### Abstract
Face Presentation Attack Detection (PAD) plays a pivotal role in securing face recognition systems against spoofing attacks. Although great progress has been made in designing face PAD methods, developing a model that can generalize well to unseen test domains remains a significant challenge. Moreover, due to different types of spoofing attacks, creating a dataset with a sufficient number of samples for training deep neural networks is a laborious task. This work proposes a comprehensive solution that combines synthetic data generation and deep ensemble learning to enhance the generalization capabilities of face PAD. Specifically, synthetic data is generated by blending a static image with spatiotemporal encoded images using alpha composition and video distillation. This way, we simulate motion blur with varying alpha values, thereby generating diverse subsets of synthetic data that contribute to a more enriched training set. Furthermore, multiple base models are trained on each subset of synthetic data using stacked ensemble learning. This allows the models to learn complementary features and representations from different synthetic subsets. The meta-features generated by the base models are used as input to a new model called the meta-model. The latter combines the predictions from the base models, leveraging their complementary information to better handle unseen target domains and enhance the overall performance. Experimental results on four datasets demonstrate low half total error rates (HTERs) on three benchmark datasets: CASIA-MFSD (8.92%), MSU-MFSD (4.81%), and OULU-NPU (6.70%). The approach shows potential for advancing presentation attack detection by utilizing large-scale synthetic data and the meta-model.

• Preprint: https://arxiv.org/pdf/2301.02145v2.pdf
