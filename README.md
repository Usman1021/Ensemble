## [Domain Generalization via Ensemble Stacking for Face Presentation Attack Detection]
#### Authors: Usman Muhammad, Jorma Laaksonen, Djamila Romaissa Beddiar and Mourad Oussalah


##

### Abstract
Face Presentation Attack Detection (PAD) plays a pivotal role in securing face recognition systems against spoofing attacks. Although great progress has been made in designing face PAD methods, developing a model that can generalize well to unseen test domains remains a significant challenge. Moreover, due to different types of spoofing attacks, creating a dataset with a sufficient number of samples for training deep neural networks is a laborious task. This work proposes a comprehensive solution that combines synthetic data generation and deep ensemble learning to enhance the generalization capabilities of Face PAD systems. Specifically, synthetic data is generated by blending static image with spatiotemporal encoded images using alpha composition and video distillation. This way, we simulate motion blur with varying alpha values, thereby generating diverse subsets of synthetic data that contribute to a more enriched training set. Furthermore, multiple base models are trained on each subset of synthetic data using stacked ensemble learning. This allows the models to learn complementary features and representations from different synthetic subsets. The meta-features generated by the base models are used as input to a new model called the meta-model. The latter combines the predictions from the base models, leveraging their complementary information to better handle unseen target domains and enhance the overall performance. Experimental results on four PAD databases demonstrate the robustness of the proposed method, as evidenced by low half-total error rates (HTERs) on each dataset, such as  CASIA-MFSD (7.01 %), Replay-Attack (36.76 %), MSU-MFSD (3.70 %), and OULU-NPU (8.32 %). The approach shows promise in advancing presentation attack detection using large-scale synthetic data and the meta-model.





### Links


• Preprint: (https://arxiv.org/pdf/2301.02145.pdf)

### Links
• Main link: https://www.mdpi.com/2072-4292/14/23/5913

• Preprint: https://www.techrxiv.org/articles/preprint/16441593
