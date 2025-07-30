CV ROC AUC scores: [0.98970864 0.98991605 0.9894716  0.99188642 0.99505679]
Mean CV ROC AUC: 0.9912079012345678
Confusion Matrix:
 [[704  46]
 [  3 747]]
              precision    recall  f1-score   support

           0      0.996     0.939     0.966       750
           1      0.942     0.996     0.968       750

    accuracy                          0.967      1500
   macro avg      0.969     0.967     0.967      1500
weighted avg      0.969     0.967     0.967      1500

Test ROC AUC: 0.9923822222222222
/usr/local/lib/python3.11/dist-packages/sklearn/calibration.py:333: UserWarning: The `cv='prefit'` option is deprecated in 1.6 and will be removed in 1.8. You can use CalibratedClassifierCV(FrozenEstimator(estimator)) instead.
  warnings.warn(
Test ROC AUC (Calibrated): 0.989584888888889
