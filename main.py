# General
from tuning import *

final_clf = v_clf

final_clf.fit(X_train_DT, Y_train)

# Final Subission Preds
submission_preds = final_clf.predict(X_test_DT)

# Export Submission Data
submission_df = pd.DataFrame({
    'PassengerId': test_passenger_ids,
    'Survived': submission_preds
})

submission_df.to_csv('submission.csv', index=False)