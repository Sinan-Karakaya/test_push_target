This is a variant of the [[SMOTE (Synthetic Minority Oversampling Technique)]] that can treat mixed data (data that contains both numerical and categorical variables).

The ***SMOTE-NC*** doesn't create new samples for the categorical variables, but copy them from their neighbors. This avoid introducing noise and outliers in mixed datasets.