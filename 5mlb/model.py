from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import LogisticRegression


# custom concatenation transformer
class ConcatTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X_ = X.copy()
        X_.fillna('', inplace=True)
        new_column_name = '_'.join(X_.columns)
        X_[new_column_name] = X_.agg(' '.join, axis=1)
        return X_[new_column_name]


numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value=0)),
])

text_transformer = Pipeline(steps=[
    ('concat', ConcatTransformer()),
    # all-in-one: tokenizer, stop-words remover, hasher, normalizer
    ('hasher', HashingVectorizer(n_features=10, stop_words='english', binary=False))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, ['vote']),
        ('text', text_transformer, ['reviewText', 'summary'])
    ],
    remainder='drop'
)

# model = Pipeline(steps=[
#     ('preproc', preprocessor),
#     ('logreg', LogisticRegression(max_iter=10000))
# ])

model = Pipeline(steps=[
    ('preproc', text_transformer),
    ('logreg', LogisticRegression(max_iter=10000))
])
