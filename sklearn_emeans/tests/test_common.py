from sklearn.utils.estimator_checks import parametrize_with_checks

from sklearn_emeans import EMeans


@parametrize_with_checks(
    [
        EMeans(),
        EMeans(n_clusters=2, n_epochs=2),
        EMeans(init="k-means||"),
        EMeans(revival="greedy"),
    ]
)
def test_sklearn_compatible_estimator(estimator, check):
    check(estimator)
