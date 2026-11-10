# Evidence summary

All values are means over the species panel under spatial block cross-validation, computed by `scripts/run_sdm.py`. Regenerate with `python scripts/run_sdm.py --species all`.

| model         |   mean_auc_spatial |   mean_tss_spatial |   mean_auc_random |   auc_inflation_random_minus_spatial |   mean_area_frac |   mean_runtime_s |   auc_at_smallest_n |   auc_at_largest_n |   auc_gain_with_n |   mean_rank_auc |
|:--------------|-------------------:|-------------------:|------------------:|-------------------------------------:|-----------------:|-----------------:|--------------------:|-------------------:|------------------:|----------------:|
| MaxEnt        |              0.789 |              0.376 |             0.89  |                                0.101 |            0.24  |             59   |               0.812 |              0.784 |            -0.028 |             2   |
| Random forest |              0.774 |              0.365 |             0.885 |                                0.111 |            0.214 |             17.6 |               0.762 |              0.774 |             0.012 |             1.3 |
| SVM (RBF)     |              0.735 |              0.313 |             0.865 |                                0.131 |            0.51  |              1.9 |               0.588 |              0.757 |             0.169 |             3.7 |
| XGBoost       |              0.715 |              0.368 |             0.883 |                                0.168 |            0.254 |              6.5 |               0.679 |              0.774 |             0.095 |             3   |

- Highest mean spatial AUC: MaxEnt (0.789).
- Largest random-to-blocked AUC drop: XGBoost (+0.168), which is the share of its accuracy that random folds hand it for free.
- Steepest gain from more presences: SVM (RBF) (+0.169 AUC from the smallest to the largest training set).
