import json
from dataclasses import dataclass
from typing import Optional, Sequence, Dict, Literal


@dataclass
class PreprocessConfig:
    """Class to represent parameters for any configurable part of the filtering process."""
    filter_too_many_columns: bool = False  # to support legacy filtering; if True, tables are dropped if num_cols > max_cols.

    # Table-level filters
    max_frac_numeric_colnames: Optional[float] = 0.1  # if None, this threshold is not applied.
    langdetect_threshold: float = 0.5
    min_rows: int = 100
    max_output_rows: int = 1000
    max_frac_unnamed_columns: Optional[float] = 0.5
    drop_extra_rows: bool = True
    min_cols: int = 6
    max_cols: int = 512
    min_dtypes: Optional[int] = None,
    pii_detect_filter_threshold: Optional[
        float] = None  # if any columns has >= this proportion of PII detected, drop the table
    code_detect_filter_threshold: Optional[
        float] = None  # if any columns has >= this proportion of code detected, drop the table

    table_quality_threshold: Optional[float] = None
    # table_quality_classifier: Optional[str] = None
    table_quality_classifier: Optional[
        str] = None
    # Whether the classifier gets applied before row/column filtering ("pre") or after ("post").
    table_quality_classifier_position: Literal["pre", "post"] = "post"

    # Column-level filters
    drop_invalid_cols: bool = True
    drop_extra_cols: bool = False  # if True, dataframes with num_cols > max_cols are randomly downsampled to max_cols.
    max_header_len_chars: int = 256
    min_unique_column_values: int = 2

    # Row-level filters
    max_value_len_chars: Optional[int] = 14336  # if None, this threshold is not applied. Note: 14336 = int(4096 * 3.5).
    filter_rows_containing_substrings: Optional[Sequence[str]] = ("@@@", "###", "└─")
    filter_rows_containing_code: bool = True
    filter_rows_containing_pii: bool = True
    drop_duplicate_rows: bool = True

    # More complex filters or filters that can operate at different levels
    max_null_like_frac: Optional[float] = 0.1  # if None, this threshold is not applied.

    def to_json(self, filename: str):
        with open(filename, "w") as f:
            json.dump(self.__dict__, f, indent=4, ensure_ascii=False, sort_keys=True)


PREPROCESS_VERSIONS: Dict[str, PreprocessConfig] = {
    # Note: specify all values explicitly so that defaults can be changed later without affecting preprocessors.
    "v3": PreprocessConfig(max_null_like_frac=None,
                           max_value_len_chars=None,
                           filter_too_many_columns=True,
                           drop_extra_cols=False,
                           max_frac_numeric_colnames=None,
                           drop_invalid_cols=False,
                           filter_rows_containing_substrings=None,
                           filter_rows_containing_pii=False,
                           filter_rows_containing_code=False,
                           drop_duplicate_rows=False,
                           max_frac_unnamed_columns=None,
                           min_dtypes=None, ),
    "v5": PreprocessConfig(max_null_like_frac=0.1,
                           drop_extra_cols=True,
                           drop_invalid_cols=True,
                           filter_rows_containing_substrings=None,
                           filter_rows_containing_pii=False,
                           filter_rows_containing_code=False,
                           drop_duplicate_rows=False,
                           max_frac_unnamed_columns=None,
                           min_dtypes=None,
                           ),
    # Note: prior to v6, there was a bug where the min_dtypes was not being properly applied
    # (effectively, all tables passed this check). Once the bug was fixed in v6, we set min_dtypes to None
    # for consistency.
    "v6": PreprocessConfig(max_null_like_frac=0.1,
                           drop_extra_cols=True,
                           drop_invalid_cols=True,
                           pii_detect_filter_threshold=0.05,
                           code_detect_filter_threshold=0.05,
                           filter_rows_containing_pii=True,
                           min_rows=64,
                           filter_rows_containing_code=True,
                           drop_duplicate_rows=True,
                           max_frac_unnamed_columns=0.5,
                           min_dtypes=None),
    "v7": PreprocessConfig(max_null_like_frac=0.1,
                           drop_extra_cols=True,
                           drop_invalid_cols=True,
                           pii_detect_filter_threshold=0.05,
                           code_detect_filter_threshold=0.05,
                           filter_rows_containing_pii=True,
                           min_rows=64,
                           filter_rows_containing_code=True,
                           drop_duplicate_rows=True,
                           max_frac_unnamed_columns=0.5,
                           min_dtypes=None,
                           table_quality_threshold=0.0055,
                           table_quality_classifier_position="pre",

                           ),
    "v8": PreprocessConfig(max_null_like_frac=0.1,
                           drop_extra_cols=True,
                           drop_invalid_cols=True,
                           pii_detect_filter_threshold=0.05,
                           code_detect_filter_threshold=0.05,
                           filter_rows_containing_pii=True,
                           min_rows=64,
                           filter_rows_containing_code=True,
                           drop_duplicate_rows=True,
                           max_frac_unnamed_columns=0.5,
                           min_dtypes=None,
                           table_quality_threshold=0.0055,
                           table_quality_classifier_position="post",
                           table_quality_classifier="/gscratch/efml/jpgard/tabliblib-official/tabliblib/xgb_quality_scorer.json",
                           ),
    "v8.1": PreprocessConfig(max_null_like_frac=0.1,
                             drop_extra_cols=True,
                             drop_invalid_cols=True,
                             pii_detect_filter_threshold=0.05,
                             code_detect_filter_threshold=0.05,
                             filter_rows_containing_pii=True,
                             min_rows=64,
                             filter_rows_containing_code=True,
                             drop_duplicate_rows=True,
                             max_frac_unnamed_columns=0.5,
                             min_dtypes=None,
                             table_quality_threshold=0.01,
                             table_quality_classifier_position="post",
                             table_quality_classifier="/gscratch/efml/jpgard/tabliblib-official/tabliblib/xgb_quality_scorer.json",

                             ),
    "v8.2": PreprocessConfig(max_null_like_frac=0.1,
                             drop_extra_cols=True,
                             drop_invalid_cols=True,
                             pii_detect_filter_threshold=0.05,
                             code_detect_filter_threshold=0.05,
                             filter_rows_containing_pii=True,
                             min_rows=64,
                             filter_rows_containing_code=True,
                             drop_duplicate_rows=True,
                             max_frac_unnamed_columns=0.5,
                             min_dtypes=None,
                             table_quality_threshold=0.02,
                             table_quality_classifier_position="post",
                             table_quality_classifier="/gscratch/efml/jpgard/tabliblib-official/tabliblib/xgb_quality_scorer.json",

                             ),
    "v8.3": PreprocessConfig(max_null_like_frac=0.1,
                             drop_extra_cols=True,
                             drop_invalid_cols=True,
                             pii_detect_filter_threshold=0.05,
                             code_detect_filter_threshold=0.05,
                             filter_rows_containing_pii=True,
                             min_rows=64,
                             filter_rows_containing_code=True,
                             drop_duplicate_rows=True,
                             max_frac_unnamed_columns=0.5,
                             min_dtypes=None,
                             table_quality_threshold=0.04,
                             table_quality_classifier_position="post",
                             table_quality_classifier="/gscratch/efml/jpgard/tabliblib-official/tabliblib/xgb_quality_scorer.json",

                             ),
    # "v9.0": PreprocessConfig(max_null_like_frac=0.1,
    #                          drop_extra_cols=True,
    #                          drop_invalid_cols=True,
    #                          pii_detect_filter_threshold=0.05,
    #                          code_detect_filter_threshold=0.05,
    #                          filter_rows_containing_pii=True,
    #                          min_rows=64,
    #                          filter_rows_containing_code=True,
    #                          drop_duplicate_rows=True,
    #                          max_frac_unnamed_columns=0.5,
    #                          min_dtypes=None,
    #                          table_quality_threshold=0.0002914782,
    #                          table_quality_classifier_position="post",
    #                          table_quality_classifier="/gscratch/efml/jpgard/tabliblib-official/tabliblib/table_quality_clf/xgb_table_quality_scorer_4571a1d6-67b0-4755-ac47-0980c878403a.json",
    #                          ),
    # "v9.1": PreprocessConfig(max_null_like_frac=0.1,
    #                          drop_extra_cols=True,
    #                          drop_invalid_cols=True,
    #                          pii_detect_filter_threshold=0.05,
    #                          code_detect_filter_threshold=0.05,
    #                          filter_rows_containing_pii=True,
    #                          min_rows=64,
    #                          filter_rows_containing_code=True,
    #                          drop_duplicate_rows=True,
    #                          max_frac_unnamed_columns=0.5,
    #                          min_dtypes=None,
    #                          table_quality_threshold=0.1192029193,
    #                          table_quality_classifier_position="post",
    #                          table_quality_classifier="/gscratch/efml/jpgard/tabliblib-official/tabliblib/table_quality_clf/xgb_table_quality_scorer_c13d2540-51fd-44e3-8ec9-83e26b99df1e.json",
    #                          ),
    # "v9.2": PreprocessConfig(max_null_like_frac=0.1,
    #                          drop_extra_cols=True,
    #                          drop_invalid_cols=True,
    #                          pii_detect_filter_threshold=0.05,
    #                          code_detect_filter_threshold=0.05,
    #                          filter_rows_containing_pii=True,
    #                          min_rows=64,
    #                          filter_rows_containing_code=True,
    #                          drop_duplicate_rows=True,
    #                          max_frac_unnamed_columns=0.5,
    #                          min_dtypes=None,
    #                          table_quality_threshold=0.0000462922,
    #                          table_quality_classifier_position="post",
    #                          table_quality_classifier="/gscratch/efml/jpgard/tabliblib-official/tabliblib/table_quality_clf/xgb_table_quality_scorer_444bef6c-bf0b-467b-9e41-6046b9abdd9d.json",
    #                          ),
"v9.0.1": PreprocessConfig(max_null_like_frac=0.1,
                             drop_extra_cols=True,
                             drop_invalid_cols=True,
                             pii_detect_filter_threshold=0.05,
                             code_detect_filter_threshold=0.05,
                             filter_rows_containing_pii=True,
                             min_rows=64,
                             filter_rows_containing_code=True,
                             drop_duplicate_rows=True,
                             max_frac_unnamed_columns=0.5,
                             min_dtypes=None,
                             table_quality_threshold=0.00137736471369862,
                             table_quality_classifier_position="post",
                             table_quality_classifier="/gscratch/efml/jpgard/tabliblib-official/tabliblib/table_quality_clf/xgb_table_quality_scorer_56248a1d-f211-4e36-8d99-db42c9bf35eb.json",
                             ),
    "v9.1.1": PreprocessConfig(max_null_like_frac=0.1,
                             drop_extra_cols=True,
                             drop_invalid_cols=True,
                             pii_detect_filter_threshold=0.05,
                             code_detect_filter_threshold=0.05,
                             filter_rows_containing_pii=True,
                             min_rows=64,
                             filter_rows_containing_code=True,
                             drop_duplicate_rows=True,
                             max_frac_unnamed_columns=0.5,
                             min_dtypes=None,
                             table_quality_threshold=0.00037273640627972766,
                             table_quality_classifier_position="post",
                             table_quality_classifier="/gscratch/efml/jpgard/tabliblib-official/tabliblib/table_quality_clf/xgb_table_quality_scorer_22b3aded-2da2-4a31-98b6-5fcfa9199e3e.json",
                             ),
    "v9.2.1": PreprocessConfig(max_null_like_frac=0.1,
                             drop_extra_cols=True,
                             drop_invalid_cols=True,
                             pii_detect_filter_threshold=0.05,
                             code_detect_filter_threshold=0.05,
                             filter_rows_containing_pii=True,
                             min_rows=64,
                             filter_rows_containing_code=True,
                             drop_duplicate_rows=True,
                             max_frac_unnamed_columns=0.5,
                             min_dtypes=None,
                             table_quality_threshold=0.004433658253401516,
                             table_quality_classifier_position="post",
                             table_quality_classifier="/gscratch/efml/jpgard/tabliblib-official/tabliblib/table_quality_clf/xgb_table_quality_scorer_a673d098-a9f7-4088-ae18-aa0c361e38bb.json",
                             ),
    "v9.3": PreprocessConfig(max_null_like_frac=0.1,
                             drop_extra_cols=True,
                             drop_invalid_cols=True,
                             pii_detect_filter_threshold=0.05,
                             code_detect_filter_threshold=0.05,
                             filter_rows_containing_pii=True,
                             min_rows=64,
                             filter_rows_containing_code=True,
                             drop_duplicate_rows=True,
                             max_frac_unnamed_columns=0.5,
                             min_dtypes=None,
                             table_quality_threshold=0.008328398689627647,
                             table_quality_classifier_position="post",
                             table_quality_classifier="/gscratch/efml/jpgard/tabliblib-official/tabliblib/table_quality_clf/xgb_table_quality_scorer_3527b5a4-4197-4b78-a571-efdf947e4c16.json",
                             ),
    "v9.4": PreprocessConfig(max_null_like_frac=0.1,
                             drop_extra_cols=True,
                             drop_invalid_cols=True,
                             pii_detect_filter_threshold=0.05,
                             code_detect_filter_threshold=0.05,
                             filter_rows_containing_pii=True,
                             min_rows=64,
                             filter_rows_containing_code=True,
                             drop_duplicate_rows=True,
                             max_frac_unnamed_columns=0.5,
                             min_dtypes=None,
                             table_quality_threshold=0.0005960157141089435,
                             table_quality_classifier_position="post",
                             table_quality_classifier="/gscratch/efml/jpgard/tabliblib-official/tabliblib/table_quality_clf/xgb_table_quality_scorer_ab676da7-ede7-4d38-91bb-e11f8e904629.json",

                             ),
    "v9.5": PreprocessConfig(max_null_like_frac=0.1,
                             drop_extra_cols=True,
                             drop_invalid_cols=True,
                             pii_detect_filter_threshold=0.05,
                             code_detect_filter_threshold=0.05,
                             filter_rows_containing_pii=True,
                             min_rows=64,
                             filter_rows_containing_code=True,
                             drop_duplicate_rows=True,
                             max_frac_unnamed_columns=0.5,
                             min_dtypes=None,
                             table_quality_threshold=0.0002231991820735857,
                             table_quality_classifier_position="post",
                             table_quality_classifier="/gscratch/efml/jpgard/tabliblib-official/tabliblib/table_quality_clf/xgb_table_quality_scorer_0f77dd03-4cc1-4fba-8c1f-6cb4217fe579.json",
                             ),
    # "Unfiltered" version of TabLib, except some filters
    # to ensure inputs do not exceed memory/disk limitations
    # and can fit into downstream context window.
    "baseline": PreprocessConfig(filter_too_many_columns=False,
                                 max_frac_numeric_colnames=1.,
                                 langdetect_threshold=0.,
                                 min_rows=1,
                                 max_output_rows=1000,
                                 max_frac_unnamed_columns=1,
                                 drop_extra_rows=True,
                                 min_cols=1,
                                 max_cols=2048,
                                 min_dtypes=None,
                                 pii_detect_filter_threshold=None,
                                 code_detect_filter_threshold=None,
                                 drop_invalid_cols=False,
                                 drop_extra_cols=False,
                                 max_header_len_chars=4096,
                                 min_unique_column_values=1,
                                 max_value_len_chars=None,
                                 filter_rows_containing_code=False,
                                 filter_rows_containing_pii=False,
                                 drop_duplicate_rows=False,
                                 max_null_like_frac=1.
                                 )
}
