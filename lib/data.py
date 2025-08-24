import hashlib
from collections import Counter
from copy import deepcopy
from dataclasses import astuple, dataclass, replace
from pathlib import Path
from typing import Any, Optional, Union, cast, Dict, List, Tuple
try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

import numpy as np
import pandas as pd
import sklearn.preprocessing
import torch
from category_encoders import LeaveOneOutEncoder
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from . import env, util
from .metrics import calculate_metrics as calculate_metrics_
from .util import TaskType

ArrayDict = Dict[str, np.ndarray]
TensorDict = Dict[str, torch.Tensor]

CAT_MISSING_VALUE = '__nan__'
CAT_RARE_VALUE = '__rare__'
Normalization = Literal['standard', 'quantile']
NumNanPolicy = Literal['drop-rows', 'mean']
CatNanPolicy = Literal['most_frequent']
CatEncoding = Literal['one-hot', 'counter']
YPolicy = Literal['default']

class StandardScaler1d(StandardScaler):
    def partial_fit(self, X, *args, **kwargs):
        assert X.ndim == 1
        return super().partial_fit(X[:, None], *args, **kwargs)

    def transform(self, X, *args, **kwargs):
        assert X.ndim == 1
        return super().transform(X[:, None], *args, **kwargs).squeeze(1)

    def inverse_transform(self, X, *args, **kwargs):
        assert X.ndim == 1
        return super().inverse_transform(X[:, None], *args, **kwargs).squeeze(1)

def get_category_sizes(X: Union[torch.Tensor, np.ndarray]) -> List[int]:
    XT = X.T.cpu().tolist() if isinstance(X, torch.Tensor) else X.T.tolist()
    return [len(set(x)) for x in XT]

@dataclass(frozen=False)
class Dataset:
    X_num: Optional[ArrayDict]
    X_cat: Optional[ArrayDict]
    y: ArrayDict
    y_info: Dict[str, Any]
    task_type: TaskType
    n_classes: Optional[int]
    num_feature_names: Optional[List[str]] = None
    cat_feature_names: Optional[List[str]] = None

    @classmethod
    def from_dir(
        cls, 
        dir_: Union[Path, str], 
        sample_size: int = None, 
        indices_dir: str = None,
        selected_features: list = None
    ) -> 'Dataset':
        dir_ = Path(dir_)
        print(f"Loading data from: {dir_}")
        df = pd.read_csv(dir_ / 'corrected_permacts.csv')
        print(f"Initial DataFrame shape: {df.shape}")
        print("[DEBUG][DATA] DataFrame columns:", df.columns.tolist())
        df = df.dropna()
        if 'Unnamed: 0' in df.columns:
            df = df.drop('Unnamed: 0', axis=1)
        print(f"Shape after dropping NaNs: {df.shape}")
        df = df.drop(['pkgname'], axis=1)
        print(f"Shape after dropping pkgname: {df.shape}")

        # === ENFORCE FEATURE LIST AND ORDER ===
        if selected_features is not None:
            # Add a check to ensure all features exist
            missing = [f for f in selected_features if f not in df.columns]
            if missing:
                raise ValueError(f"Missing features in DataFrame: {missing}")
            print("[DEBUG][DATA] Selected features:", selected_features)
            df = df[selected_features + ['status']]  # keep target
            print("[DEBUG][DATA] DataFrame columns after selection:", df.columns.tolist())

        # === Subset using provided indices (for both subset and full dataset) ===
        if indices_dir is not None:
            # Save the original index before resetting
            df['orig_index'] = df.index
            df = df.reset_index(drop=True)
            orig_index_to_new = dict(zip(df['orig_index'], df.index))
            # Load indices based on sample_size or use "full" for full dataset
            if sample_size is not None:
                train_idx = np.load(f"{indices_dir}/train_indices_{sample_size}.npy")
                val_idx = np.load(f"{indices_dir}/val_indices_{sample_size}.npy")
                test_idx = np.load(f"{indices_dir}/test_indices_{sample_size}.npy")
                print(f"Loaded indices for sample size {sample_size}:")
            else:
                train_idx = np.load(f"{indices_dir}/train_indices_full.npy")
                val_idx = np.load(f"{indices_dir}/val_indices_full.npy")
                test_idx = np.load(f"{indices_dir}/test_indices_full.npy")
                print(f"Loaded indices for full dataset:")
            print(f"  train: {train_idx.shape}, val: {val_idx.shape}, test: {test_idx.shape}")
            # Map original indices to new positions
            train_pos = [orig_index_to_new[i] for i in train_idx if i in orig_index_to_new]
            val_pos = [orig_index_to_new[i] for i in val_idx if i in orig_index_to_new]
            test_pos = [orig_index_to_new[i] for i in test_idx if i in orig_index_to_new]
            df_train = df.iloc[train_pos]
            df_val = df.iloc[val_pos]
            df_test = df.iloc[test_pos]
            X_train, y_train = df_train.drop(['status', 'orig_index'], axis=1), df_train['status']
            X_val, y_val = df_val.drop(['status', 'orig_index'], axis=1), df_val['status']
            X_test, y_test = df_test.drop(['status', 'orig_index'], axis=1), df_test['status']
        else:
            X = df.drop(['status'], axis=1)
            y = df['status']
            X_train, X_temp, y_train, y_temp = train_test_split(
                X, y, test_size=0.3, random_state=42, stratify=y
            )
            X_val, X_test, y_val, y_test = train_test_split(
                X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
            )

        print("[DEBUG][DATA] X_train columns:", X_train.columns.tolist())
        print("[DEBUG][DATA] X_val columns:", X_val.columns.tolist())
        print("[DEBUG][DATA] X_test columns:", X_test.columns.tolist())

        # Store feature names before converting to numpy arrays
        if selected_features is not None:
            all_types = df.dtypes
            num_features = [f for f in selected_features if all_types[f] in ['int64', 'float64']]
            cat_features = [f for f in selected_features if all_types[f] == 'object']
            print("Final feature order used for training (should match XGBoost):")
            print(selected_features)
            print("Numerical features (in order):", num_features)
            print("Categorical features (in order):", cat_features)
        else:
            num_features = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
            cat_features = X_train.select_dtypes(include=['object']).columns.tolist()

        X_num = {
            'train': X_train.select_dtypes(include=['int64', 'float64']).values.astype(np.float32),
            'val': X_val.select_dtypes(include=['int64', 'float64']).values.astype(np.float32),
            'test': X_test.select_dtypes(include=['int64', 'float64']).values.astype(np.float32)
        }
        X_cat = {
            'train': X_train.select_dtypes(include=['object']).values,
            'val': X_val.select_dtypes(include=['object']).values,
            'test': X_test.select_dtypes(include=['object']).values
        }
        print("[DEBUG][DATA] num_features:", num_features)
        print("[DEBUG][DATA] cat_features:", cat_features)
        print("[DEBUG][DATA] X_num shapes:", {k: v.shape for k, v in X_num.items()})
        print("[DEBUG][DATA] X_cat shapes:", {k: v.shape for k, v in X_cat.items()})

        y_dict = {
            'train': y_train.values,
            'val': y_val.values,
            'test': y_test.values
        }
        print("Final feature order used for training:")
        print(list(X_train.columns))
        print("[DEBUG][DATA] Dataset.num_feature_names:", num_features)
        print("[DEBUG][DATA] Dataset.cat_feature_names:", cat_features)
        return Dataset(
            X_num=X_num,
            X_cat=X_cat,
            y=y_dict,
            y_info={},
            task_type=TaskType.BINCLASS,
            n_classes=2,
            num_feature_names=num_features,
            cat_feature_names=cat_features
        )

    @property
    def is_binclass(self) -> bool:
        return self.task_type == TaskType.BINCLASS

    @property
    def is_multiclass(self) -> bool:
        return self.task_type == TaskType.MULTICLASS

    @property
    def is_regression(self) -> bool:
        return self.task_type == TaskType.REGRESSION

    @property
    def n_num_features(self) -> int:
        return 0 if self.X_num is None else self.X_num['train'].shape[1]

    @property
    def n_cat_features(self) -> int:
        return 0 if self.X_cat is None else self.X_cat['train'].shape[1]

    @property
    def n_features(self) -> int:
        return self.n_num_features + self.n_cat_features

    def size(self, part: Optional[str]) -> int:
        return sum(map(len, self.y.values())) if part is None else len(self.y[part])

    @property
    def nn_output_dim(self) -> int:
        if self.is_multiclass:
            assert self.n_classes is not None
            return self.n_classes
        else:
            return 1

    def get_category_sizes(self, part: str) -> List[int]:
        return [] if self.X_cat is None else get_category_sizes(self.X_cat[part])

    def calculate_metrics(
        self,
        predictions: Dict[str, np.ndarray],
        prediction_type: Optional[str],
    ) -> Dict[str, Any]:
        metrics = {
            x: calculate_metrics_(
                self.y[x], predictions[x], self.task_type, prediction_type, self.y_info
            )
            for x in predictions
        }
        if self.task_type == TaskType.REGRESSION:
            score_key = 'rmse'
            score_sign = -1
        else:
            score_key = 'accuracy'
            score_sign = 1
        for part_metrics in metrics.values():
            part_metrics['score'] = score_sign * part_metrics[score_key]
        return metrics

def num_process_nans(dataset: Dataset, policy: Optional[NumNanPolicy]) -> Dataset:
    assert dataset.X_num is not None
    nan_masks = {k: np.isnan(v) for k, v in dataset.X_num.items()}
    if not any(x.any() for x in nan_masks.values()):  # type: ignore[code]
        assert policy is None
        return dataset

    assert policy is not None
    if policy == 'drop-rows':
        valid_masks = {k: ~v.any(1) for k, v in nan_masks.items()}
        assert valid_masks[
            'test'
        ].all(), 'Cannot drop test rows, since this will affect the final metrics.'
        new_data = {}
        for data_name in ['X_num', 'X_cat', 'y']:
            data_dict = getattr(dataset, data_name)
            if data_dict is not None:
                new_data[data_name] = {
                    k: v[valid_masks[k]] for k, v in data_dict.items()
                }
        dataset = replace(dataset, **new_data)
    elif policy == 'mean':
        new_values = np.nanmean(dataset.X_num['train'], axis=0)
        X_num = deepcopy(dataset.X_num)
        for k, v in X_num.items():
            num_nan_indices = np.where(nan_masks[k])
            v[num_nan_indices] = np.take(new_values, num_nan_indices[1])
        dataset = replace(dataset, X_num=X_num)
    else:
        assert util.raise_unknown('policy', policy)
    return dataset

def normalize(
    X: ArrayDict, normalization: Normalization, seed: Optional[int]
) -> ArrayDict:
    X_train = X['train']
    if normalization == 'standard':
        normalizer = sklearn.preprocessing.StandardScaler()
    elif normalization == 'quantile':
        normalizer = sklearn.preprocessing.QuantileTransformer(
            output_distribution='normal',
            n_quantiles=max(min(X['train'].shape[0] // 30, 1000), 10),
            subsample=1e9,
            random_state=seed,
        )
        noise = 1e-3
        if noise > 0:
            assert seed is not None
            stds = np.std(X_train, axis=0, keepdims=True)
            noise_std = noise / np.maximum(stds, noise)  # type: ignore[code]
            X_train = X_train + noise_std * np.random.default_rng(seed).standard_normal(
                X_train.shape
            )
    else:
        util.raise_unknown('normalization', normalization)
    normalizer.fit(X_train)
    return {k: normalizer.transform(v) for k, v in X.items()}  # type: ignore[code]

def cat_process_nans(X: ArrayDict, policy: Optional[CatNanPolicy]) -> ArrayDict:
    assert X is not None
    nan_masks = {k: v == CAT_MISSING_VALUE for k, v in X.items()}
    if any(x.any() for x in nan_masks.values()):  # type: ignore[code]
        if policy is None:
            X_new = X
        elif policy == 'most_frequent':
            imputer = SimpleImputer(missing_values=CAT_MISSING_VALUE, strategy=policy)  # type: ignore[code]
            imputer.fit(X['train'])
            X_new = {k: cast(np.ndarray, imputer.transform(v)) for k, v in X.items()}
        else:
            util.raise_unknown('categorical NaN policy', policy)
    else:
        assert policy is None
        X_new = X
    return X_new

def cat_drop_rare(X: ArrayDict, min_frequency: float) -> ArrayDict:
    assert 0.0 < min_frequency < 1.0
    min_count = round(len(X['train']) * min_frequency)
    X_new = {x: [] for x in X}
    for column_idx in range(X['train'].shape[1]):
        counter = Counter(X['train'][:, column_idx].tolist())
        popular_categories = {k for k, v in counter.items() if v >= min_count}
        for part in X_new:
            X_new[part].append(
                [
                    (x if x in popular_categories else CAT_RARE_VALUE)
                    for x in X[part][:, column_idx].tolist()
                ]
            )
    return {k: np.array(v).T for k, v in X_new.items()}

def cat_encode(
    X: ArrayDict,
    encoding: Optional[CatEncoding],
    y_train: Optional[np.ndarray],
    seed: Optional[int],
) -> Tuple[ArrayDict, bool]:  # (X, is_converted_to_numerical)
    if encoding != 'counter':
        y_train = None

    # Step 1. Map strings to 0-based ranges
    unknown_value = np.iinfo('int64').max - 3
    encoder = sklearn.preprocessing.OrdinalEncoder(
        handle_unknown='use_encoded_value',  # type: ignore[code]
        unknown_value=unknown_value,  # type: ignore[code]
        dtype='int64',  # type: ignore[code]
    ).fit(X['train'])
    X = {k: encoder.transform(v) for k, v in X.items()}
    max_values = X['train'].max(axis=0)
    for part in ['val', 'test']:
        for column_idx in range(X[part].shape[1]):
            X[part][X[part][:, column_idx] == unknown_value, column_idx] = (
                max_values[column_idx] + 1
            )

    # Step 2. Encode.
    if encoding is None:
        return (X, False)
    elif encoding == 'one-hot':
        encoder = sklearn.preprocessing.OneHotEncoder(
            handle_unknown='ignore', sparse=False, dtype=np.float32  # type: ignore[code]
        )
        encoder.fit(X['train'])
        return ({k: encoder.transform(v) for k, v in X.items()}, True)  # type: ignore[code]
    elif encoding == 'counter':
        assert y_train is not None
        assert seed is not None
        encoder = LeaveOneOutEncoder(sigma=0.1, random_state=seed, return_df=False)
        encoder.fit(X['train'], y_train)
        X = {k: encoder.transform(v).astype('float32') for k, v in X.items()}  # type: ignore[code]
        if not isinstance(X['train'], pd.DataFrame):
            X = {k: v.values for k, v in X.items()}  # type: ignore[code]
        return (X, True)  # type: ignore[code]
    else:
        util.raise_unknown('encoding', encoding)

def build_target(
    y: ArrayDict, policy: Optional[YPolicy], task_type: TaskType
) -> Tuple[ArrayDict, Dict[str, Any]]:
    info: Dict[str, Any] = {'policy': policy}
    if policy is None:
        pass
    elif policy == 'default':
        if task_type == TaskType.REGRESSION:
            mean, std = float(y['train'].mean()), float(y['train'].std())
            y = {k: (v - mean) / std for k, v in y.items()}
            info['mean'] = mean
            info['std'] = std
    else:
        util.raise_unknown('policy', policy)
    return y, info

@dataclass(frozen=True)
class Transformations:
    seed: int = 0
    normalization: Optional[Normalization] = None
    num_nan_policy: Optional[NumNanPolicy] = None
    cat_nan_policy: Optional[CatNanPolicy] = None
    cat_min_frequency: Optional[float] = None
    cat_encoding: Optional[CatEncoding] = None
    y_policy: Optional[YPolicy] = 'default'

def transform_dataset(
    dataset: Dataset,
    transformations: Transformations,
    cache_dir: Optional[Path],
) -> Dataset:
    # WARNING: the order of transformations matters. Moreover, the current
    # implementation is not ideal in that sense.
    if cache_dir is not None:
        transformations_md5 = hashlib.md5(
            str(transformations).encode('utf-8')
        ).hexdigest()
        transformations_str = '__'.join(map(str, astuple(transformations)))
        cache_path = (
            cache_dir / f'cache__{transformations_str}__{transformations_md5}.pickle'
        )
        if cache_path.exists():
            cache_transformations, value = util.load_pickle(cache_path)
            if transformations == cache_transformations:
                print(
                    f"Using cached features: {cache_dir.name + '/' + cache_path.name}"
                )
                return value
            else:
                raise RuntimeError(f'Hash collision for {cache_path}')
    else:
        cache_path = None

    if dataset.X_num is not None:
        dataset = num_process_nans(dataset, transformations.num_nan_policy)

    X_num = dataset.X_num
    if dataset.X_cat is None:
        replace(transformations, cat_nan_policy=None, cat_min_frequency=None, cat_encoding=None)
        # assert transformations.cat_nan_policy is None
        # assert transformations.cat_min_frequency is None
        # assert transformations.cat_encoding is None
        X_cat = None
    else:
        X_cat = cat_process_nans(dataset.X_cat, transformations.cat_nan_policy)
        if transformations.cat_min_frequency is not None:
            X_cat = cat_drop_rare(X_cat, transformations.cat_min_frequency)
        X_cat, is_num = cat_encode(
            X_cat,
            transformations.cat_encoding,
            dataset.y['train'],
            transformations.seed,
        )
        if is_num:
            X_num = (
                X_cat
                if X_num is None
                else {x: np.hstack([X_num[x], X_cat[x]]) for x in X_num}
            )
            X_cat = None

    if X_num is not None and transformations.normalization is not None:
        X_num = normalize(X_num, transformations.normalization, transformations.seed)

    y, y_info = build_target(dataset.y, transformations.y_policy, dataset.task_type)

    dataset = replace(dataset, X_num=X_num, X_cat=X_cat, y=y, y_info=y_info)
    if cache_path is not None:
        util.dump_pickle((transformations, dataset), cache_path)
    return dataset

def build_dataset(
    path: Union[str, Path], transformations: Transformations, cache: bool,
    sample_size: int = None, indices_dir: str = None, selected_features: list = None
) -> Dataset:
    path = Path(path)
    dataset = Dataset.from_dir(
        path, 
        sample_size=sample_size, 
        indices_dir=indices_dir, 
        selected_features=selected_features
    )
    return transform_dataset(dataset, transformations, path if cache else None)

def prepare_tensors(
    dataset: Dataset, device: Union[str, torch.device]
) -> Tuple[Optional[TensorDict], Optional[TensorDict], TensorDict]:
    print(f"dataset.X_num['train'].shape: {dataset.X_num['train'].shape}")
    if isinstance(device, str):
        device = torch.device(device)
    X_num, X_cat, Y = (
        None if x is None else {k: torch.as_tensor(v) for k, v in x.items()}
        for x in [dataset.X_num, dataset.X_cat, dataset.y]
    )
    if device.type != 'cpu':
        X_num, X_cat, Y = (
            None if x is None else {k: v.to(device) for k, v in x.items()}
            for x in [X_num, X_cat, Y]
        )
    assert X_num is not None
    assert Y is not None
    if not dataset.is_multiclass:
        Y = {k: v.float() for k, v in Y.items()}
    print(f"X_num shape in prepare_tensors: {X_num['train'].shape}")
    return X_num, X_cat, Y

def load_dataset_info(dataset_dir_name: str) -> Dict[str, Any]:
    path = env.DATA / dataset_dir_name
    info = util.load_json(path / 'info.json')
    info['size'] = info['train_size'] + info['val_size'] + info['test_size']
    info['n_features'] = info['n_num_features'] + info['n_cat_features']
    info['path'] = path
    return info