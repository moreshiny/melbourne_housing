# Melbourne Housing Prices Prediction

This notebook shows initial cleaning and feature engineering on the Melbourne Housing dataset.

The data is from Kaggle and can be found [here](https://www.kaggle.com/anthonypino/melbourne-housing-market)

Prices in the set are predicted using a Random Forest regressor.


```python
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate
from sklearn.impute import KNNImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import pandas_profiling as pdpr

import matplotlib.pyplot as pyplot
pyplot.rcParams['figure.dpi'] = 300
pyplot.rcParams['savefig.dpi'] = 300
```

# Data

First, let's laod the data and split of a test set to gage our performance
gainst at the end.


```python
full_data = pd.read_csv("./data/Melbourne_housing_FULL.csv")

```


```python
y = full_data.loc[:, 'Price']
X = full_data.drop(columns=['Price'])

train_size = 0.8

X_train, X_test, y_train, y_test = train_test_split(
    X, y, train_size=train_size, random_state=42
)

```

# Cleaning

We cannot predict anything where the target is missing. Add the target value
back to the dataset and drop all rows that have a missing target. Then separate
the features and target again.


```python
train_full = pd.concat([X_train, y_train], axis=1)
train_with_target = train_full.dropna(subset=['Price'])

X_train = train_with_target.drop(columns=['Price'])
y_train = train_with_target.loc[:, 'Price']

```

A small helper to ease creating dummies for various variables.


```python
def make_dummies(df: pd.DataFrame, column: str) -> None:
    """Turns the indicated column into dummies, omitting the first. Removes
    the original column.

    Args:
        frame (pd.DataFrame): Input frame containing column.
        column (str): Name of the column to turn into dummies.

    Returns:
        None
    """
    dummies = pd.get_dummies(
        df.loc[:, column], prefix=column, drop_first=True)
    df = pd.concat([X_train, dummies], axis=1)
    df.drop(columns=[column], inplace=True)
    return df

```

Adress has too many unique values, we drop it. We also drop Method and
SellerG as these showed no effect during cross-validation.

We also drop Bedrooms2 since it is highly correlated with Rooms and shows no
effect in cross-validation.

Buiding Area has too many missing values, so gets dropped too.


```python
print(X_train.loc[:, 'Address'].value_counts())
X_train.drop(columns=['Address'], inplace=True)

X_train.drop(columns=['Method'], inplace=True)
X_train.drop(columns=['SellerG'], inplace=True)

X_train.drop(columns=['Bedroom2'], inplace=True)

print("BuildingArea Missing", X_train.loc[:,'BuildingArea'].isnull().sum())
X_train.drop(columns=['BuildingArea'], inplace=True)

```

    5 Charles St        3
    14 James St         3
    36 Aberfeldie St    3
    14 Northcote St     3
    1/1 Clarendon St    3
                       ..
    91 Henty St         1
    12 Gordon Ct        1
    12/32 Argyle St     1
    11/165 Noone St     1
    22 Yongala St       1
    Name: Address, Length: 21475, dtype: int64
    BuildingArea Missing 13243


Type is categorical but not ordinal so we convert it to dummies.


```python
print(X_train.loc[:, 'Type'].value_counts())
X_train = make_dummies(X_train, 'Type')
```

    h    14866
    u     4675
    t     2276
    Name: Type, dtype: int64


We keep only the year sold from date, as Month showed no effect in cross-
validation.


```python
X_train.loc[:, 'Date'] = pd.to_datetime(X_train.loc[:, 'Date'])
X_train.loc[:, 'Year_sold'] = X_train.loc[:, 'Date'].dt.year
X_train.drop(columns=['Date'], inplace=True)
```

Distance only has one missing value, which we fill with the mean.


```python
# Distance -keep and fill one missing
print("Missing in Distance:", X_train.loc[:,'Distance'].isnull().sum())
median_dist = X_train.loc[:, 'Distance'].dropna().mean()
X_train.loc[:, 'Distance'].fillna(median_dist, inplace=True)
X_train.loc[:, 'Distance'] = np.log(X_train.loc[:, 'Distance']+1)
```

    Missing in Distance: 1


Bathroom is also correlated with Rooms (i.e. property size) but we can keep it
as Bathrooms_per_room which shows some significance in cross-validation after
filling missing values with the mode.


```python
print("Bathroom missing", X_train.loc[:, 'Bathroom'].isnull().sum())
mode_bathroom = X_train.loc[:, 'Bathroom'].mode()[0]
X_train.loc[:, 'Bathroom'].fillna(mode_bathroom, inplace=True)

X_train.loc[:, 'Bathroom_per_room'] =\
    X_train.loc[:, 'Bathroom'] / X_train.loc[:, 'Rooms']

X_train.drop(columns=['Bathroom'], inplace=True)
```

    Bathroom missing 5129


The same logic applies to Car.


```python
print("Car missing", X_train.loc[:, 'Car'].isnull().sum())

car_mode = X_train.loc[:, 'Car'].mode()[0]
X_train.loc[:, 'Car'].fillna(car_mode, inplace=True)

X_train.loc[:, 'Car_per_room'] =\
    X_train.loc[:, 'Car'] / X_train.loc[:, 'Rooms']
X_train.drop(columns=['Car'], inplace=True)
```

    Car missing 5441


Year built is mostly missing but where present it has a strong effect. To 
capture this effect we turn it into categories, with one for missing values.


```python
X_train.loc[:, 'YearBuilt'] = pd.cut(
    X_train.loc[:, 'YearBuilt'],
    bins=[0, 1800, 1900, 1945, 2000, 3000],
    labels=['1800s', '1900s', 'prewar', 'postwar', 'new'],
    ordered=False
).astype(str)

X_train.loc[:, 'YearBuilt'].fillna('unknown', inplace=True)

X_train = make_dummies(X_train, 'YearBuilt')
```

Location is crucial in real estate, so we do our best to fill missing 
location values from other available data, falling back to just the council
mean if closer mean cannot be calculated.


```python
mean_locs = X_train\
    .loc[:, ['CouncilArea', 'Postcode', 'Suburb', 'Lattitude', 'Longtitude']]\
    .groupby(['CouncilArea', 'Postcode', 'Suburb'])\
    .mean()

mean_locs_council = X_train\
    .loc[:, ['CouncilArea', 'Lattitude', 'Longtitude']]\
    .groupby(['CouncilArea'])\
    .mean()

msk_empty_locs =\
    (X_train.loc[:, 'Lattitude'].isnull()) |\
    (X_train.loc[:, 'Lattitude'].isnull())

for index, item in X_train.loc[msk_empty_locs, :].iterrows():
    # try to fill the mean matching that location
    try:
        X_train.loc[index, 'Lattitude'] = mean_locs\
            .loc[item['CouncilArea']]\
            .loc[item['Postcode']]\
            .loc[item['Suburb'], "Lattitude"]
        X_train.loc[index, 'Longtitude'] = mean_locs\
            .loc[item['CouncilArea']]\
            .loc[item['Postcode']]\
            .loc[item['Suburb'], 'Longtitude']
    except KeyError:
        pass

msk_empty_locs =\
    (X_train.loc[:, 'Lattitude'].isnull()) |\
    (X_train.loc[:, 'Lattitude'].isnull())

for index, item in X_train.loc[msk_empty_locs, :].iterrows():
    # try to fill based just on CouncilArea
    try:
        X_train.loc[index, 'Lattitude'] = mean_locs_council\
            .loc[item['CouncilArea'], 'Lattitude']
        X_train.loc[index, 'Longtitude'] = mean_locs_council\
            .loc[item['CouncilArea'], 'Longtitude']
    except KeyError:
        pass

# Fill any remaining missing with the overall mean
mean_lat = X_train.loc[:, 'Lattitude'].mean()
mean_long = X_train.loc[:, 'Longtitude'].mean()

X_train.loc[:, 'Lattitude'].fillna(mean_lat, inplace=True)
X_train.loc[:, 'Longtitude'].fillna(mean_long, inplace=True)

```

After imputing lat/long we no longer need the geographical categories.


```python
X_train.drop(columns=['CouncilArea', 'Postcode', 'Suburb'], inplace=True)
```

To capture interaction effects between lattitude, logitude, and distance we add
a normalised product of these. This may caputure specific city location effects
better than the individual terms.


```python
mean_lat = X_train.loc[:, 'Lattitude'].mean()
mean_long = X_train.loc[:, 'Longtitude'].mean()
mean_dist = X_train.loc[:, 'Distance'].mean()

X_train.loc[:, 'LatLong'] = (X_train.loc[:, 'Lattitude'] - mean_lat) * (
    X_train.loc[:, "Longtitude"] - mean_long) * (X_train.loc[:, 'Distance'] - mean_dist)
```

Distance no longer contributes in cross-validation after this change, so we
drop it later, but keep it for now to imput Landsize.

Landsize has many missing values holds some potential information. We correct
the outliers in the 99th percentile and fill missing values using nearest
neighbours by location and type of property.

To reduce the right skew we apply a log transformation.


```python
landsize_99p = np.percentile(X_train.loc[:, 'Landsize'].values, [99.0])[0]

msk_over99perc_landsize = X_train.loc[:, 'Landsize'] > landsize_99p
X_train.loc[msk_over99perc_landsize, 'Landsize'] = landsize_99p

imputer = KNNImputer(n_neighbors=100)

imputer.fit(X_train.loc[:, ['Type_t', 'Type_u', 'Lattitude',
                            'Longtitude', 'Distance', 'Landsize']]
            )
X_train.loc[:, 'Landsize'] = imputer\
    .transform(X_train.loc[:, ['Type_t', 'Type_u', 'Lattitude',
                               'Longtitude', 'Distance', 'Landsize']])[:, 5]

landsize_mean = X_train.loc[:, 'Landsize'].mean()
X_train.loc[:, 'Landsize'] = np.log(X_train.loc[:, 'Landsize'] + landsize_mean)
```

Now we can drop distance.


```python
X_train.drop(columns=['Distance'], inplace=True)
```

Region does not add any information beyond lat/long so we drop it.


```python
X_train.drop(columns=['Regionname'], inplace=True)
```

Propertycount only has a few missing values that we will with the median. We
also transform it to reduce the skewness.


```python
# Propertycount - keep this, already numerical - fill couple of missing with mean
median_pcount = X_train.loc[:, 'Propertycount'].mean()
X_train.loc[:, 'Propertycount'].fillna(median_pcount, inplace=True)
X_train.loc[:, 'Propertycount'] = np.sqrt(X_train.loc[:, 'Propertycount'])
```

Finally, the number of rooms is vital but again we transform it to reduce
skeweness.


```python
X_train.loc[:, 'Rooms'] = np.sqrt(X_train.loc[:, 'Rooms'])
```

That leaves us with a clean set of colums to train our model.


```python
print(X_train.columns)
```

    Index(['Rooms', 'Landsize', 'Lattitude', 'Longtitude', 'Propertycount',
           'Type_t', 'Type_u', 'Year_sold', 'Bathroom_per_room', 'Car_per_room',
           'YearBuilt_1900s', 'YearBuilt_nan', 'YearBuilt_new',
           'YearBuilt_postwar', 'YearBuilt_prewar', 'LatLong'],
          dtype='object')


# Exploratory

Three sets of pandas profiling output (all values, target cleaned, and after
cleaning) were used in assessing the above cleaning steps:


```python
profile = pdpr.ProfileReport(train_full)
profile.to_file(output_file='profile.html')
```

    Summarize dataset: 100%|██████████| 233/233 [00:50<00:00,  4.59it/s, Completed]
    Generate report structure: 100%|██████████| 1/1 [00:11<00:00, 11.14s/it]
    Render HTML: 100%|██████████| 1/1 [00:09<00:00,  9.31s/it]
    Export report to file: 100%|██████████| 1/1 [00:00<00:00, 38.64it/s]



```python
profile = pdpr.ProfileReport(train_with_target)
profile.to_file(output_file='profile_cleaned_target.html')
```

    Summarize dataset: 100%|██████████| 233/233 [00:50<00:00,  4.64it/s, Completed]
    Generate report structure: 100%|██████████| 1/1 [00:11<00:00, 11.13s/it]
    Render HTML: 100%|██████████| 1/1 [00:09<00:00,  9.07s/it]
    Export report to file: 100%|██████████| 1/1 [00:00<00:00, 48.01it/s]



```python
profile = pdpr.ProfileReport(X_train)
profile.to_file(output_file='profile_clean_features.html')
```

    Summarize dataset: 100%|██████████| 111/111 [00:22<00:00,  4.84it/s, Completed]
    Generate report structure: 100%|██████████| 1/1 [00:09<00:00,  9.25s/it]
    Render HTML: 100%|██████████| 1/1 [00:04<00:00,  4.17s/it]
    Export report to file: 100%|██████████| 1/1 [00:00<00:00, 87.89it/s]


# Models

We fit a linear regression and a random forrest regressor to the cleaned data,
applying a standard scalar first in each case.


```python
pipe_rf = Pipeline([
    ('scaler', StandardScaler()),
    ('model', RandomForestRegressor(min_samples_split=10))
])

rf_cv = cross_validate(
    pipe_rf,
    X_train,
    y_train,
    scoring='neg_mean_squared_error',
    return_estimator=True
)
```


```python
print("RF scores:", np.round(np.sqrt(-rf_cv['test_score'])))
print("RF mean score:", np.round(np.mean(np.sqrt(-rf_cv['test_score']))))
```

    RF scores: [324443. 289369. 280751. 303719. 337606.]
    RF mean score: 307178.0


The mean squared error is still quite high. As a next step features
should be reviewed again for possible improvements before hyperparameters of
the random forest are tuned before a final evaluation against the test set.
