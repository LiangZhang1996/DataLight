## Download our offline dataset
Download the offline dataset from [offline dataset](https://drive.google.com/drive/folders/1RR6KT4jZvc0MDA_2s1b_Yj7OirgU5hAb?usp=sharing) and put them at this folder.

These datasets are described as follows:
- `random10.pkl`: generated with random policy with action duration as 10s;
- `random15.pkl`: generated with random policy with action duration as 15s;
- `random20.pkl`: generated with random policy with action duration as 20s;
- `medium10.pkl`: generated with `FRAP` with action duration as 10s;
- `medium15.pkl`: generated with `FRAP` with action duration as 15s;
- `medium20.pkl`: generated with `FRAP` with action duration as 20s;
- `expert10.pkl`: generated with `Advanced-CoLight` with action duration as 10s;
- `expert15.pkl`: generated with `Advanced-CoLight` with action duration as 15s;
- `expert20.pkl`: generated with `Advanced-CoLight`with action duration as 20s;
- `cycle10.pkl`: generated with `FixedTime` with action duration as 10s;
- `cycle15.pkl`: generated with `FixedTime` with action duration as 15s;
- `cycle20.pkl`: generated with `FixedTime` with action duration as 20s;

Specifically, when train the `DT`, a complete trajectory is utlized, they are: `random10-2.pkl`, `medium10-2.pkl`, and `expert10-2.pkl`.

## Prepare your own offline dataset

You can use [AttentionLight](https://github.com/LiangZhang1996/AttentionLight.git) to prepare your own offline dataset.
