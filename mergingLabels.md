```python
import pandas as pd
import numpy as np


trainLabel= pd.read_csv("/home/abir/Study/Summer21/Opencampus/jesterV1/jester-v1-train.csv",delimiter=";")
validationLabel= pd.read_csv("/home/abir/Study/Summer21/Opencampus/jesterV1/jester-v1-validation.csv",delimiter=";")
#df = pd.DataFrame(pd.read_csv("/home/abir/Study/Summer21/Opencampus/jesterV1/jester-v1-train.csv",delimiter=""))

print("train Label shape : ",trainLabel.shape)
print(trainLabel.head())
print("\nvalidation Label shape : ",validationLabel.shape)
print(validationLabel.head())

#merged = trainLabel.merge(validationLabel,on='ID')
#print("\nmerged shape : ",merged.shape)
#merged.head()

merged = pd.concat([trainLabel,validationLabel])
print("\nconcat list shape: ",merged.shape)
merged.head()

sorted_df = merged.sort_values(by=["ID"], ascending=True)
sorted_df

```

    train Label shape :  (118562, 2)
           ID                      Label
    0   34870           Drumming Fingers
    1   56557  Sliding Two Fingers Right
    2  129112   Sliding Two Fingers Down
    3   63861     Pulling Two Fingers In
    4  131717     Sliding Two Fingers Up
    
    validation Label shape :  (14787, 2)
           ID                     Label
    0    9223                  Thumb Up
    1  107090  Pushing Two Fingers Away
    2   42920              Swiping Left
    3  106485                Thumb Down
    4  142201     Rolling Hand Backward
    
    concat list shape:  (133349, 2)





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ID</th>
      <th>Label</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>97184</th>
      <td>1</td>
      <td>Doing other things</td>
    </tr>
    <tr>
      <th>111045</th>
      <td>3</td>
      <td>Pushing Two Fingers Away</td>
    </tr>
    <tr>
      <th>59351</th>
      <td>4</td>
      <td>Swiping Right</td>
    </tr>
    <tr>
      <th>36296</th>
      <td>5</td>
      <td>Rolling Hand Backward</td>
    </tr>
    <tr>
      <th>82397</th>
      <td>6</td>
      <td>Drumming Fingers</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>93345</th>
      <td>148088</td>
      <td>Sliding Two Fingers Up</td>
    </tr>
    <tr>
      <th>67048</th>
      <td>148089</td>
      <td>Zooming In With Full Hand</td>
    </tr>
    <tr>
      <th>50277</th>
      <td>148090</td>
      <td>Swiping Left</td>
    </tr>
    <tr>
      <th>14299</th>
      <td>148091</td>
      <td>Swiping Up</td>
    </tr>
    <tr>
      <th>34007</th>
      <td>148092</td>
      <td>Pulling Two Fingers In</td>
    </tr>
  </tbody>
</table>
<p>133349 rows × 2 columns</p>
</div>




```python
# Getting first 3 rows from df
first100 = sorted_df.head(100)

# save the dataframe as a csv file
first100.to_csv("labels_smallDataset.csv")

first100
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ID</th>
      <th>Label</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>97184</th>
      <td>1</td>
      <td>Doing other things</td>
    </tr>
    <tr>
      <th>111045</th>
      <td>3</td>
      <td>Pushing Two Fingers Away</td>
    </tr>
    <tr>
      <th>59351</th>
      <td>4</td>
      <td>Swiping Right</td>
    </tr>
    <tr>
      <th>36296</th>
      <td>5</td>
      <td>Rolling Hand Backward</td>
    </tr>
    <tr>
      <th>82397</th>
      <td>6</td>
      <td>Drumming Fingers</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>19623</th>
      <td>110</td>
      <td>Zooming Out With Full Hand</td>
    </tr>
    <tr>
      <th>106794</th>
      <td>111</td>
      <td>Zooming In With Two Fingers</td>
    </tr>
    <tr>
      <th>48525</th>
      <td>112</td>
      <td>Doing other things</td>
    </tr>
    <tr>
      <th>25515</th>
      <td>113</td>
      <td>Pulling Hand In</td>
    </tr>
    <tr>
      <th>107647</th>
      <td>114</td>
      <td>No gesture</td>
    </tr>
  </tbody>
</table>
<p>100 rows × 2 columns</p>
</div>




```python
numpylist = sorted_df.to_numpy()
print(numpylist)

top100 = numpylist[0:100]
print("\n First 100 elements :\n ",top100 )
top100.shape
```

    [[1 'Doing other things']
     [3 'Pushing Two Fingers Away']
     [4 'Swiping Right']
     ...
     [148090 'Swiping Left']
     [148091 'Swiping Up']
     [148092 'Pulling Two Fingers In']]
    
     First 100 elements :
      [[1 'Doing other things']
     [3 'Pushing Two Fingers Away']
     [4 'Swiping Right']
     [5 'Rolling Hand Backward']
     [6 'Drumming Fingers']
     [7 'Sliding Two Fingers Left']
     [8 'Turning Hand Counterclockwise']
     [9 'Swiping Right']
     [10 'Thumb Down']
     [11 'Sliding Two Fingers Down']
     [12 'Sliding Two Fingers Down']
     [13 'Sliding Two Fingers Down']
     [14 'Pushing Hand Away']
     [16 'Thumb Down']
     [17 'Shaking Hand']
     [19 'Pushing Two Fingers Away']
     [20 'Doing other things']
     [22 'Zooming In With Full Hand']
     [23 'Sliding Two Fingers Right']
     [24 'Turning Hand Clockwise']
     [25 'Thumb Up']
     [26 'Sliding Two Fingers Down']
     [27 'Rolling Hand Forward']
     [28 'Pulling Two Fingers In']
     [29 'Sliding Two Fingers Right']
     [30 'Swiping Down']
     [31 'Stop Sign']
     [33 'Doing other things']
     [34 'Zooming In With Two Fingers']
     [35 'Drumming Fingers']
     [36 'Sliding Two Fingers Up']
     [37 'Zooming Out With Two Fingers']
     [38 'Swiping Right']
     [39 'Sliding Two Fingers Left']
     [40 'Pushing Hand Away']
     [41 'Shaking Hand']
     [42 'Sliding Two Fingers Up']
     [43 'Stop Sign']
     [44 'Zooming In With Full Hand']
     [45 'Sliding Two Fingers Down']
     [46 'Zooming In With Full Hand']
     [47 'Swiping Right']
     [48 'Thumb Down']
     [50 'No gesture']
     [51 'Swiping Right']
     [52 'Zooming Out With Full Hand']
     [54 'Sliding Two Fingers Right']
     [55 'Thumb Down']
     [56 'Rolling Hand Forward']
     [57 'Doing other things']
     [58 'Zooming Out With Two Fingers']
     [59 'Stop Sign']
     [60 'Swiping Up']
     [61 'Rolling Hand Backward']
     [62 'Sliding Two Fingers Up']
     [63 'Swiping Left']
     [65 'Swiping Right']
     [66 'Zooming In With Two Fingers']
     [67 'Swiping Down']
     [68 'Shaking Hand']
     [69 'Sliding Two Fingers Up']
     [70 'No gesture']
     [72 'Pulling Two Fingers In']
     [73 'Pushing Hand Away']
     [74 'Zooming In With Two Fingers']
     [75 'Doing other things']
     [76 'Doing other things']
     [77 'Thumb Down']
     [79 'Pushing Hand Away']
     [80 'Doing other things']
     [81 'Rolling Hand Backward']
     [83 'Zooming Out With Two Fingers']
     [85 'Sliding Two Fingers Down']
     [86 'Thumb Up']
     [87 'Swiping Right']
     [88 'Zooming In With Full Hand']
     [89 'Sliding Two Fingers Up']
     [90 'Pushing Hand Away']
     [91 'Sliding Two Fingers Left']
     [92 'Pulling Two Fingers In']
     [93 'Sliding Two Fingers Down']
     [94 'Pulling Hand In']
     [95 'Swiping Right']
     [96 'Swiping Right']
     [99 'Pulling Hand In']
     [100 'Swiping Right']
     [101 'Thumb Up']
     [102 'Rolling Hand Backward']
     [103 'Zooming Out With Full Hand']
     [104 'Pulling Hand In']
     [105 'Stop Sign']
     [106 'Pulling Two Fingers In']
     [107 'Swiping Left']
     [108 'Sliding Two Fingers Up']
     [109 'Pushing Two Fingers Away']
     [110 'Zooming Out With Full Hand']
     [111 'Zooming In With Two Fingers']
     [112 'Doing other things']
     [113 'Pulling Hand In']
     [114 'No gesture']]





    (100, 2)


