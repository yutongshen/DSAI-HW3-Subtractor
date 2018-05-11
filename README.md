# Data Science and Artificial Intelligence Practice Homework
DSAI HW3-Substractor

## Prerequisite
- Python 3.6.4

## Install Dependency
```sh
$ pip install -r requirements.txt
```

## Usage
```sh
$ python main.py [-o OPTION] [-d DATA] [-m MODEL]
```
|                            | Description                                    |
| ---                        | ---                                            |
| **General Options**        |                                                |
| -h, --help                 | show this help message and exit                |
| **Advance Options**        |                                                |
| -o gen                     | data generation                                |
| -o train                   | training model                                 |
| -o report\_training\_data  | show all training data                         |
| -o report\_validation\_data| show all validation data                       |
| -o report\_testing\_data   | show all testing data                          |
| -o report\_accuracy        | show accuracy                                  |
| -o test                    | input formula by self                          |
| -d DATA                    | input the path of training (or generation) data|
|                            | (default: `src/data.pkl`)                      |
| -m MODEL                   | input the path of model                        |
|                            | (default: `src/my\_model.h5`)                  |


## Architecture
### Data
- Training Data: 18,000
- Validation Data: 2,000
- Testing Data: 60,000

### Model
![model](img/seq2seq.png)

- Using sequence to sequence model
- Encoder: bi-directional LSTM (Hidden Size = 256)
- Decoder: LSTM (Hidden Size = 512)

| Layer (type)                    | Output Shape        | Param #    | Connected to                     |
| ------------------------------- | ------------------- | ---------: | -------------------------------- |
| input\_1 (InputLayer)           | (None, 7, 12)       | 0          |                                  |
| ------------------------------- | ------------------- | ---------- | -------------------------------- |
| bidirectional\_1 (Bidirectional)| \[(None, 512), ...  | 550912     | input\_1[0][0]                   |
| ------------------------------- | ------------------- | ---------- | -------------------------------- |
| reshape\_1 (Reshape)            | (None, 1, 512)      | 0          | bidirectional\_1[0][0]           |
| ------------------------------- | ------------------- | ---------- | -------------------------------- |
| concatenate\_1 (Concatenate)    | (None, 512)         | 0          | bidirectional\_1[0][1]           |
|                                 |                     |            | bidirectional\_1[0][3]           |
| ------------------------------- | ------------------- | ---------- | -------------------------------- |
| concatenate\_2 (Concatenate)    | (None, 512)         | 0          | bidirectional\_1[0][2]           |
|                                 |                     |            | bidirectional\_1[0][4]           |
| ------------------------------- | ------------------- | ---------- | -------------------------------- |
| lstm\_2 (LSTM)                  | \[(None, 512), ...  | 2099200    | reshape\_1[0][0]                 |
|                                 |                     |            | concatenate\_1[0][0]             |
|                                 |                     |            | concatenate\_2[0][0]             |
|                                 |                     |            | reshape\_1[0][0]                 |
|                                 |                     |            | lstm\_2[0][1]                    |
|                                 |                     |            | lstm\_2[0][2]                    |
|                                 |                     |            | reshape\_1[0][0]                 |
|                                 |                     |            | lstm\_2[1][1]                    |
|                                 |                     |            | lstm\_2[1][2]                    |
|                                 |                     |            | reshape\_1[0][0]                 |
|                                 |                     |            | lstm\_2[2][1]                    |
|                                 |                     |            | lstm\_2[2][2]                    |
| ------------------------------- | ------------------- | ---------- | -------------------------------- |
| dense\_1 (Dense)                | (None, 12)          | 6156       | lstm\_2[0][0]                    |
| ------------------------------- | ------------------- | ---------- | -------------------------------- |
| dense\_2 (Dense)                | (None, 12)          | 6156       | lstm\_2[1][0]                    |
| ------------------------------- | ------------------- | ---------- | -------------------------------- |
| dense\_3 (Dense)                | (None, 12)          | 6156       | lstm\_2[2][0]                    |
| ------------------------------- | ------------------- | ---------- | -------------------------------- |
| dense\_4 (Dense)                | (None, 12)          | 6156       | lstm\_2[3][0]                    |
| ------------------------------- | ------------------- | ---------- | -------------------------------- |
| concatenate\_3 (Concatenate)    | (None, 48)          | 0          | dense\_1[0][0]                   |
|                                 |                     |            | dense\_2[0][0]                   | 
|                                 |                     |            | dense\_3[0][0]                   |
|                                 |                     |            | dense\_4[0][0]                   |
| ------------------------------- | ------------------- | ---------- | -------------------------------- |
| reshape\_2 (Reshape)            | (None, 4, 12)       | 0          | concatenate\_3[0][0]             |
| ------------------------------- | ------------------- | ---------- | -------------------------------- |
| Total params: 2,674,736         |                     |            |                                  | 
| Trainable params: 2,674,736     |                     |            |                                  | 
| Non-trainable params: 0         |                     |            |                                  | 

### Result
| Iteration | Training - Loss | Training - Accuracy | Validation - Loss | Validation - Accuracy |
| ---:      | ---:            | ---:                | ---:              | ---:                  |
| 1         | 1.5882          | 0.4285              | 1.4052            | 0.4711                |
| 7         | 0.2379          | 0.9257              | 0.2542            | 0.9132                |
| 9         | 0.1509          | 0.9541              | 0.1500            | 0.9561                |
| 15        | 0.0394          | 0.9921              | 0.0726            | 0.9773                |
| 31        | 0.0056          | 0.9998              | 0.0305            | 0.9903                |
| 100       | 2.9910e-04      | 1.0000              | 0.0251            | 0.9920                |

## Related Link
- [nbviewer](https://nbviewer.jupyter.org/github/yutongshen/DSAI-HW3-Subtractor/blob/master/Subtractor.ipynb)

## Authors
[Yu-Tong Shen](https://github.com/yutongshen/)
