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
$ python main.py [-o OPTION] [-t TYPE] [-d DATA] [-m MODEL]
```
|                            | Description                                    |
| ---                        | ---                                            |
| **General Options**        |                                                |
| -h, --help                 | show this help message and exit                |
| **Operational Options**    |                                                |
| -o gen                     | data generation                                |
| -o train                   | training model                                 |
| -o report\_training\_data  | show all training data                         |
| -o report\_validation\_data| show all validation data                       |
| -o report\_testing\_data   | show all testing data                          |
| -o report\_accuracy        | show accuracy                                  |
| -o test                    | input formula by self                          |
| **Calculational Options**  | **Default `-t sub`**                           |
| -t sub                     | subtraction                                    |
| -t sub\_add                | subtraction mix with addition                  |
| -t multiply                | multiplication                                 |
| **Advance Options**        |                                                |
| -d DATA                    | input the path of training (or generation) data|
|                            | (default: `src/data.pkl`)                      |
| -m MODEL                   | input the path of model                        |
|                            | (default: `src/my_model.h5`)                   |


## Architecture
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
| lambda\_1 (Lambda)              | (None, 4, 12)       | 0          | reshape\_2[0][0]                 |
| ------------------------------- | ------------------- | ---------- | -------------------------------- |
| Total params: 2,674,736         |                     |            |                                  | 
| Trainable params: 2,674,736     |                     |            |                                  | 
| Non-trainable params: 0         |                     |            |                                  | 

### Data Size
- Training Data: 18,000
- Validation Data: 2,000
- Testing Data: 60,000

### Result
- Subtraction
```sh
$ python main.py -o report_training_data -t sub -d src/sub_data.pkl
```

```sh
150-87  = 63  
554-4   = 550 
195-0   = 195 
306-52  = 254 
61-14   = 47  
519-84  = 435 
78-3    = 75  
8-7     = 1   
54-9    = 45  
9-5     = 4   
...
349-4   = 345 
996-688 = 308 
200-64  = 136 
586-55  = 531 
509-99  = 410 
510-141 = 369 
708-9   = 699 
585-35  = 550 
794-3   = 791 
58-39   = 19  
```

```sh
$ python main.py -o report_validation_data -t sub -d src/sub_data.pkl
```

```sh
935-772 = 163
860-23  = 837
488-331 = 157
263-11  = 252
670-90  = 580
810-0   = 810
299-65  = 234
964-548 = 416
687-44  = 643
208-9   = 199
...
991-661 = 330
860-206 = 654
71-32   = 39
654-9   = 645
89-84   = 5
286-6   = 280
519-113 = 406
854-27  = 827
573-75  = 498
917-412 = 505
```

```sh
$ python main.py -o train -t sub -d src/sub_data.pkl -m src/sub_model.h5
```
| Iteration | Training - Loss | Training - Accuracy | Validation - Loss | Validation - Accuracy |
| ---:      | ---:            | ---:                | ---:              | ---:                  |
| 1         | 1.5936          | 0.4265              | 1.3623            | 0.4994                |
| 6         | 0.2564          | 0.9145              | 0.2541            | 0.9163                |
| 10        | 0.0570          | 0.9853              | 0.0486            | 0.9905                |
| 23        | 0.0023          | 1.0000              | 0.0057            | 0.9990                |
| 31        | 5.8233e-04      | 1.0000              | 0.0026            | 0.9995                |
| 93        | 1.2759e-06      | 1.0000              | 4.7339e-04        | 0.9999                |

- Subtraction & Addition
```sh
$ python main.py -o report_training_data -t sub_add -d src/sub_add_data.pkl
```

```sh
26+0    = 26
8-6     = 2
529-29  = 500
25+18   = 43
5-2     = 3
386-3   = 383
925+900 = 1825
304+524 = 828
608-2   = 606
7+910   = 917
...
775-5   = 770
69+9    = 78
2+288   = 290
495-30  = 465
194-18  = 176
82+174  = 256
501-35  = 466
3+751   = 754
31+73   = 104
87-78   = 9
```

```sh
$ python main.py -o report_validation_data -t sub_add -d src/sub_add_data.pkl
```

```sh
83-17   = 66
64-21   = 43
542-11  = 531
687-6   = 681
196-4   = 192
973-3   = 970
2+770   = 772
75-25   = 50
731-70  = 661
647+40  = 687
...
95-27   = 68
296-80  = 216
397-25  = 372
368+183 = 551
3+320   = 323
547+8   = 555
879+620 = 1499
2+41    = 43
70+22   = 92
0+685   = 685
```

```sh
$ python main.py -o train -t sub_add -d src/sub_add_data.pkl -m src/sub_add_model.h5
```
| Iteration | Training - Loss | Training - Accuracy | Validation - Loss | Validation - Accuracy |
| ---:      | ---:            | ---:                | ---:              | ---:                  |
| 1         | 1.6767          | 0.4052              | 1.5652            | 0.4274                |
| 14        | 0.1651          | 0.9491              | 0.2732            | 0.9002                |
| 19        | 0.0455          | 0.9912              | 0.1445            | 0.9530                |
| 28        | 0.0070          | 0.9999              | 0.0967            | 0.9660                |
| 94        | 1.0389e-04      | 1.0000              | 0.1103            | 0.9704                |

- Multiplication
```sh
$ python main.py -o report_training_data -t multiply -d src/multiply_data.pkl 
```

```sh
3*9     = 27
884*705 = 623220
638*92  = 58696
2*23    = 46
0*56    = 0
804*62  = 49848
288*938 = 270144
76*2    = 152
2*758   = 1516
23*304  = 6992
...
5*607   = 3035
802*2   = 1604
61*86   = 5246
643*84  = 54012
959*230 = 220570
80*502  = 40160
920*112 = 103040
41*362  = 14842
499*22  = 10978
734*2   = 1468
```

```sh
$ python main.py -o report_validation_data -t multiply -d src/multiply_data.pkl 
```

```sh
30*325  = 9750
8*330   = 2640
9*305   = 2745
3*966   = 2898
80*80   = 6400
8*629   = 5032
774*643 = 497682
603*6   = 3618
49*489  = 23961
674*55  = 37070
...
410*221 = 90610
1*746   = 746
236*0   = 0
877*516 = 452532
594*32  = 19008
5*227   = 1135
982*830 = 815060
448*577 = 258496
696*13  = 9048
48*978  = 46944
```

```sh
$ python main.py -o train -t multiply -d src/multiply_data.pkl -m src/multiply_model.h5
```
| Iteration | Training - Loss | Training - Accuracy | Validation - Loss | Validation - Accuracy |
| ---:      | ---:            | ---:                | ---:              | ---:                  |
| 1         | 1.6845          | 0.3909              | 1.6515            | 0.3782                |
| 10        | 0.8715          | 0.6581              | 1.0188            | 0.5925                |
| 20        | 0.3881          | 0.8680              | 0.7656            | 0.7196                |
| 30        | 0.0734          | 0.9904              | 0.9744            | 0.7302                |
| 78        | 1.5602e-04      | 1.0000              | 1.5366            | 0.7372                |

## Related Link
- [nbviewer](https://nbviewer.jupyter.org/github/yutongshen/DSAI-HW3-Subtractor/blob/master/Subtractor.ipynb)

## Authors
[Yu-Tong Shen](https://github.com/yutongshen/)
