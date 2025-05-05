# handwriting_recognition (group project)
CRNN based model to predict handwritten words

how to use:

after cloning the github repository
```
python train.py --epochs 50
```
supports following arguments
```
--data_path  default='./data'
--img_height  default=64
--img_width  default=320
--fraction  default=1.0

    # Model args
--rnn_hidden  default=256
--rnn_layers  default=2
--dropout  default=0.3

    # Training args
--epochs  default=20
--batch_size  default=16
--lr  default=0.001  #Learning rate
--clip_grad  default=1.0 #Gradient clipping 
--num_workers  default=8 # Number of dataloader workers
--save_dir  default='./saved_models'
--force_cpu
```

```
python test.py --data_path path_to_best_model.pth
```
