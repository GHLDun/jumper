This website is anonymized during double-blind review.

# Intro
Implementation of Jumper for OI-Level dataset.

# Requirements
Python
NumPy
Pytorch 0.2.3


# Commands
To start training, run
`python runjumper.py --model 6 --cuda --lr 0.1 --batch_size 50 --epoch 10000 --drate 0.5  --prate 0.1 --mark newdataset --task oi-level --n_sample 20000 --interval 100 --n_hids 20  --n_filter 200`

After training, we can get the decision process  by
`python runjumper.py --model 6 --cuda --lr 0.1 --batch_size 50 --epoch 10000 --drate 0.5  --prate 0.1 --mark newdataset --task oi-level --n_sample 20000 --interval 100 --n_hids 20  --n_filter 200 --save {modelpath}  --eval`
