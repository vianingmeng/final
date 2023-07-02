期末作业第一题模型在https://pan.baidu.com/s/1Olf5frXmCz6SJ9tDeB6A7A，提取码54sb。
具体过程如下：
Step1：训练自监督模型SimSiam并进行Linear Classification Protocol 

```python
python simsiam.py --epoches 81 
python finetune.py --epoches 50
```

Step2：训练没有预训练的Resnet18模型进行比较

```
python resnet18.py --pretrained False --epoches 130
```

Step3：训练预训练的Resnet18模型进行比较

```
python resnet18.py --pretrained True --epoches 50
```

Step4：在已训练好的SimSiam模型基础上继续训练并Linear Classification Protocol 

```
python simsiam_con.py  
python finetune.py --epoches 50
```

