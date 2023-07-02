本项目是神经网络与深度学习期末PJ的第一题。
报告结果产出，具体步骤如下：

Step0:下载训练好的模型 https://pan.baidu.com/s/1rotFxl0VsmW48W5sccOY0w 提取码为 1rbv,并解压至当前目录

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

