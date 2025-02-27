from src.datasets import UTSig

dataset = UTSig(2, 3, 80, True)
train_dataset, test_dataset = dataset.get_train(), dataset.get_test()

print(len(train_dataset))
print(len(test_dataset))