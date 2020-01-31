import torch

class RunningAverage():
    """A simple class that maintains the running average of a quantity

    Example:
    ```
    loss_avg = RunningAverage()
    loss_avg.update(2)
    loss_avg.update(4)
    loss_avg() = 3
    ```
    """

    def __init__(self):
        self.steps = 0
        self.total = 0

    def func(self, *values):
        raise NotImplemented

    def update(self, *values):
        val = self.func(*values)
        self.total += val
        self.steps += 1

    def __call__(self):
        return self.total / float(self.steps)

class AverageMeter(RunningAverage):
    def __init__(self):
        super().__init__()

    def func(self, loss_value):
        return loss_value

class Accuracy():
    def __init__(self, ignore_index=None):
        self.correct = 0.0
        self.total = 0.0
        self.ignore_index = ignore_index

    def update(self, outputs, labels):
        outputs = outputs.detach().cpu()
        labels = labels.detach().cpu()

        _, predicted = torch.max(outputs.data, 1)

        if self.ignore_index != None:
            labels = labels[(labels != self.ignore_index).nonzero()]
            predicted = predicted[(labels != self.ignore_index).nonzero()]

        self.total += labels.size(0)
        self.correct += (predicted == labels).sum().item()

    def __call__(self):
        return self.correct / self.total
