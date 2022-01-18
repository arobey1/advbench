import time

try:
    import wandb

    wandb_log = True
except ImportError:
    wandb_log = False


class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self, avg_mom=0.5):
        self.avg_mom = avg_mom
        self.reset()
        self.print = True

    def reset(self):
        self.val = 0
        self.avg = 0  # running average of whole epoch
        self.smooth_avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.smooth_avg = (
            val
            if self.count == 0
            else self.avg * self.avg_mom + val * (1 - self.avg_mom)
        )
        self.avg = self.sum / self.count


class TimeMeter:
    def __init__(self):
        self.batch_time = AverageMeter()
        self.data_time = AverageMeter()
        self.start = time.time()

    def batch_start(self):
        self.data_time.update(time.time() - self.start)

    def batch_end(self):
        self.batch_time.update(time.time() - self.start)
        self.start = time.time()


if wandb:

    class WBHistogramMeter:
        def __init__(self):
            self.print = False

        def reset(self):
            pass

        def update(self, val):
            wandb.log({"delta": wandb.Histogram(delta)})

else:

    class WBHistogramMeter:
        def __init__(self):
            self.print = False

        def reset(self):
            pass

        def update(self, delta):
            pass
