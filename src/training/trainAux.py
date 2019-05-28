import time

from matplotlib import pyplot as plt

_losses = []


def lossCallback(training_loss):
    print(f'Training loss: {round(training_loss, 2)}')
    _losses.append(training_loss)


def epochCallback(epochs_num, epoch):
    print(f'Epoch {epoch} / {epochs_num}')


def progressCallback(so_far, remaining_time):
    remaining_time = time.gmtime(remaining_time)
    print(
        f'Processed {so_far} images; Remaining time: {remaining_time.tm_mday - 1} days, '
        f'{remaining_time.tm_hour} hours, {remaining_time.tm_min} minutes')


def evalEveryCallback():
    print(f'Starting model evaluation')


def endCallback(figpath, elapsed_time_path, epochs, evalEvery, elapsed_time):
    elapsed_time = time.gmtime(elapsed_time)
    printLogTrainingTime(elapsed_time, elapsed_time_path)

    plt.plot(range(1, epochs + 1, evalEvery), _losses, '--o')
    plt.legend(['Training loss'])
    plt.title('Learning Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig(figpath, dpi=600)


def printLogTrainingTime(elapsed_time, filepath):
    elapsed_time_msg = (f'Training finished in {elapsed_time.tm_mday - 1} days, '
                        f'{elapsed_time.tm_hour} hours, {elapsed_time.tm_min} minutes')
    print(elapsed_time_msg)
    with open(filepath, 'w') as f:
        f.write(elapsed_time_msg)
