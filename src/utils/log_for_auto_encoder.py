import datetime

from src.utils.project_root import PROJECT_ROOT


def blue(text):
    return '\033[94m' + text + '\033[0m'


def logging(file, epoch, train_result, test_result):
    def log_line(loss, count):
        return f"{loss / count * 10000:.6f}"

    train_log = log_line(*train_result)
    test_log = log_line(*test_result)

    print(epoch, train_log, blue(test_log))
    if file:
        log = f"{epoch} {train_log} {test_log}\n"
        file.write(log)


def get_log_file(num_point):
    directory = PROJECT_ROOT / 'result' / f"auto_encoder"
    now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    file_name = f"{num_point}_{now}.txt"
    file = open(directory / file_name, "w")
    print(f"The AutoEncoder Experiment is started at {now}.")
    index = "Epoch Train_Loss Test_Loss\n"
    file.write(index)
    print(index, end="")
    return file
