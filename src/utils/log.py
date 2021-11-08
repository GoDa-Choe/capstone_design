import datetime
from pathlib import Path

from src.dataset.category import CATEGORY

PROJECT_ROOT = Path("/home/goda/Undergraduate/capstone_design_base")


def blue(text):
    return '\033[94m' + text + '\033[0m'


def logging(file, epoch, train_result, test_result):
    def log_line(loss, correct, count):
        return f"{loss / count:.6f} {correct / count:.6f}"

    def category_log_line_for_monitor(category_correct, category_count):
        log = ""
        for i in range(len(category_correct)):
            if category_count[i] == 0:  # for reduced MVP12 zero division error exception
                log += f"{CATEGORY[i]}-{0:.2f}  "
            else:
                log += f"{CATEGORY[i]}-{category_correct[i] / category_count[i]:.2f}  "
        return log

    def category_log_line(category_correct, category_count):
        log = ""
        for i in range(len(category_correct)):
            if category_count[i] == 0:  # for reduced MVP12 zero division error exception
                log += f"{0:.2f} "
            else:
                log += f"{category_correct[i] / category_count[i]:.2f} "
        return log

    total_test_result = test_result[:3]
    category_test_result = test_result[3:]

    train_log = log_line(*train_result)
    total_test_log = log_line(*total_test_result)

    category_test_log = category_log_line(*category_test_result)
    category_test_log_for_monitor = category_log_line_for_monitor(*category_test_result)

    print(epoch, train_log, blue(total_test_log), category_test_log_for_monitor)

    if file:
        log = f"{epoch} {train_log} {total_test_log} {category_test_log}\n"
        file.write(log)


def get_log_file(train_shape: str, test_shape: str):
    directory = PROJECT_ROOT / 'result' / f"{train_shape}_{test_shape}"
    now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    file_name = f"{now}.txt"
    file = open(directory / file_name, "w")
    print(f"The Experiment from {train_shape.capitalize()} to {test_shape.capitalize()} is started at {now}.")
    index = "Epoch Train_Loss Train_Accuracy Test_Loss Test_Accuracy\n"
    file.write(index)
    print(index, end="")
    return file
