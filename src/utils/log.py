import datetime
from pathlib import Path

from src.dataset.category import CATEGORY

from src.utils.project_root import PROJECT_ROOT


def blue(text):
    return '\033[94m' + text + '\033[0m'


def logging_for_cd_test(validation_result):
    def log_line(loss, batch_index):
        return f"{loss / batch_index * 10_000:.6f}"

    validation_log = log_line(*validation_result)

    print(blue(validation_log))


def logging_for_cd_train(file, epoch, train_result, validation_result):
    def log_line(loss, batch_index):
        return f"{loss / batch_index * 10_000:.6f}"

    train_log = log_line(*train_result)
    validation_log = log_line(*validation_result)

    print(epoch, train_log, blue(validation_log))

    if file:
        log = f"{epoch} {train_log} {validation_log}\n"
        file.write(log)


def logging_for_test(test_result):
    def log_line(loss, batch_index, correct, count):
        return f"{loss / batch_index:.6f} {correct / count:.6f}"

    def category_log_line(category_correct, category_count):
        log = ""
        for i in range(len(category_correct)):
            if category_count[i] == 0:  # for reduced MVP12 zero division error exception
                log += f"None "
            else:
                log += f"{category_correct[i] / category_count[i]:.2f} "
        return log

    def category_log_line_for_monitor(category_correct, category_count):
        log = ""
        for i in range(len(category_correct)):
            if category_count[i] == 0:  # for reduced MVP12 zero division error exception
                log += f"{CATEGORY[i]}-None "
            else:
                log += f"{CATEGORY[i]}-{category_correct[i] / category_count[i]:.2f} "
        return log

    total_test_result = test_result[:4]
    category_test_result = test_result[4:]

    total_test_log = log_line(*total_test_result)

    category_test_log = category_log_line(*category_test_result)
    category_test_log_for_monitor = category_log_line_for_monitor(*category_test_result)

    print(blue(total_test_log), category_test_log_for_monitor)
    print(category_test_log)


def logging_for_train(file, epoch, train_result, validation_result):
    def log_line(loss, batch_index, correct, count):
        return f"{loss / batch_index:.6f} {correct / count:.6f}"

    def category_log_line_for_monitor(category_correct, category_count):
        log = ""
        for i in range(len(category_correct)):
            if category_count[i] == 0:  # for reduced MVP12 zero division error exception
                log += f"{CATEGORY[i]}-None "
            else:
                log += f"{CATEGORY[i]}-{category_correct[i] / category_count[i]:.2f} "
        return log

    def category_log_line(category_correct, category_count):
        log = ""
        for i in range(len(category_correct)):
            if category_count[i] == 0:  # for reduced MVP12 zero division error exception
                log += f"None "
            else:
                log += f"{category_correct[i] / category_count[i]:.2f} "
        return log

    total_validation_result = validation_result[:4]
    category_validation_result = validation_result[4:]

    train_log = log_line(*train_result)
    total_test_log = log_line(*total_validation_result)

    category_test_log = category_log_line(*category_validation_result)
    category_test_log_for_monitor = category_log_line_for_monitor(*category_validation_result)

    print(epoch, train_log, blue(total_test_log), category_test_log_for_monitor)

    if file:
        log = f"{epoch} {train_log} {total_test_log} {category_test_log}\n"
        file.write(log)


def logging(file, epoch, train_result, test_result):
    def log_line(loss, correct, count):
        return f"{loss / count:.6f} {correct / count:.6f}"

    def category_log_line_for_monitor(category_correct, category_count):
        log = ""
        for i in range(len(category_correct)):
            if category_count[i] == 0:  # for reduced MVP12 zero division error exception
                log += f"{CATEGORY[i]}-{0:.2f} "
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

    total_test_result = test_result[:4]
    category_test_result = test_result[4:]

    train_log = log_line(*train_result)
    total_test_log = log_line(*total_test_result)

    category_test_log = category_log_line(*category_test_result)
    category_test_log_for_monitor = category_log_line_for_monitor(*category_test_result)

    print(epoch, train_log, blue(total_test_log), category_test_log_for_monitor)

    if file:
        log = f"{epoch} {train_log} {total_test_log} {category_test_log}\n"
        file.write(log)


def get_log_file(experiment_type: str, dataset_type: str, train_shape: str, validation_shape: str = None,
                 test_shape=None):
    now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    directory = PROJECT_ROOT / 'result' / experiment_type / dataset_type

    if experiment_type == "train":
        file_name = f"{train_shape}_{now}.txt"
        start_log = f"The {experiment_type.capitalize()} Experiment for {train_shape.capitalize()} is started at {now}."
    else:  # experiment_type == "test"
        file_name = f"{train_shape}_{test_shape}_{now}.txt"
        start_log = f"The {experiment_type.capitalize()} Experiment from {train_shape.capitalize()} to {test_shape.capitalize()} is started at {now}."
    print(start_log)

    file = open(directory / file_name, "w")

    if experiment_type == "train":
        index = f"Epoch Train_Loss Train_Accuracy Validation_Loss Validation_Accuracy\n"
    else:  # experiment_type == "test"
        index = f"Test_Loss Test_Accuracy\n"
    file.write(index)
    print(index, end="")

    return file


def get_log_for_auto_encoder(dataset_type: str, loss_type="ce"):
    now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    directory = PROJECT_ROOT / 'result/train/' / dataset_type

    file_name = f"{loss_type}_{now}.txt"
    start_log = f"The {loss_type.capitalize()} Experiment for is started at {now}."
    print(start_log)

    file = open(directory / file_name, "w")

    if loss_type == "ce":
        index = f"Epoch Train_CE Train_Accuracy Validation_CE Validation_Accuracy\n"

        file.write(index)
        print(index, end="")
    elif loss_type == 'cd':
        index = f"Epoch Train_CD Validation_CD \n"
        print(index, end="")
    return file
