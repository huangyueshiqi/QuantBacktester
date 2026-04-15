import pickle
import pandas as pd


def load_data(is_ts):
    """
    Loads the dataset from the pickle file.
    """
    if is_ts:
        with open("/home/quant/qlib/dataset_ts.pkl", "rb") as file_dataset:
            dataset = pickle.load(file_dataset)
        return dataset
    else:
        with open("dataset.pkl", "rb") as file_dataset:
            dataset = pickle.load(file_dataset)
        return dataset


def load_model(model_name):
    """
    Loads the model from the pickle file.
    """
    model_name = "/home/quant/qlib/saved_models/" + model_name + ".pkl"
    with open(model_name, "rb") as file_model:
        model = pickle.load(file_model)
    return model


def load_data_all(is_ts):
    """
    Loads the dataset from the pickle file.
    """
    if is_ts:
        with open("model_input/" + "dataset_ts_all.pkl", "rb") as file_dataset:
            dataset = pickle.load(file_dataset)
        return dataset
    else:
        with open("dataset_all.pkl", "rb") as file_dataset:
            dataset = pickle.load(file_dataset)
        return dataset


def get_pred_scores(modelname, is_ts):
    """
    Gets the predictions and scores from the model.
    """
    model = load_model(modelname)
    if "_all" in modelname:
        dataset = load_data_all(is_ts)
    else:
        dataset = load_data(is_ts)
    pred_score = model.predict(dataset)
    pred_score = pred_score.rename('score')
    pred_score = pred_score.to_frame('score')
    pred_score = pred_score.reset_index()
    return pred_score


if __name__ == "__main__":
    # 主程序入口
    pred_score = get_pred_scores("gru_company", True)

    # 可选：保存结果到文件
    pred_score.to_csv("input/prediction_results.csv", index=False)
    print("预测结果已保存到 prediction_results.csv")

    # 显示前几行结果
    print("\n预测结果前10行：")
    print(pred_score.head(10))