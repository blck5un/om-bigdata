'''
Run etl and predict under single mlflow run
'''
import argparse
import os
import logging
import mlflow

FORMAT = '%(asctime)s %(levelname)s %(message)s'
logging.basicConfig(level=logging.INFO, format=FORMAT, datefmt='%Y-%m-%d %H:%M:%S')


def parse_args():
    parser = argparse.ArgumentParser(description="HW 5mlb by hethwynq: etl_predict.py")
    parser.add_argument(
        "--test_path_in",
        type=str,
        help="path to test dataset",
    )
    parser.add_argument(
        "--pred_path_out",
        type=str,
        help="path to output predictions",
    )
    parser.add_argument(
        "--sklearn_model",
        type=str,
        help="sklearn model name in MLFlow backend",
    )
    parser.add_argument(
        "--model_version",
        type=str,
        help="sklearn model version in MLFlow backend",
    )
    return parser.parse_args()


def etl_predict(run_id=None):
    # start mlflow run
    mlrun = mlflow.start_run(run_name="ETL_PREDICT")
    # parse command-line arguments
    args = parse_args()
    logging.info(f"test_path_in: {args.test_path_in}")
    logging.info(f"pred_path_out: {args.pred_path_out}")
    logging.info(f"sklearn_model: {args.sklearn_model}")
    logging.info(f"model_version: {args.model_version}")
    # end mlflow run
    mlflow.end_run()
    # define etl_path_out
    etl_path_out = "hdfs:///user/hethwynq/hw5b/test.parquet"
    # run etl
    os.system(f"python etl.py --etl_path_in {args.test_path_in} --etl_path_out {etl_path_out} \
                --run_id {mlrun.info.run_id}")
    # run predict
    os.system(f"python predict.py --test_path {etl_path_out} --pred_path_out {args.pred_path_out} \
                --sklearn_model {args.sklearn_model} --model_version {args.model_version} --run_id {mlrun.info.run_id}")


if __name__ == "__main__":
    etl_predict()
