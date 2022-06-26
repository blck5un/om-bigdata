'''
Run etl and train under single mlflow run
'''
import argparse
import os
import logging
import mlflow


FORMAT = '%(asctime)s %(levelname)s %(message)s'
logging.basicConfig(level=logging.INFO, format=FORMAT, datefmt='%Y-%m-%d %H:%M:%S')


def parse_args():
    parser = argparse.ArgumentParser(description="HW 5mlb by hethwynq: main.py")
    parser.add_argument(
        "--train_path_in",
        type=str,
        help="path to train dataset",
    )
    parser.add_argument(
        "--sklearn_model",
        type=str,
        help="sklearn model name in MLFlow backend",
    )
    parser.add_argument(
        "--model_param1",
        type=float,
        default=10.0,
        help="inverse of regularization strength (default: 1.0)",
    )    
    return parser.parse_args()


def main(run_id=None):
    # start mlflow run
    mlrun = mlflow.start_run(run_name="MAIN")
    # parse command-line arguments
    args = parse_args()
    logging.info(f"train_path_in: {args.train_path_in}")
    logging.info(f"sklearn_model: {args.sklearn_model}")
    logging.info(f"model_param1: {args.model_param1}")
    # end mlflow run
    mlflow.end_run()
    # define etl_path_out
    etl_path_out = "hdfs:///user/hethwynq/hw5b/train.parquet"
    # run etl
    os.system(f"python etl.py --etl_path_in {args.train_path_in} --etl_path_out {etl_path_out} \
                --run_id {mlrun.info.run_id}")
    # run train
    os.system(f"python train.py --train_path {etl_path_out} --sklearn_model {args.sklearn_model} \
                --model_param1 {args.model_param1} --run_id {mlrun.info.run_id}")
    # etl(args.train_path_in, etl_path_out, run_id=mlrun.info.run_id)
    # train(etl_path_out, args.sklearn_model, args.model_param1, run_id=mlrun.info.run_id)


if __name__ == "__main__":
    main()
