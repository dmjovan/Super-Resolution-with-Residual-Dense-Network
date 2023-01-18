import argparse

from super_resolution_handler import SuperResolutionHandler

parser = argparse.ArgumentParser(description="Parsing program arguments and algorithm hyper-parameters...")
parser.add_argument("--mode", type=str)
parser.add_argument("--random_crop_train_dataset", type=str)
parser.add_argument("--validation", type=str)
parser.add_argument("--model_path", type=str)
parser.add_argument("--hyper_params_path", type=str)
parser.add_argument("--test_image_path", type=str)

if __name__ == "__main__":
    args = parser.parse_args()

    super_resolution_handler = SuperResolutionHandler(mode=str(args.mode).lower(),
                                                      random_crop_train_dataset=str(
                                                          args.random_crop_train_dataset).lower(),
                                                      validation=str(args.validation).lower(),
                                                      model_path=args.model_path,
                                                      hyper_params_path=args.hyper_params_path,
                                                      test_image_path=args.test_image_path)

    super_resolution_handler.run()
