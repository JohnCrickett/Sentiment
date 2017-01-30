import argparse
import sys
import time

from modelling import train_and_save_model


def get_arguments():
    parser = argparse.ArgumentParser(description=
                                     'Portishedge Sentiment Modeller.')
    parser.add_argument('-t', '--train', action='store_true',
                        default=False, dest='train')
    parser.add_argument('-d', '--data', action='store', dest='data')
    parser.add_argument('-p', '--predict', action='store_true',
                        default=False, dest='predict')
    parser.add_argument('-m', '--model', action='store', dest='model')
    parser.add_argument('--article', action='store', dest='article')

    if len(sys.argv) == 1:
        parser.print_help()
        exit(-1)

    return parser.parse_args()


def main():
    args = get_arguments()
    start_time = time.time()

    if args.train:
        if args.data is not None:
            if args.model is not None:
                train_and_save_model(args.data, args.model)
                print("%f seconds" % (time.time() - start_time))
                exit(0)
            else:
                print('Please provide a model filename')
        else:
            print('Please supply a data file to model from')
            exit(-1)
    elif args.predict:
        if args.model is not None:
            if args.article is not None:
                model = load_model(args.model)
                prediction = predict(model, article)
                print('Prediction: {p}'.format(p=prediction))
                exit(0)
            else:
                print('Please provide an article to predict')
        else:
            print('Please provide a model to use for prediction')
    else:
        print('Error: please set either model or predict flags')
    exit(-1)


if __name__ == '__main__':
    main()
