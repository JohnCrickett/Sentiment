import argparse
import sys
import time

from modelling import check_model, load_data, load_model, predict, \
    train_and_save_model


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
    parser.add_argument('-v', '--verbose', action='store_true', dest='verbose')
    parser.add_argument('-c', '--check',
                        action='store_true', dest='check_model')

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
                data = load_data(args.data)
                train_and_save_model(data, args.model)
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
    elif args.check_model:
        if args.data is not None:
            model = load_model(args.model)
            data = load_data(args.data)
            check_model(model, data)
        else:
            print('Please provide the data to check against')
            exit(-1)
    else:
        print('Error: please set either model, predict or check flags')
    exit(-1)


if __name__ == '__main__':
    main()
