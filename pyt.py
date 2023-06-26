
import argparse

# Create the parser and add arguments
parser = argparse.ArgumentParser()

parser.add_argument('-epochs', '--e', type=int, help='The total number of training epochs')
parser.add_argument('-dataset_type', '--type', type=str, help='Set the variable to LIVE or BVI depending on your choice or training dataset')
parser.add_argument('-dataset_csv_path', '--csv', type=str, help='Set the path to dataset CSV')
parser.add_argument('-video_path', '--vp', type=str, help='Set the path to the videos directory')
parser.add_argument('-spatial_pooling', '--sp', type=str, help='Set aptial pooling to lstm, rnn, or gru ')
parser.add_argument('-embedding_size', '--size', type=int, help='Set the embedding sitz (the size of the transformer token) ')
parser.add_argument('-weights_folder', '--wf', type=str, help='Set the path to the directory containing all training weights')
parser.add_argument('-best_score_weights', '--bsw', type=str, help='Set the path to the .pth containtnig the best weights')

args = parser.parse_args()



