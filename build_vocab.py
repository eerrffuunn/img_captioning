import nltk
import pickle
import argparse
from collections import Counter
from pycocotools.coco import COCO


class Lexicon:
    """Class to manage word-index mapping."""
    def __init__(self):
        self.word_to_index = {}
        self.index_to_word = {}
        self.current_index = 0

    def insert(self, word):
        if word not in self.word_to_index:
            self.word_to_index[word] = self.current_index
            self.index_to_word[self.current_index] = word
            self.current_index += 1

    def get(self, word):
        return self.word_to_index.get(word, self.word_to_index['<unk>'])

    def size(self):
        return len(self.word_to_index)


def create_lexicon(data_path, min_freq):
    """Generate a lexicon from image captions."""
    coco_data = COCO(data_path)
    word_counts = Counter()
    
    for idx, annotation_id in enumerate(coco_data.anns.keys()):
        caption = str(coco_data.anns[annotation_id]['caption'])
        tokens = nltk.tokenize.word_tokenize(caption.lower())
        word_counts.update(tokens)

        if (idx + 1) % 1000 == 0:
            print(f"Processed {idx + 1} captions.")

    # Filter words by frequency
    valid_words = [word for word, count in word_counts.items() if count >= min_freq]

    # Initialize lexicon and add special tokens
    lexicon = Lexicon()
    for token in ['<pad>', '<start>', '<end>', '<unk>']:
        lexicon.insert(token)

    for word in valid_words:
        lexicon.insert(word)
    
    return lexicon


def run(arguments):
    lexicon = create_lexicon(data_path=arguments.data_path, min_freq=arguments.min_freq)
    with open(arguments.save_path, 'wb') as file:
        pickle.dump(lexicon, file)
    
    print(f"Lexicon size: {lexicon.size()}")
    print(f"Lexicon saved at: {arguments.save_path}")


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(description="Lexicon Builder Arguments")
    arg_parser.add_argument('--data_path', type=str, 
                            default='data/annotations/captions_train2014.json', 
                            help='Path to the COCO captions dataset')
    arg_parser.add_argument('--save_path', type=str, default='./data/lexicon.pkl', 
                            help='Path to save the lexicon')
    arg_parser.add_argument('--min_freq', type=int, default=4, 
                            help='Minimum word frequency to include in lexicon')
    
    parsed_args = arg_parser.parse_args()
    run(parsed_args)
