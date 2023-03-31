class Dictionary(object):
    def __init__(self, dataset):
        self.word2idx = dict()
        self.idx2word = list()
        self.add_word(0)

        words = set()
        for stream in dataset:
            words.update([message[1] for message in stream['message_sequence']])
        words = list(words)
        words.sort()
        for word in words:
            self.add_word(word)

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)
