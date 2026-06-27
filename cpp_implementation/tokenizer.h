#ifndef TOKENIZER_H
#define TOKENIZER_H

#include <string>
#include <unordered_map>
#include <vector>

//Tokenizer to convert characters into numbers and the other way around
class Tokenizer{
    public:
        Tokenizer(std::vector<std::string> &lines);
        std::string decode(const std::vector<int> &v) const;
        std::vector<int> encode(const std::string &s) const;
        int get_vocab_size(void) const {return vocab_size;}
    private:
        std::unordered_map<char, int> char_to_token;
        std::unordered_map<int, char> token_to_char;
        int vocab_size;
};
#endif