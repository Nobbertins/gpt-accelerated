#include "tokenizer.h"

Tokenizer::Tokenizer(std::vector<std::string> &lines){
            std::string alph = "abcdefghijklmnopqrstuvwxyz";
            int i = 0;
            for(auto c : alph){
                char_to_token.emplace(c, i);
                token_to_char.emplace(i, c);
                i++;
            }
            vocab_size = 26;
}
std::string Tokenizer::decode(const std::vector<int> &v) const {
    std::string s;
    for(const auto num : v){
        s += token_to_char.at(num);
    }
    return s;
}
std::vector<int> Tokenizer::encode(const std::string &s) const {
    std::vector<int> v;
    //start and end with BOS token
    v.push_back(vocab_size);
    for(const auto ch : s){
        v.push_back(char_to_token.at(ch));
    }
    v.push_back(vocab_size);
    return v;
}