#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <unordered_map>
#include <chrono>
#include <memory>
#include <random>
#include "autograd.h"
#include "gpt.h"

//timer to time code
class Timer
{
private:
    // Type aliases to make accessing nested type easier
    using Clock = std::chrono::steady_clock;
    using Second = std::chrono::duration<double, std::ratio<1> >;

    std::chrono::time_point<Clock> m_beg{ Clock::now() };

public:

    void reset()
    {
        m_beg = Clock::now();
    }

    double elapsed() const
    {
        return std::chrono::duration_cast<Second>(Clock::now() - m_beg).count();
    }
};


//Tokenizer to convert characters into numbers and the other way around
class Tokenizer{
    public:
        Tokenizer(std::vector<std::string> &lines){
            int i = 0;
            for(const auto& name : lines){
                for(const char c: name){ 
                    char_to_token.emplace(c, i);
                    token_to_char.emplace(i, c);
                    i = char_to_token.size();//duplicate items ignored
                }
            }
            vocab_size = i;
        }
        std::string decode(const std::vector<int> &v) const {
            std::string s;
            for(const auto num : v){
                s += token_to_char.at(num);
            }
            return s;
        }
        std::vector<int> encode(const std::string &s) const {
            std::vector<int> v;
            for(const auto ch : s){
                v.push_back(char_to_token.at(ch));
            }
            return v;
        }
        int get_vocab_size(void) const {return vocab_size;}
    private:
        std::unordered_map<char, int> char_to_token;
        std::unordered_map<int, char> token_to_char;
        int vocab_size;
};

int main(){
    Timer t;
    //read file input and store in lines
    std::ifstream file("input.txt");
    std::string line;
    std::vector<std::string> lines;
    if (file.is_open()) {
        while (std::getline(file, line)) {
            lines.push_back(line);
        }
        file.close();
    } else {
        std::cerr << "Unable to open file\n";
    }

    const Tokenizer tokenizer(lines);

    std::cout << "Vocab Size: " << tokenizer.get_vocab_size() << std::endl;
    std::cout << "Num Docs: " << lines.size() << std::endl;

    //construct GPT2
    const int n_layer = 1; //depth of the transformer neural network (number of layers)
    const int n_embd = 16;  //width of the network (embedding dimension)
    const int block_size = 16; //maximum context length of the attention window (note: the longest name is 15 characters)
    const int n_head = 4; //number of attention heads

    const int vocab_size = 27; //26 characters + BOS

    GPT gpt(n_layer, n_embd, block_size, n_head, vocab_size);

    int params = gpt.get_num_of_params();
    std::cout << "Num Params: " << params << std::endl;

    //adam optimizer values for training
    float learning_rate = 0.01f;
    float beta1 = 0.85f;
    float beta2 = 0.99f;
    float eps_adam = 1e-8f;
    std::vector<float> m;
    m.reserve(params);
    std::vector<float> v;
    v.reserve(params);

    const int num_steps = 1000;

    for(int i = 0; i < num_steps; ++i){
        
    }

    std::cout << "Elapsed: " << t.elapsed() << std::endl;

    return 0;
}