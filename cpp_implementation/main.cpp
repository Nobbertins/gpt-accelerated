#include <iostream> //talk to you
#include <fstream> //read dataset
#include <string> //read dataset

#include <vector> //the goat
#include <memory> //std::shared_ptr
#include <random> //for init weights and inference token selection
#include <cmath> //std::pow

#include "autograd.h"
#include "gpt.h"
#include "tokenizer.h"
#include "timer.h"
#include "arena.h"

int main(){

    //test
    Arena grad_arena(9999);
    Arena data_arena(9999);
    Arena tensor_arena(9999);
    Tensor<float> mat({3, 2}, grad_arena, data_arena, tensor_arena);
    mat[{0, 0}] = 5.6;
    
    Tensor<float> mat2({3, 2}, grad_arena, data_arena, tensor_arena);
    mat2[{0, 0}] = 2.3;

    Tensor<float> mat3 = mat.matmul(mat2);
    for(auto i{0uz}; i < mat3.total_size; i++) mat3.grads[i] = 1.0f;
    mat3.backward();
    std::cout << mat3[{0, 0}] << ", " << mat.grads[0] << std::endl;

    return 0;


    //start timer
    Timer t;
    //random numbers!!!
    std::mt19937 generator(std::chrono::system_clock::now().time_since_epoch().count());
    
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

    //init tokenizer (encode/decode text <-> tokens)
    const Tokenizer tokenizer(lines);

    std::cout << "Vocab Size: " << tokenizer.get_vocab_size() << std::endl;
    int num_lines = lines.size();
    std::cout << "Num Docs: " << num_lines << std::endl;

    //construct GPT2
    const int n_layer = 1; //depth of the transformer neural network (number of layers)
    const int n_embd = 16;  //width of the network (embedding dimension)
    const int block_size = 16; //maximum context length of the attention window (note: the longest name is 15 characters)
    const int n_head = 4; //number of attention heads

    const int vocab_size = 27; //26 characters + BOS

    GPT gpt(n_layer, n_embd, block_size, n_head, vocab_size);

    GPT::Vector params = gpt.get_params();
    std::cout << "Num Params: " << params.size() << std::endl;

    //adam optimizer values for training
    float learning_rate = 0.01f;
    float beta1 = 0.85f;
    float beta2 = 0.99f;
    float eps_adam = 1e-8f;
    std::vector<float> m(params.size());
    std::vector<float> v(params.size());

    double last_time = t.elapsed();
    std::cout << "Init Time: " << last_time << std::endl;

    //TRAINING TIME YIPPEEE
    const int num_steps = 1000;

    for(int step = 0; step < num_steps; step++){
        //tokenize a name
        std::string doc = lines[step % num_lines];
        std::vector<int> tokens = tokenizer.encode(doc);
        int n = tokens.size() - 1;
        if(block_size < n) n = block_size;
        //forward pass
        std::vector<std::vector<GPT::Vector>> keys(n_layer);
        std::vector<std::vector<GPT::Vector>> values(n_layer);
        GPT::Vector losses;
        for(int pos_id = 0; pos_id < n; pos_id++){
            int token_id = tokens[pos_id];
            int target_id = tokens[pos_id + 1];
            GPT::Vector logits = gpt.forward(token_id, pos_id, keys, values);
            logits = GPT::softmax(logits);  
            //cross-entropy loss:
            losses.push_back(Vlog(logits[target_id]) * std::make_shared<Value>(-1.0f));
        }
        std::shared_ptr<Value> loss = std::make_shared<Value>(0.0f);
        for(auto& l : losses) loss = loss + l;
        loss = loss / std::make_shared<Value>(static_cast<float>(n));
        
        //backward pass
        loss->grad = 1.0f;
        backward(loss);

        //adam optimizer update
        float lr_t = learning_rate * (1 - static_cast<float>(step) / num_steps);
        for(int i = 0;  i < params.size(); i++){
            m[i] = beta1 * m[i] + (1.0f - beta1) * params[i]->grad;
            v[i] = beta2 * v[i] + (1.0f - beta2) * std::pow(params[i]->grad , 2);
            float m_hat = m[i] / (1.0f - std::pow(beta1, step + 1));
            float v_hat = v[i] / (1.0f - std::pow(beta2, step + 1));
            params[i]->data -= lr_t * m_hat / (sqrtf(v_hat) + eps_adam);
            params[i]->grad = 0;
        }
        std::cout << "step " << step + 1 << " / " << num_steps << "   |   loss " << loss->data << "\r" << std::flush;
    }

    std::cout << "\nTraining Time: " << t.elapsed() - last_time << std::endl;
    last_time = t.elapsed();

    //inference 20 names
    float temperature = 0.5;
    std::cout << "\n--- inference (new, hallucinated names) ---" << std::endl;
    for(int i = 0; i < 20; ++i){
        std::vector<std::vector<GPT::Vector>> keys(n_layer), values(n_layer);
        int token_id = vocab_size - 1;//BOS
        std::vector<int> generated_tokens;
        for(int pos_id = 0; pos_id < block_size; ++pos_id){
            GPT::Vector logits = gpt.forward(token_id, pos_id, keys, values);
            std::vector<float> raw_logits;
            for(auto l : logits) raw_logits.push_back(l->data / temperature);
            //raw softmax
            float max_val = *std::max_element(raw_logits.begin(), raw_logits.end());
            float sum = 0.0f;
            for(auto &p : raw_logits) { p = expf(p - max_val); sum += p; }
            for(auto &p : raw_logits) {p /= sum;}
            //sample a token over the distribution
            std::uniform_real_distribution<float> dist(0.0f, 1.0f);
            float r = dist(generator);
            float cumulative = 0.0f;
            token_id = 0;
            for(int a = 0; a < raw_logits.size(); ++a){
                cumulative += raw_logits[a];
                if(r < cumulative){ token_id = a; break; }
            }
            if(token_id == vocab_size - 1) break; //end of name
            generated_tokens.push_back(token_id);
        }
        std::cout << "sample " << i + 1 << ": " << tokenizer.decode(generated_tokens) << std::endl;
    }

    std::cout << "Inference Time: " << t.elapsed() - last_time << std::endl;

    return 0;
}