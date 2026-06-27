#include "gpt.h"
#include "autograd.h"
#include <random>
#include <chrono>

GPT::Matrix GPT::generate_matrix(const int nout, const int nin, const float stdev) {
    static std::mt19937 generator(std::chrono::system_clock::now().time_since_epoch().count());
    std::normal_distribution<float> distribution(0.0f, stdev);
    
    Matrix output;
    output.reserve(nout);

    for (int i = 0; i < nout; i++) {
        std::vector<std::shared_ptr<Value>> v;
        v.reserve(nin);
        for (int j = 0; j < nin; j++) {
            v.push_back(std::make_shared<Value>(distribution(generator)));
        }
        output.push_back(std::move(v));
    }
    return output;
}

void GPT::build_layers(void){
    layers.reserve(n_layer);
    for(int i = 0; i < n_layer; i++){
        Transformer_Layer layer = {
            generate_matrix(n_embd, n_embd),
            generate_matrix(n_embd, n_embd),
            generate_matrix(n_embd, n_embd),
            generate_matrix(n_embd, n_embd),
            generate_matrix(n_embd * 4, n_embd),
            generate_matrix(n_embd, n_embd * 4)
        };
        layers.push_back(std::move(layer));
    }
}

//helper function for GPT::get_params
inline void append_matrix_to_vec(const GPT::Matrix &m, std::vector<std::shared_ptr<Value>> &v){
    for(const auto& row : m){
        for(const auto& p : row) v.push_back(p);
    }
}

GPT::Vector GPT::get_params(){
    GPT::Vector params;
    append_matrix_to_vec(wte, params);
    append_matrix_to_vec(wpe, params);
    append_matrix_to_vec(lm_head, params);
    for(const auto &layer : layers){
        append_matrix_to_vec(layer.attn_wq, params);
        append_matrix_to_vec(layer.attn_wk, params);
        append_matrix_to_vec(layer.attn_wv, params);
        append_matrix_to_vec(layer.attn_wproj, params);
        append_matrix_to_vec(layer.mlp1, params);
        append_matrix_to_vec(layer.mlp2, params);
    }
    return params;
}

GPT::Vector GPT::linear(const GPT::Vector &x, const Matrix &m){
    GPT::Vector output;
    for(const auto &row: m){
        std::shared_ptr<Value> sum = std::make_shared<Value>(0.0f);
        for(int i = 0; i < row.size(); i++){
            sum = sum + row[i] * x[i];
        }
        output.push_back(sum);
    }
    return output;
}

GPT::Vector GPT::rmsnorm(const GPT::Vector &x){
    std::vector<std::shared_ptr<Value>> output;
    std::shared_ptr<Value> square_sum = std::make_shared<Value>(0.0f);
    for(const auto &val : x){
        square_sum = square_sum + val * val;
    }
    std::shared_ptr<Value> ms = square_sum / std::make_shared<Value>(x.size());
    std::shared_ptr<Value> scaleFactor = Vpow((ms + std::make_shared<Value>(1e-5)), std::make_shared<Value>(-0.5f));
    for(const auto &val : x){
        output.push_back(scaleFactor * val);
    }
    return output;
}

GPT::Vector GPT::softmax(const GPT::Vector &x){
    std::vector<std::shared_ptr<Value>> output;
    float max_val = x[0]->data;
    for(const auto &val : x){
        if(val->data > max_val) max_val = val->data;
    }
    std::shared_ptr<Value> total = std::make_shared<Value>(0.0f);
    std::vector<std::shared_ptr<Value>> exps;
    for(const auto &val : x){
        std::shared_ptr<Value> exp = Vexp(val - std::make_shared<Value>(max_val));
        exps.push_back(exp);
        total = total + exp;
    }
    for(const auto &val : exps){
        output.push_back(val / total);
    }
    return output;
}
GPT::Vector GPT::forward(int token_id, int pos_id, std::vector<std::vector<Vector>> &keys, std::vector<std::vector<Vector>> &values){
    //create embedding vector to forward through GPT
    Vector x;
    //apply token and pos embeddings
    Vector tok_emb = wte[token_id];
    Vector pos_emb = wpe[pos_id];
    for(int i = 0; i < tok_emb.size(); ++i){
        x.push_back(tok_emb[i] + pos_emb[i]);
    }
    x = rmsnorm(x);
    for(int li = 0; li < n_layer; ++li){
        //kqv vectors
        Vector x_residual = x;
        x = rmsnorm(x);
        Vector q = linear(x, layers[li].attn_wq);
        Vector k = linear(x, layers[li].attn_wk);
        Vector v = linear(x, layers[li].attn_wv);
        keys[li].push_back(k);
        values[li].push_back(v);
        //attention heads
        Vector x_attn;
        for(int h = 0; h < n_head; ++h){
            int hs = h * head_dim;
            Vector q_h = Vector(q.begin() + hs, q.begin() + hs + head_dim);
            std::vector<Vector> k_h;
            //load and split attention head kv vectors from kv cache
            for(int i = 0; i < keys[li].size(); ++i)
                k_h.push_back(Vector(keys[li][i].begin() + hs, keys[li][i].begin() + hs + head_dim));
            std::vector<Vector> v_h;
            for(int i = 0; i < values[li].size(); ++i)
                v_h.push_back(Vector(values[li][i].begin() + hs, values[li][i].begin() + hs + head_dim));
            //calc attention q * all the k output
            Vector attn_logits;
            for(auto& k_hh : k_h){
                std::shared_ptr<Value> sum = std::make_shared<Value>(0.0f);
                for(int j = 0; j < head_dim; ++j){
                    sum = sum + k_hh[j] * q_h[j];
                }
                sum = sum / (Vpow(std::make_shared<Value>(head_dim), std::make_shared<Value>(0.5f)));
                attn_logits.push_back(sum);
            }
            
           //softmax result
            Vector attn_weights = softmax(attn_logits);
            
           //multiply logits by corresponding v
            Vector head_out;
            for(int j = 0; j < head_dim; ++j){
                std::shared_ptr<Value> sum = std::make_shared<Value>(0.0f);
                for(int t = 0; t < v_h.size(); ++t){
                    sum = sum + v_h[t][j] * attn_weights[t];
                }
                head_out.push_back(sum);
            }

            //append to concatenated attention output
            x_attn.insert(x_attn.end(), head_out.begin(), head_out.end());
        }
        
        x = linear(x_attn, layers[li].attn_wproj);
            
        //add back residual (one of the core transformer ideas, attn and MLP blocks calculate differences not direct outputs)
        for(int i = 0; i < x.size(); ++i){
            x[i] = x[i] + x_residual[i];
        }

        //MLP block
        x_residual = x;
        x = rmsnorm(x);
        x = linear(x, layers[li].mlp1);
        for(int i = 0; i < x.size(); ++i){
            x[i] = Vrelu(x[i]);
        }
        x = linear(x, layers[li].mlp2);
        for(int i = 0; i < x.size(); ++i){
            x[i] = x[i] + x_residual[i];
        }
    }
    //convert model output (embedding) to vocab distribution
    x = linear(x, lm_head);
    return x;
}