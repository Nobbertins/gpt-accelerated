    #ifndef GPT_H
    #define GPT_H

    #include <vector>
    #include "autograd.h"
    class GPT{
        public:
            GPT(const int n_layer, const int n_embd, const int block_size, const int n_head, const int vocab_size) : 
            n_layer(n_layer), n_embd(n_embd), block_size(block_size), n_head(n_head), vocab_size(vocab_size), 
            head_dim(n_embd / n_head),
            wte(generate_matrix(vocab_size, n_embd)),
            wpe(generate_matrix(block_size, n_embd)),
            lm_head(generate_matrix(vocab_size, n_embd))
            {
                build_layers();
            }
            const int n_layer;
            const int n_embd;
            const int block_size;
            const int n_head;
            const int head_dim;
            const int vocab_size;
            
            
            using Vector = std::vector<std::shared_ptr<Value>>;
            using Matrix = std::vector<Vector>;
            
            Matrix wte;
            Matrix wpe;
            Matrix lm_head;
            struct Transformer_Layer{
                Matrix attn_wq;
                Matrix attn_wk;
                Matrix attn_wv;
                Matrix attn_wproj;
                Matrix mlp1;
                Matrix mlp2;
            };
            std::vector<Transformer_Layer> layers;
            //returns all parameters in one list
            Vector get_params();
            //vector computations
            static Vector linear(const Vector &x, const Matrix &m);
            static Vector rmsnorm(const Vector &x);
            static Vector softmax(const Vector &x);
            //the forward pass
            Vector forward(int token_id, int pos_id, std::vector<std::vector<Vector>> &keys, std::vector<std::vector<Vector>> &values);
        private:
            void build_layers();
            static Matrix generate_matrix(const int nout, const int nin, const float stdev = 0.08f);
    };

    #endif