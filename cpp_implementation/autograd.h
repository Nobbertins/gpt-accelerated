#ifndef AUTOGRAD_H
#define AUTOGRAD_H


#include <memory>
#include <vector>
#include <unordered_set>
#include <algorithm>

#include <numeric>
#include <initializer_list>
#include <functional>
#include <stdexcept>
#include <cmath>

#include "arena.h"

//Nodes for autograd
class Value{
    public:
        explicit Value(float x) : data(x), grad(0){}
        Value(float x, std::vector<std::shared_ptr<Value>> c, std::vector<float> g) : data(x), grad(0), children(std::move(c)), localGrads(std::move(g)) {}
        float data;
        float grad;
        std::vector<std::shared_ptr<Value>> children; //nodes resulting in current node
        std::vector<float> localGrads; //gradient w.r.t each child
};

template <typename T>
class Tensor{
    public:
        std::vector<std::size_t> shape;
        std::size_t total_size;

        T* data = nullptr;
        T* grads = nullptr;
        const Tensor** children = nullptr;

        std::function<void()> backward  = [](){throw std::runtime_error("undefined backward");};

        //human-initialized tensors
        Tensor(std::initializer_list<std::size_t> dims, Arena& grad_arena, Arena& data_arena, Arena& tensor_arena, std::size_t num_children=2)
        : shape(dims), grad_arena(grad_arena), data_arena(data_arena), tensor_arena(tensor_arena) {
            tensor_init(num_children);
            calculate_jumps();
        }
        
        //tensor op initialized tensors
        Tensor(const std::vector<std::size_t>& dims, Arena& grad_arena, Arena& data_arena, Arena& tensor_arena, std::size_t num_children=2)
        : shape(dims), grad_arena(grad_arena), data_arena(data_arena), tensor_arena(tensor_arena) {
            tensor_init(num_children);
            calculate_jumps();
        }

        //data access operators
        const T& operator[](std::initializer_list<std::size_t> dims) const {
            auto idx = std::inner_product(dims.begin(), dims.end(), jumps.begin(), 0uz);
            if (idx >= total_size) throw std::runtime_error("Tensor Out of Bounds");
            return data[idx];
        }
        T& operator[](std::initializer_list<std::size_t> dims){
            auto idx = std::inner_product(dims.begin(), dims.end(), jumps.begin(), 0uz);
            if (idx >= total_size) throw std::runtime_error("Tensor Out of Bounds");
            return data[idx];
        }
        //pass data or grad
        const T& get_value(const T* array, std::initializer_list<std::size_t> dims) const{
            auto idx = std::inner_product(dims.begin(), dims.end(), jumps.begin(), 0uz);
            if (idx >= total_size) throw std::runtime_error("Tensor Out of Bounds");
            return array[idx];
        }
        T& get_value(T* array, std::initializer_list<std::size_t> dims) const {
            auto idx = std::inner_product(dims.begin(), dims.end(), jumps.begin(), 0uz);
            if (idx >= total_size) throw std::runtime_error("Tensor Out of Bounds");
            return array[idx];
        }

        //if you try to use a transpose's gradients, you die!
        const Tensor transpose() const{
            if(shape.size() == 2) throw std::runtime_error("you can only transpose a matrix");
            Tensor out({shape[1], shape[0]}, grad_arena, data_arena, tensor_arena);
            for(auto row{0uz}; row < shape[0]; ++row){
                for(auto col{0uz}; col < shape[1]; ++col){
                    out[{row, col}] = (*this)[{col, row}];
                }
            }
            return out;
        }
        //TENSOR OPS (1-D)
        Tensor operator+(const Tensor& other) const{
            Tensor out(shape, grad_arena, data_arena, tensor_arena);
            std::transform(data, data + total_size, other.data, out.data, std::plus<>{});

            out.children[0] = this;
            out.children[1] = &other;

            Tensor* out_ptr = &out;

            out.backward = [out_ptr](){
                //A + B = C (dC/dA, dC/dB = 1)
                const Tensor* a = out_ptr->children[0];
                const Tensor* b = out_ptr->children[1];
                for(auto i{0uz}; i < out_ptr->total_size; i++){
                    a->grads[i] += out_ptr->grads[i];
                    b->grads[i] += out_ptr->grads[i];
                }
            };
            return out;
        }
        Tensor operator-(const Tensor& other) const {
            Tensor out(shape, grad_arena, data_arena, tensor_arena);
            std::transform(data, data + total_size, other.data, out.data, std::minus<>{});
            
            out.children[0] = this;
            out.children[1] = &other;

            Tensor* out_ptr = &out;

            out.backward = [out_ptr](){
                //A - B = C (dC/dA = 1, dC/dB = -1)
                const Tensor* a = out_ptr->children[0];
                const Tensor* b = out_ptr->children[1];
                for(auto i{0uz}; i < out_ptr->total_size; i++){
                    a->grads[i] += out_ptr->grads[i];
                    b->grads[i] += -out_ptr->grads[i];
                }
            };
            return out;
        }
        Tensor operator*(const Tensor& other) const {
            Tensor out(shape, grad_arena, data_arena, tensor_arena);
            std::transform(data, data + total_size, other.data, out.data, std::multiplies<T>());

            out.children[0] = this;
            out.children[1] = &other;

            Tensor* out_ptr = &out;

            out.backward = [out_ptr](){
                //A * B = C (dC/dA = B, dC/dB = A)
                const Tensor* a = out_ptr->children[0];
                const Tensor* b = out_ptr->children[1];
                for(auto i{0uz}; i < out_ptr->total_size; i++){
                    a->grads[i] += out_ptr->grads[i] * b->data[i];
                    b->grads[i] += out_ptr->grads[i] * a->data[i];
                }
            };
            return out;
        }
        Tensor operator/(const Tensor& other) const {
            Tensor out(shape, grad_arena, data_arena, tensor_arena);
            std::transform(data, data + total_size, other.data, out.data, std::divides<T>());

            out.children[0] = this;
            out.children[1] = &other;

            Tensor* out_ptr = &out;

            out.backward = [out_ptr](){
                //A / B = C (dC/dA = 1/B, dC/dB = -AB^-2)
                const Tensor* a = out_ptr->children[0];
                const Tensor* b = out_ptr->children[1];
                for(auto i{0uz}; i < out_ptr->total_size; i++){
                    a->grads[i] += out_ptr->grads[i] / b->data[i];
                    b->grads[i] += out_ptr->grads[i] / (b->data[i] * b->data[i]) * -(a->data[i]);
                }
            };
            return out;
        }
        Tensor relu() const {
            Tensor out(shape, grad_arena, data_arena, tensor_arena, 1uz);
            std::transform(data, data + total_size, out.data, [](T num){return std::max(0, num);});

            out.children[0] = this;

            Tensor* out_ptr = &out;

            out.backward = [out_ptr](){
                //relu(A) = C (dC/dA = 1 if A > 0)
                const Tensor* a = out_ptr->children[0];
                for(auto i{0uz}; i < out_ptr->total_size; i++){
                    if(a->data[i] > 0) a->grads[i] += out_ptr->grads[i];
                }
            };
            return out;
        }
        Tensor log() const {
            Tensor out(shape, grad_arena, data_arena, tensor_arena, 1uz);
            std::transform(data, data + total_size, out.data, [](T num){return std::log(num);});

            out.children[0] = this;

            Tensor* out_ptr = &out;

            out.backward = [out_ptr](){
                //log(A) = C (dC/dA = 1 / A)
                const Tensor* a = out_ptr->children[0];
                for(auto i{0uz}; i < out_ptr->total_size; i++){
                    a->grads[i] += out_ptr->grads[i] / a->data[i];
                }
            };
            return out;
        }
        Tensor exp() const {
            Tensor out(shape, grad_arena, data_arena, tensor_arena, 1uz);
            std::transform(data, data + total_size, out.data, [](T num){return std::exp(num);});

            out.children[0] = this;

            Tensor* out_ptr = &out;

            out.backward = [out_ptr](){
                //exp(A) = C (dC/dA = exp(A))
                const Tensor* a = out_ptr->children[0];
                for(auto i{0uz}; i < out_ptr->total_size; i++){
                    a->grads[i] += out_ptr->grads[i] * std::exp(a->data[i]);
                }
            };
            return out;
        }
        Tensor pow(T exponent) const {
            Tensor out(shape, grad_arena, data_arena, tensor_arena, 1uz);
            std::transform(data, data + total_size, out.data, [exponent](T num){return std::pow(num, exponent);});

            out.children[0] = this;

            Tensor* out_ptr = &out;

            out.backward = [out_ptr, exponent](){
                //A^b = C (dC/dA = bA^(b-1))
                const Tensor* a = out_ptr->children[0];
                for(auto i{0uz}; i < out_ptr->total_size; i++){
                    a->grads[i] += out_ptr->grads[i] * (exponent * std::pow(a->data[i], exponent - 1));
                }
            };
            return out;
        }
        //give your data or grad pointers and each tensor object, and this helper function shall cook
        static void matrix_multiply(const T* a, const T* b, T* out, const Tensor& a_tensor, const Tensor& b_tensor, const Tensor& out_tensor){
            for(auto col2{0uz}; col2 < b_tensor.shape[1]; col2++){
                for(auto row{0uz}; row < a_tensor.shape[0]; row++){
                    T product = 0;
                    for(auto col{0uz}; col < a_tensor.shape[1]; col++){
                        product += a_tensor.get_value(a, {row, col}) * b_tensor.get_value(b, {col, col2});
                    }
                    out_tensor.get_value(out, {row, col2}) = product;
                }
            }
        }
        //2-D
        Tensor matmul(const Tensor& other) const{

            if(shape[1] == other.shape[0]) throw std::runtime_error("wrong matmul dimensions...");
            Tensor out({shape[0], other.shape[1]}, grad_arena, data_arena, tensor_arena);
            matrix_multiply(this->data, other.data, out.data, *this, other, out);

            out.children[0] = this;
            out.children[1] = &other;

            Tensor* out_ptr = &out;

            out.backward = [out_ptr](){
                //AB = C (dL/dA = grad matrix * Bt, dL/dB = At * grad matrix)
                const Tensor* a = out_ptr->children[0];
                const Tensor* b = out_ptr->children[1];

                //don't use these grads
                const Tensor a_t = a->transpose();
                const Tensor b_t = b->transpose();

                matrix_multiply(out_ptr->grads, b_t.data, a->grads, *out_ptr, b_t, *a);
                matrix_multiply(a_t.data, out_ptr->grads, b->grads, a_t, *out_ptr, *b);
            };
            return out;
        }

    private:
        std::vector<std::size_t> jumps;
        Arena& grad_arena;
        Arena& data_arena;
        Arena& tensor_arena;
        void calculate_jumps(){
            //calculate jumps (each jump is the product of all subsequent dims)
            jumps.resize(shape.size());
            jumps[shape.size() - 1] = 1;
            for(auto i{shape.size() - 1}; i > 0; i--){
                jumps[i - 1] = jumps[i] * shape[i];
            }
        }
        void tensor_init(std::size_t num_children){
            total_size = std::accumulate(shape.begin(), shape.end(), 1uz, std::multiplies<std::size_t>());
            data = data_arena.alloc<T>(total_size);
            grads = grad_arena.alloc<T>(total_size);
            children = tensor_arena.alloc<const Tensor*>(num_children);
            for(auto i{0uz}; i < total_size; i++){
                data[i] = 0;
                grads[i] = 0;
            }
        }
};

//builds topo sort of graph
void buildTopo(std::shared_ptr<Value> &v, std::vector<std::shared_ptr<Value>> &topo, std::unordered_set<std::shared_ptr<Value>> &visited);
void backward(std::shared_ptr<Value> &v);

std::shared_ptr<Value> operator+(const std::shared_ptr<Value> a, const std::shared_ptr<Value> b);
std::shared_ptr<Value> operator*(const std::shared_ptr<Value> a, const std::shared_ptr<Value> b);
std::shared_ptr<Value> operator-(const std::shared_ptr<Value> a, const std::shared_ptr<Value> b);
std::shared_ptr<Value> operator/(const std::shared_ptr<Value> a, const std::shared_ptr<Value> b);
std::shared_ptr<Value> Vrelu(const std::shared_ptr<Value> a);
std::shared_ptr<Value> Vlog(const std::shared_ptr<Value> a);
std::shared_ptr<Value> Vexp(const std::shared_ptr<Value> a);
std::shared_ptr<Value> Vpow(const std::shared_ptr<Value> a, const std::shared_ptr<Value> b);

#endif