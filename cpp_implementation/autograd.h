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

        T* data;

        //mutable so these elements can be changed through a const Tensor&
        mutable T* grads;
        mutable const Tensor** children;

        std::function<void()> backward  = [](){throw std::runtime_error("undefined backward");};

        //human-initialized tensors
        Tensor(std::initializer_list<std::size_t> dims, Arena& grad_arena, Arena& data_arena, Arena& tensor_arena) 
        : shape(dims), grad_arena(grad_arena), data_arena(data_arena), tensor_arena(tensor_arena) {

            total_size = std::accumulate(shape.begin(), shape.end(), 1uz, std::multiplies<std::size_t>());
            data = data_arena.alloc<T>(total_size);
            grads = grad_arena.alloc<T>(total_size);
            children = tensor_arena.alloc<const Tensor*>(2);

            for(auto i{0uz}; i < total_size; i++){
                data[i] = 0;
                grads[i] = 0;
            }

            calculate_jumps();
        }
        
        //tensor op initialized tensors
        Tensor(const std::vector<std::size_t>& dims, Arena& grad_arena, Arena& data_arena, Arena& tensor_arena) 
        : shape(dims), grad_arena(grad_arena), data_arena(data_arena), tensor_arena(tensor_arena) {

            total_size = std::accumulate(shape.begin(), shape.end(), 1uz, std::multiplies<std::size_t>());
            data = data_arena.alloc<T>(total_size);
            grads = grad_arena.alloc<T>(total_size);
            children = tensor_arena.alloc<const Tensor*>(2);

            for(auto i{0uz}; i < total_size; i++){
                data[i] = 0;
                grads[i] = 0;
            }

            calculate_jumps();
        }

        const T& operator[](std::initializer_list<std::size_t> dims) const {
            auto idx{0uz};
            for(auto i{0uz}; i < jumps.size(); i++){
                idx += dims.begin()[i] * jumps[i];
            }
            if (idx >= total_size) throw std::runtime_error("Tensor Out of Bounds");
            return data[idx];
        }

        T& operator[](std::initializer_list<std::size_t> dims){
            auto idx{0uz};
            for(auto i{0uz}; i < jumps.size(); i++){
                idx += dims.begin()[i] * jumps[i];
            }
            if (idx >= total_size) throw std::runtime_error("Tensor Out of Bounds");
            return data[idx];
        }

        Tensor operator+(const Tensor& other) const{
            Tensor out(shape, grad_arena, data_arena, tensor_arena);
            for(auto i{0uz}; i < total_size; i++){
                out.data[i] = data[i] + other.data[i];
            }
            
            out.children[0] = this;
            out.children[1] = &other;

            Tensor* out_ptr = &out;

            out.backward = [out_ptr](){

                //a + b = c (dc/da = 1)
                for(auto child{0uz}; child < 2; child++){
                    for(auto i{0uz}; i < out_ptr->total_size; i++){
                        ((out_ptr->children[child])->grads)[i] = out_ptr->grads[i];
                    }
                }
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