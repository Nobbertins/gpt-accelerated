#include <memory>
#include <vector>
#include <cmath>
#include "autograd.h"

//builds topo sort of graph
void buildTopo(std::shared_ptr<Value> &v, std::vector<std::shared_ptr<Value>> &topo, std::unordered_set<std::shared_ptr<Value>> &visited){
    if(visited.insert(v).second){
        for(auto& child : v->children){
            buildTopo(child, topo, visited);
        }
        topo.push_back(v);
    }
}

void backward(std::shared_ptr<Value> &v){ //assumes v->grad is set
    std::vector<std::shared_ptr<Value>> topo;
    std::unordered_set<std::shared_ptr<Value>> visited;
    buildTopo(v, topo, visited);
    std::reverse(topo.begin(), topo.end());
    //apply backprop
    for(auto& node : topo){
        for(int i = 0; i < node->children.size(); i++){
            (node->children[i])->grad += node->grad * node->localGrads[i];
        }
    }
}

std::shared_ptr<Value> operator+(const std::shared_ptr<Value> a, const std::shared_ptr<Value> b){
    std::vector<std::shared_ptr<Value>> children = {a, b};
    std::vector<float> grads = {1.0f, 1.0f};
    return std::make_shared<Value>(a->data + b->data, children, grads);
}

std::shared_ptr<Value> operator*(const std::shared_ptr<Value> a, const std::shared_ptr<Value> b){
    std::vector<std::shared_ptr<Value>> children = {a, b};
    std::vector<float> grads = {b->data, a->data};
    return std::make_shared<Value>(a->data * b->data, children, grads);
}

std::shared_ptr<Value> operator-(const std::shared_ptr<Value> a, const std::shared_ptr<Value> b){
    std::vector<std::shared_ptr<Value>> children = {a, b};
    std::vector<float> grads = {1.0f, -1.0f};
    return std::make_shared<Value>(a->data - b->data, children, grads);
}

std::shared_ptr<Value> operator/(const std::shared_ptr<Value> a, const std::shared_ptr<Value> b){
    std::vector<std::shared_ptr<Value>> children = {a, b};
    std::vector<float> grads = {1.0f/(b->data), a->data * -1.0f/(b->data * b->data)};
    return std::make_shared<Value>(a->data / b->data, children, grads);
}

std::shared_ptr<Value> Vrelu(const std::shared_ptr<Value> a){
    std::vector<std::shared_ptr<Value>> children = {a};
    std::vector<float> grads = {static_cast<float>(a->data > 0)};
    return std::make_shared<Value>(std::max(0.0f, a->data), children, grads);
}

std::shared_ptr<Value> Vlog(const std::shared_ptr<Value> a){
    std::vector<std::shared_ptr<Value>> children = {a};
    std::vector<float> grads = {1.0f/a->data};
    return std::make_shared<Value>(logf(a->data), children, grads);
}

std::shared_ptr<Value> Vexp(const std::shared_ptr<Value> a){
    std::vector<std::shared_ptr<Value>> children = {a};
    std::vector<float> grads = {expf(a->data)};
    return std::make_shared<Value>(expf(a->data), children, grads);
}

std::shared_ptr<Value> Vpow(const std::shared_ptr<Value> a, const std::shared_ptr<Value> b){
    std::vector<std::shared_ptr<Value>> children = {a};
    std::vector<float> grads = {b->data * powf(a->data, b->data - 1.0f)};
    return std::make_shared<Value>(powf(a->data, b->data), children, grads);
}