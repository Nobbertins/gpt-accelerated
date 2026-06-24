#ifndef AUTOGRAD_H
#define AUTOGRAD_H

#include <memory>
#include <vector>
#include <unordered_set>
#include <algorithm>

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

//builds topo sort of graph
void buildTopo(std::shared_ptr<Value> v, std::vector<std::shared_ptr<Value>> &topo, std::unordered_set<std::shared_ptr<Value>> &visited);
void backward(std::shared_ptr<Value> v);

std::shared_ptr<Value> operator+(const std::shared_ptr<Value> a, const std::shared_ptr<Value> b);
std::shared_ptr<Value> operator*(const std::shared_ptr<Value> a, const std::shared_ptr<Value> b);
std::shared_ptr<Value> operator-(const std::shared_ptr<Value> a, const std::shared_ptr<Value> b);
std::shared_ptr<Value> operator/(const std::shared_ptr<Value> a, const std::shared_ptr<Value> b);
std::shared_ptr<Value> Vrelu(const std::shared_ptr<Value> a);
std::shared_ptr<Value> Vlog(const std::shared_ptr<Value> a);
std::shared_ptr<Value> Vexp(const std::shared_ptr<Value> a);
std::shared_ptr<Value> Vpow(const std::shared_ptr<Value> a, const std::shared_ptr<Value> b);

#endif