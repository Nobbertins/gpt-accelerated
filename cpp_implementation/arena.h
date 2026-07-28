#ifndef ARENA_H
#define ARENA_H

#include <cstddef>
#include <memory>

class Arena{
    public:
        Arena(std::size_t bytes){
            start = next = new std::byte[bytes];
            end = start + bytes;
        }
        ~Arena(){
            delete[] start;
        }
        template <typename T>
        T* alloc(std::size_t num){
            void* ptr = next;
            std::size_t space = end - next;
            void* aligned = std::align(alignof(T), sizeof(T) * num, ptr, space);
            if(aligned == nullptr) throw std::runtime_error("Allocator Size Exceeded!");
            next = static_cast<std::byte*>(aligned) + sizeof(T) * num;
            return static_cast<T*>(aligned);
        }
    private:
        std::byte* start;
        std::byte* next;
        std::byte* end;
};


#endif
