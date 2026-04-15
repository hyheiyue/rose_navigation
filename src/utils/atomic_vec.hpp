#pragma once
#include <atomic>
#include <cassert>
#include <memory>
template<typename T>
static inline void atomic_add(std::atomic<T>& v, T delta) {
    T old = v.load(std::memory_order_relaxed);
    while (!v.compare_exchange_weak(
        old,
        old + delta,
        std::memory_order_acq_rel,
        std::memory_order_relaxed
    ))
    {}
}

template<typename T>
class AtomicVec {
public:
    AtomicVec() = default;

    explicit AtomicVec(size_t n) {
        resize(n);
    }

    AtomicVec(AtomicVec&& other) noexcept {
        move_from(std::move(other));
    }

    AtomicVec& operator=(AtomicVec&& other) noexcept {
        if (this != &other) {
            move_from(std::move(other));
        }
        return *this;
    }

    AtomicVec(const AtomicVec&) = delete;
    AtomicVec& operator=(const AtomicVec&) = delete;

    void resize(size_t n) {
        data_.reset(n ? new std::atomic<T>[n] : nullptr);
        size_ = n;

        for (size_t i = 0; i < n; ++i) {
            data_[i].store(T {}, std::memory_order_relaxed);
        }
    }

    std::atomic<T>& operator[](size_t i) {
        assert(i < size_);
        return data_[i];
    }

    const std::atomic<T>& operator[](size_t i) const {
        assert(i < size_);
        return data_[i];
    }

    size_t size() const {
        return size_;
    }

    void clear() {
        data_.reset();
        size_ = 0;
    }

private:
    std::unique_ptr<std::atomic<T>[]> data_;
    size_t size_ = 0;

    void move_from(AtomicVec&& other) {
        data_ = std::move(other.data_);
        size_ = other.size_;
        other.size_ = 0;
    }
};
