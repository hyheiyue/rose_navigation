#pragma once
#include <condition_variable>
#include <deque>
#include <mutex>
#include <optional>
namespace rose_nav {
template<typename T>
class LockQueue {
public:
    explicit LockQueue(size_t max_size = 0): max_size_(max_size), stop_(false) {}

    void push(T&& value) {
        std::unique_lock<std::mutex> lock(mutex_);

        if (max_size_ > 0) {
            if (queue_.size() >= max_size_) {
                queue_.pop_front();
            }
        }

        queue_.push_back(std::move(value));
        lock.unlock();
        cv_.notify_one();
    }

    bool wait_and_pop(T& value) {
        std::unique_lock<std::mutex> lock(mutex_);
        cv_.wait(lock, [&] { return stop_ || !queue_.empty(); });

        if (stop_)
            return false;

        value = std::move(queue_.front());
        queue_.pop_front();
        return true;
    }

    std::optional<T> try_pop() {
        std::lock_guard<std::mutex> lock(mutex_);
        if (queue_.empty())
            return std::nullopt;

        T value = std::move(queue_.front());
        queue_.pop_front();
        return value;
    }

    size_t size() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return queue_.size();
    }

    bool empty() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return queue_.empty();
    }

    void clear() {
        std::lock_guard<std::mutex> lock(mutex_);
        queue_.clear();
    }

    void stop() {
        std::lock_guard<std::mutex> lock(mutex_);
        stop_ = true;
        cv_.notify_all();
    }

private:
    std::deque<T> queue_;
    mutable std::mutex mutex_;
    std::condition_variable cv_;
    size_t max_size_;
    bool stop_;
};
} // namespace rose_nav
