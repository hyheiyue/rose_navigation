#pragma once
#include <chrono>
namespace rose_nav::utils {
template<class PublisherT>
inline bool publisher_sub(const PublisherT& publisher) noexcept {
    return publisher && publisher->get_subscription_count() > 0;
}
template<typename Func>
void dt_once(Func&& func, std::chrono::duration<double> dt) noexcept {
    static auto last_call = std::chrono::steady_clock::now();

    auto now = std::chrono::steady_clock::now();
    if (now - last_call >= dt) {
        last_call = now;
        func();
    }
}

} // namespace rose_nav::utils