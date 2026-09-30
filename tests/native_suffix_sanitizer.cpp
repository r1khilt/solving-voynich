// Standalone memory/undefined-behavior validation; no corpus or fitted key access.
#include <cassert>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>
struct Result { double log_likelihood; std::uint64_t nodes; std::uint64_t edges; };
extern "C" int suffix_marginal(const double*, const std::uint32_t*, std::uint64_t,
    std::uint32_t, std::uint32_t, const std::uint64_t*, const std::uint32_t*,
    const std::uint32_t*, std::uint32_t, std::uint64_t, double, std::uint64_t,
    std::uint64_t, Result*, char*, std::uint64_t) noexcept;
int main() {
    double probabilities[] = {.7, .3, .2, .8, .4, .6};
    std::uint32_t transitions[] = {1, 2, 1, 0, 2, 1};
    unsigned checks = 0;
    for (std::uint32_t length = 0; length < 129; ++length) {
        std::vector<std::uint64_t> offsets{0};
        std::vector<std::uint32_t> letters, ends;
        for (std::uint32_t offset = 0; offset < length; ++offset) {
            letters.push_back(0); ends.push_back(offset + 1);
            if (offset + 2 <= length) { letters.push_back(1); ends.push_back(offset + 2); }
            offsets.push_back(letters.size());
        }
        // Supply non-null allocated buffers even for zero matches.
        std::uint32_t dummy = 0;
        auto call = [&](std::uint64_t nodes, std::uint64_t edges, double rho) {
            Result result; char error[256];
            int code = suffix_marginal(probabilities, transitions, 3, 2, 0, offsets.data(),
                letters.empty() ? &dummy : letters.data(), ends.empty() ? &dummy : ends.data(),
                length, letters.size(), rho, nodes, edges, &result, error, sizeof(error));
            if (code == 0) assert(std::isfinite(result.log_likelihood));
            ++checks;
            return code;
        };
        assert(call(500000, 2000000, .2) == 0);
        assert(call(500000, 2000000, 0.) == 1);
        if (length > 2) {
            assert(call(1, 2000000, .2) == 2);
            assert(call(500000, 1, .2) == 2);
            ends[0] = 0; assert(call(500000, 2000000, .2) == 1); ends[0] = 1;
            letters[0] = 2; assert(call(500000, 2000000, .2) == 1); letters[0] = 0;
            transitions[0] = 3; assert(call(500000, 2000000, .2) == 1); transitions[0] = 1;
            probabilities[0] = std::numeric_limits<double>::quiet_NaN();
            assert(call(500000, 2000000, .2) == 1); probabilities[0] = .7;
        }
    }
    assert(checks == 1014);
}
