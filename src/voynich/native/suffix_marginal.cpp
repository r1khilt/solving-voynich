// Exact forward log-sum over (observed offset, finite source state).
// No beam, Viterbi bookkeeping, source approximation or probability pruning.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <limits>
#include <new>
#include <unordered_map>
#include <vector>

struct Result {
    double log_likelihood;
    std::uint64_t nodes;
    std::uint64_t edges;
};
static_assert(sizeof(Result) == 24, "Unsupported result ABI");

extern "C" int suffix_marginal_abi() { return 1; }

extern "C" int suffix_marginal(
    const double* probabilities, const std::uint32_t* transitions,
    std::uint64_t states, std::uint32_t width, std::uint32_t root,
    const std::uint64_t* offsets, const std::uint32_t* letters,
    const std::uint32_t* ends, std::uint32_t length, std::uint64_t matches,
    double rho, std::uint64_t max_nodes, std::uint64_t max_edges,
    Result* result, char* error, std::uint64_t error_capacity) noexcept {
    auto fail = [&](int code, const char* message) {
        if (error && error_capacity) std::snprintf(error, error_capacity, "%s", message);
        return code;
    };
    if (!probabilities || !transitions || !offsets || !letters || !ends || !result ||
        !states || states > UINT32_MAX || !width || root >= states || length > 1000000 ||
        !max_nodes || !max_edges || !std::isfinite(rho) || !(rho > 0 && rho < 1))
        return fail(1, "Invalid native settings");
    result->log_likelihood = -std::numeric_limits<double>::infinity();
    result->nodes = 1;
    result->edges = 0;
    if (offsets[0] != 0 || offsets[length] != matches)
        return fail(1, "Invalid match offsets");
    for (std::uint32_t position = 0; position < length; ++position) {
        if (offsets[position] > offsets[position + 1] || offsets[position + 1] > matches)
            return fail(1, "Invalid match range");
        for (auto j = offsets[position]; j < offsets[position + 1]; ++j)
            if (letters[j] >= width || ends[j] <= position || ends[j] > length)
                return fail(1, "Invalid match edge");
    }
    try {
        std::vector<std::unordered_map<std::uint32_t, double>> graph(length + 1);
        graph[0].emplace(root, 0.);
        const double cont = std::log1p(-rho), stop = std::log(rho);
        for (std::uint32_t position = 0; position < length; ++position) {
            for (const auto& item : graph[position]) {
                const auto state = item.first;
                for (auto j = offsets[position]; j < offsets[position + 1]; ++j) {
                    const auto index = static_cast<std::uint64_t>(state) * width + letters[j];
                    const auto following = transitions[index];
                    const double probability = probabilities[index];
                    if (following >= states || !std::isfinite(probability) ||
                        !(probability > 0 && probability <= 1))
                        return fail(1, "Invalid source row or transition");
                    if (++result->edges > max_edges) return fail(2, "Exact lattice edge cap; no pruning");
                    const double value = item.second + (cont + std::log(probability));
                    auto inserted = graph[ends[j]].emplace(following, value);
                    if (inserted.second) {
                        if (++result->nodes > max_nodes) return fail(2, "Exact lattice node cap; no pruning");
                    } else {
                        double& current = inserted.first->second;
                        const double high = std::max(current, value), low = std::min(current, value);
                        current = high + std::log1p(std::exp(low - high));
                    }
                }
            }
            // Edges only point forward; no output paths/backpointers are needed.
            std::unordered_map<std::uint32_t, double>().swap(graph[position]);
        }
        if (graph[length].empty()) return 0;
        double peak = -std::numeric_limits<double>::infinity();
        for (const auto& item : graph[length]) peak = std::max(peak, item.second);
        double sum = 0., correction = 0.;
        for (const auto& item : graph[length]) {
            const double value = std::exp(item.second - peak) - correction;
            const double next = sum + value;
            correction = (next - sum) - value;
            sum = next;
        }
        result->log_likelihood = peak + std::log(sum) + stop;
        if (!std::isfinite(result->log_likelihood)) return fail(4, "Nonfinite supported likelihood");
        return 0;
    } catch (const std::bad_alloc&) {
        return fail(3, "Native lattice allocation failed");
    } catch (const std::exception& exception) {
        return fail(5, exception.what());
    } catch (...) {
        return fail(5, "Unknown native exception");
    }
}
