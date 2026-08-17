#include <stringzillas/stringzillas.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <limits>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

static std::vector<std::string> load_lines(char const *path, std::size_t limit = 0) {
    std::ifstream input(path);
    if (!input) throw std::runtime_error(std::string("failed to open ") + path);
    std::vector<std::string> lines;
    std::string line;
    while ((!limit || lines.size() != limit) && std::getline(input, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        lines.push_back(std::move(line));
    }
    return lines;
}

static std::vector<sz_string_view_t> make_views(std::vector<std::string> const &strings) {
    std::vector<sz_string_view_t> views;
    views.reserve(strings.size());
    for (auto const &string : strings) views.push_back({string.data(), string.size()});
    return views;
}

struct sparse_matches_t {
    std::vector<sz_u64_t> query_ids;
    std::vector<sz_u32_t> dictionary_ids;
    std::vector<sz_u8_t> distances;

    void resize(std::size_t count) {
        query_ids.resize(count);
        dictionary_ids.resize(count);
        distances.resize(count);
    }
};

static bool mode_enabled(std::string_view modes, std::string_view requested) {
    while (!modes.empty()) {
        std::size_t const separator = modes.find(',');
        std::string_view const mode = modes.substr(0, separator);
        if (mode == requested) return true;
        if (separator == std::string_view::npos) break;
        modes.remove_prefix(separator + 1);
    }
    return false;
}

static double percentile(std::vector<double> values, double fraction) {
    if (values.empty()) return 0;
    std::sort(values.begin(), values.end());
    std::size_t const rank = static_cast<std::size_t>(fraction * (values.size() - 1));
    return values[rank];
}

static bool dump_matches(std::size_t dictionary_size, std::size_t queries_size, std::uint8_t bound,
                         sparse_matches_t const &matches, std::string const &path) {
    std::vector<std::vector<std::pair<sz_u32_t, sz_u8_t>>> grouped(queries_size);
    for (std::size_t match = 0; match != matches.query_ids.size(); ++match) {
        if (matches.query_ids[match] >= queries_size || matches.dictionary_ids[match] >= dictionary_size ||
            matches.distances[match] > bound)
            return false;
        grouped[matches.query_ids[match]].push_back({matches.dictionary_ids[match], matches.distances[match]});
    }
    for (auto &query_matches : grouped) std::sort(query_matches.begin(), query_matches.end());

    std::ofstream output(path, std::ios::binary);
    if (!output) return false;
    char const magic[8] = {'S', 'Z', 'L', 'E', 'V', '0', '0', '1'};
    std::uint64_t const dictionary_size_u64 = dictionary_size;
    std::uint64_t const queries_size_u64 = queries_size;
    output.write(magic, sizeof(magic));
    output.write(reinterpret_cast<char const *>(&dictionary_size_u64), sizeof(dictionary_size_u64));
    output.write(reinterpret_cast<char const *>(&queries_size_u64), sizeof(queries_size_u64));
    output.write(reinterpret_cast<char const *>(&bound), sizeof(bound));
    for (auto const &query_matches : grouped) {
        std::uint64_t const matches_size = query_matches.size();
        output.write(reinterpret_cast<char const *>(&matches_size), sizeof(matches_size));
        for (auto const &match : query_matches) {
            output.write(reinterpret_cast<char const *>(&match.first), sizeof(match.first));
            output.write(reinterpret_cast<char const *>(&match.second), sizeof(match.second));
        }
    }
    return output.good();
}

template <bool utf8_>
static sz_status_t index_init(sz_sequence_t const *dictionary, sz_size_t max_distance, void **index,
                              char const **error) {
    if constexpr (utf8_)
        return szs_levenshtein_index_utf8_init(dictionary, max_distance, NULL, sz_caps_sp_k, index, error);
    else return szs_levenshtein_index_init(dictionary, max_distance, NULL, sz_caps_sp_k, index, error);
}

template <bool utf8_>
static sz_status_t index_find(void *index, szs_device_scope_t device, sz_sequence_t const *queries,
                              sz_size_t bound, sparse_matches_t &matches, sz_size_t *matches_found,
                              char const **error) {
    sz_size_t const capacity = matches.query_ids.size();
    sz_u64_t *query_ids = capacity ? matches.query_ids.data() : NULL;
    sz_u32_t *dictionary_ids = capacity ? matches.dictionary_ids.data() : NULL;
    sz_u8_t *distances = capacity ? matches.distances.data() : NULL;
    if constexpr (utf8_)
        return szs_levenshtein_index_utf8_find(index, device, queries, bound, query_ids, dictionary_ids, distances,
                                               capacity, matches_found, error);
    else
        return szs_levenshtein_index_find(index, device, queries, bound, query_ids, dictionary_ids, distances,
                                          capacity, matches_found, error);
}

template <bool utf8_>
static void index_free(void *index) {
    if constexpr (utf8_) szs_levenshtein_index_utf8_free(index);
    else szs_levenshtein_index_free(index);
}

template <bool utf8_>
static int run(std::vector<std::string> const &dictionary, std::vector<std::string> const &queries,
               std::vector<std::uint8_t> const &max_distances, std::string const &dump_prefix,
               int query_repeats, std::size_t query_threads, std::size_t batches_per_repeat,
               std::size_t cache_evict_bytes, std::string_view modes, bool shared_index) {
    std::vector<sz_string_view_t> dictionary_views = make_views(dictionary);
    std::vector<sz_string_view_t> query_views = make_views(queries);
    sz_sequence_t dictionary_sequence, query_sequence;
    sz_sequence_from_string_views(dictionary_views.data(), dictionary_views.size(), &dictionary_sequence);
    sz_sequence_from_string_views(query_views.data(), query_views.size(), &query_sequence);

    char const *error = NULL;
    szs_device_scope_t device = NULL;
    if (szs_device_scope_init_cpu_cores(query_threads, &device, &error) != sz_success_k) {
        std::cerr << "device initialization failed: " << (error ? error : "unknown error") << '\n';
        return 3;
    }

    std::vector<std::uint8_t> cache_evict_buffer(cache_evict_bytes);
    std::uint64_t cache_evict_checksum = 0;
    for (std::uint8_t max_distance : max_distances) {
        void *index = NULL;
        auto const build_start = std::chrono::steady_clock::now();
        if (sz_status_t status = index_init<utf8_>(&dictionary_sequence, max_distance, &index, &error);
            status != sz_success_k) {
            std::cerr << "build failed: " << int(status) << " " << (error ? error : "") << '\n';
            szs_device_scope_free(device);
            return 4;
        }
        double const build_seconds =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - build_start).count();
        std::cout << "k=" << unsigned(max_distance) << " build=" << build_seconds << "s\n";

        std::uint8_t const first_bound = shared_index ? 1 : max_distance <= 2 ? max_distance : 3;
        for (std::uint8_t bound = first_bound; bound <= max_distance; ++bound) {
            if (mode_enabled(modes, "cold")) {
                for (int repeat = 0; repeat != query_repeats; ++repeat) {
                    void *cold_index = NULL;
                    auto const cold_build_start = std::chrono::steady_clock::now();
                    if (sz_status_t status = index_init<utf8_>(&dictionary_sequence, max_distance, &cold_index, &error);
                        status != sz_success_k) {
                        std::cerr << "cold build failed: " << int(status) << " " << (error ? error : "") << '\n';
                        index_free<utf8_>(index);
                        szs_device_scope_free(device);
                        return 5;
                    }
                    double const cold_build_seconds =
                        std::chrono::duration<double>(std::chrono::steady_clock::now() - cold_build_start).count();
                    for (std::size_t offset = 0; offset < cache_evict_buffer.size(); offset += 64) {
                        ++cache_evict_buffer[offset];
                        cache_evict_checksum += cache_evict_buffer[offset];
                    }
                    sparse_matches_t cold_matches;
                    auto const start = std::chrono::steady_clock::now();
                    sz_size_t cold_required = 0;
                    sz_status_t status = index_find<utf8_>(cold_index, device, &query_sequence, bound, cold_matches,
                                                            &cold_required, &error);
                    if (status != sz_success_k && status != sz_unexpected_dimensions_k) {
                        std::cerr << "cold sizing failed: " << int(status) << " " << (error ? error : "") << '\n';
                        index_free<utf8_>(cold_index);
                        index_free<utf8_>(index);
                        szs_device_scope_free(device);
                        return 5;
                    }
                    cold_matches.resize(cold_required);
                    sz_size_t cold_count = 0;
                    status = index_find<utf8_>(cold_index, device, &query_sequence, bound, cold_matches, &cold_count,
                                               &error);
                    double const elapsed =
                        std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
                    index_free<utf8_>(cold_index);
                    if (status != sz_success_k || cold_count != cold_required) {
                        std::cerr << "cold search failed: " << int(status) << " " << (error ? error : "") << '\n';
                        index_free<utf8_>(index);
                        szs_device_scope_free(device);
                        return 5;
                    }
                    std::cout << "k=" << unsigned(bound) << " mode=cold_end_to_end repeat=" << repeat
                              << " query=" << elapsed << "s matches=" << cold_count
                              << " prep_build=" << cold_build_seconds << "s threads=" << query_threads << '\n';
                }
            }

            sparse_matches_t matches;
            sz_size_t required = 0;
            sz_status_t sizing_status = index_find<utf8_>(index, device, &query_sequence, bound, matches,
                                                           &required, &error);
            if (sizing_status != sz_success_k && sizing_status != sz_unexpected_dimensions_k) {
                std::cerr << "output sizing failed: " << int(sizing_status) << " " << (error ? error : "") << '\n';
                index_free<utf8_>(index);
                szs_device_scope_free(device);
                return 5;
            }
            matches.resize(required);
            sz_size_t materialized = 0;
            if (sz_status_t status = index_find<utf8_>(index, device, &query_sequence, bound, matches, &materialized,
                                                        &error);
                status != sz_success_k || materialized != required) {
                std::cerr << "correctness materialization failed: " << int(status) << " " << (error ? error : "")
                          << '\n';
                index_free<utf8_>(index);
                szs_device_scope_free(device);
                return 6;
            }

            for (int repeat = 0; mode_enabled(modes, "warm") && repeat != query_repeats; ++repeat) {
                for (std::size_t offset = 0; offset < cache_evict_buffer.size(); offset += 64) {
                    ++cache_evict_buffer[offset];
                    cache_evict_checksum += cache_evict_buffer[offset];
                }
                auto const start = std::chrono::steady_clock::now();
                sz_size_t matches_count = 0;
                for (std::size_t batch = 0; batch != batches_per_repeat; ++batch) {
                    sz_size_t batch_matches = 0;
                    if (sz_status_t status = index_find<utf8_>(index, device, &query_sequence, bound, matches,
                                                                &batch_matches, &error);
                        status != sz_success_k) {
                        std::cerr << "search failed: " << int(status) << " " << (error ? error : "") << '\n';
                        index_free<utf8_>(index);
                        szs_device_scope_free(device);
                        return 6;
                    }
                    if (batch_matches != required) {
                        std::cerr << "search returned a different result count after output sizing\n";
                        index_free<utf8_>(index);
                        szs_device_scope_free(device);
                        return 6;
                    }
                    matches_count += batch_matches;
                }
                double const elapsed =
                    std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
                std::cout << "k=" << unsigned(bound) << " mode=warm_presized repeat=" << repeat
                          << " query=" << elapsed / batches_per_repeat << "s matches="
                          << matches_count / batches_per_repeat << " threads=" << query_threads << " batches="
                          << batches_per_repeat << " cache_evict_bytes=" << cache_evict_bytes
                          << " output_element_bytes="
                          << sizeof(sz_u64_t) + sizeof(sz_u32_t) + sizeof(sz_u8_t) << '\n';
            }

            for (int repeat = 0; mode_enabled(modes, "steady") && repeat != query_repeats; ++repeat) {
                for (std::size_t offset = 0; offset < cache_evict_buffer.size(); offset += 64) {
                    ++cache_evict_buffer[offset];
                    cache_evict_checksum += cache_evict_buffer[offset];
                }
                sparse_matches_t growable;
                if (queries.size() > std::numeric_limits<std::size_t>::max() / 8) {
                    std::cerr << "query count is too large for the starting output estimate\n";
                    index_free<utf8_>(index);
                    szs_device_scope_free(device);
                    return 6;
                }
                auto const start = std::chrono::steady_clock::now();
                growable.resize(queries.size() * 8);
                sz_size_t found = 0;
                sz_status_t status = index_find<utf8_>(index, device, &query_sequence, bound, growable, &found, &error);
                bool const retried = status == sz_unexpected_dimensions_k && found > growable.query_ids.size();
                if (retried) {
                    growable.resize(found);
                    status = index_find<utf8_>(index, device, &query_sequence, bound, growable, &found, &error);
                }
                double const elapsed =
                    std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
                if (status != sz_success_k || found != required) {
                    std::cerr << "steady search failed: " << int(status) << " " << (error ? error : "") << '\n';
                    index_free<utf8_>(index);
                    szs_device_scope_free(device);
                    return 6;
                }
                std::cout << "k=" << unsigned(bound) << " mode=steady_growable repeat=" << repeat
                          << " query=" << elapsed << "s matches=" << found << " retried=" << retried
                          << " threads=" << query_threads << '\n';
            }

            if (mode_enabled(modes, "latency")) {
                std::vector<double> latencies;
                latencies.reserve(queries.size());
                sparse_matches_t service_matches;
                service_matches.resize(8);
                std::size_t service_capacity = 8;
                sz_size_t service_matches_count = 0;
                for (std::size_t query_index = 0; query_index != queries.size(); ++query_index) {
                    sz_string_view_t one_view = query_views[query_index];
                    sz_sequence_t one_query;
                    sz_sequence_from_string_views(&one_view, 1, &one_query);
                    auto const start = std::chrono::steady_clock::now();
                    sz_size_t found = 0;
                    sz_status_t status = index_find<utf8_>(index, device, &one_query, bound, service_matches, &found,
                                                            &error);
                    if (status == sz_unexpected_dimensions_k && found > service_capacity) {
                        service_capacity = found;
                        service_matches.resize(service_capacity);
                        status = index_find<utf8_>(index, device, &one_query, bound, service_matches, &found, &error);
                    }
                    double const elapsed =
                        std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
                    if (status != sz_success_k) {
                        std::cerr << "latency search failed: " << int(status) << " " << (error ? error : "") << '\n';
                        index_free<utf8_>(index);
                        szs_device_scope_free(device);
                        return 6;
                    }
                    latencies.push_back(elapsed);
                    service_matches_count += found;
                }
                if (service_matches_count != required) {
                    std::cerr << "latency search returned a different result count\n";
                    index_free<utf8_>(index);
                    szs_device_scope_free(device);
                    return 6;
                }
                std::cout << "k=" << unsigned(bound) << " mode=single_query_latency samples=" << latencies.size()
                          << " p50=" << percentile(latencies, 0.50) << "s p95=" << percentile(latencies, 0.95)
                          << "s p99=" << percentile(latencies, 0.99) << "s matches=" << service_matches_count
                          << " final_capacity=" << service_capacity << '\n';
            }
            if (!dump_prefix.empty()) {
                std::string const path = dump_prefix + ".k" + std::to_string(bound) + ".bin";
                if (!dump_matches(dictionary.size(), queries.size(), bound, matches, path)) {
                    std::cerr << "dump failed: " << path << '\n';
                    index_free<utf8_>(index);
                    szs_device_scope_free(device);
                    return 7;
                }
            }
        }
        index_free<utf8_>(index);
    }
    szs_device_scope_free(device);
    if (cache_evict_bytes) std::cerr << "cache_evict_checksum=" << cache_evict_checksum << '\n';
    return 0;
}

int main(int argc, char **argv) {
    if (argc < 3 || argc > 5) {
        std::cerr << "usage: levenshtein_index DICTIONARY QUERIES [QUERY_LIMIT] [DUMP_PREFIX]\n";
        return 2;
    }
    std::size_t const query_limit = argc >= 4 ? std::stoull(argv[3]) : 0;
    std::string const dump_prefix = argc == 5 ? argv[4] : "";
    auto const dictionary = load_lines(argv[1]);
    auto const queries = load_lines(argv[2], query_limit);
    bool const utf8 = std::getenv("SZ_LEVENSHTEIN_UTF8") != nullptr;
    std::string const modes = std::getenv("SZ_LEVENSHTEIN_MODES") ? std::getenv("SZ_LEVENSHTEIN_MODES")
                                                                   : "warm,steady,latency";
    bool const shared_index = std::getenv("SZ_LEVENSHTEIN_INDEX_PLAN") &&
                              std::string_view(std::getenv("SZ_LEVENSHTEIN_INDEX_PLAN")) == "shared";
    std::cout << "dictionary=" << dictionary.size() << " queries=" << queries.size()
              << " semantics=" << (utf8 ? "utf8-codepoints" : "bytes") << '\n';

    std::vector<std::uint8_t> max_distances = {1, 2, 4};
    int const query_repeats = std::getenv("SZ_LEVENSHTEIN_REPEATS")
                                  ? std::stoi(std::getenv("SZ_LEVENSHTEIN_REPEATS"))
                                  : 3;
    std::size_t const query_threads = std::getenv("SZ_LEVENSHTEIN_THREADS")
                                          ? std::stoull(std::getenv("SZ_LEVENSHTEIN_THREADS"))
                                          : 1;
    std::size_t const batches_per_repeat = std::getenv("SZ_LEVENSHTEIN_BATCHES_PER_REPEAT")
                                                ? std::stoull(std::getenv("SZ_LEVENSHTEIN_BATCHES_PER_REPEAT"))
                                                : 1;
    std::size_t const cache_evict_mb = std::getenv("SZ_LEVENSHTEIN_CACHE_EVICT_MB")
                                           ? std::stoull(std::getenv("SZ_LEVENSHTEIN_CACHE_EVICT_MB"))
                                           : 0;
    if (query_repeats <= 0 || query_threads == 0 || batches_per_repeat == 0 ||
        cache_evict_mb > std::size_t(-1) / (1024 * 1024)) {
        std::cerr << "repeat, thread, batch, or cache setting is invalid\n";
        return 2;
    }
    if (char const *requested_max = std::getenv("SZ_LEVENSHTEIN_MAX_DISTANCE")) {
        int const parsed = std::stoi(requested_max);
        if (parsed < 1 || parsed >= std::numeric_limits<std::uint8_t>::max()) {
            std::cerr << "SZ_LEVENSHTEIN_MAX_DISTANCE must be between 1 and 254\n";
            return 2;
        }
        max_distances = {static_cast<std::uint8_t>(parsed)};
    }
    if (shared_index && !std::getenv("SZ_LEVENSHTEIN_MAX_DISTANCE")) max_distances = {4};
    if (!mode_enabled(modes, "cold") && !mode_enabled(modes, "warm") && !mode_enabled(modes, "steady") &&
        !mode_enabled(modes, "latency")) {
        std::cerr << "SZ_LEVENSHTEIN_MODES must include cold, warm, steady, or latency\n";
        return 2;
    }
    std::size_t const cache_evict_bytes = cache_evict_mb * 1024 * 1024;
    return utf8 ? run<true>(dictionary, queries, max_distances, dump_prefix, query_repeats, query_threads,
                            batches_per_repeat, cache_evict_bytes, modes, shared_index)
                : run<false>(dictionary, queries, max_distances, dump_prefix, query_repeats, query_threads,
                             batches_per_repeat, cache_evict_bytes, modes, shared_index);
}
