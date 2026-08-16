/**
 *  @brief Native RapidFuzz baseline for repeated immutable-dictionary Levenshtein retrieval.
 *
 *  This file is intentionally not part of the default build because it requires rapidfuzz-cpp.
 *  Pin the dependency revision and compile it explicitly, for example:
 *
 *      g++ -std=c++20 -O3 -DNDEBUG -march=native -I rapidfuzz-cpp \
 *          levenshtein/rapidfuzz.cpp -o rapidfuzz_levenshtein_index
 *
 *  `RF_REPEATS=0` skips timing and only emits exact comparison artifacts when `DUMP_PREFIX` is present.
 */
#include <rapidfuzz/distance/Levenshtein.hpp>

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

struct match_t {
    std::uint32_t id;
    std::uint8_t distance;
};

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

static bool dump_matches(std::vector<std::string> const &dictionary, std::vector<std::string> const &queries,
                         std::uint8_t bound, std::string const &path) {
    std::ofstream output(path, std::ios::binary);
    if (!output) return false;
    char const magic[8] = {'S', 'Z', 'L', 'E', 'V', '0', '0', '1'};
    std::uint64_t const dictionary_size = dictionary.size();
    std::uint64_t const queries_size = queries.size();
    output.write(magic, sizeof(magic));
    output.write(reinterpret_cast<char const *>(&dictionary_size), sizeof(dictionary_size));
    output.write(reinterpret_cast<char const *>(&queries_size), sizeof(queries_size));
    output.write(reinterpret_cast<char const *>(&bound), sizeof(bound));

    std::vector<match_t> matches;
    for (auto const &query : queries) {
        matches.clear();
        rapidfuzz::CachedLevenshtein<char> scorer(query);
        for (std::uint32_t id = 0; id != dictionary.size(); ++id) {
            std::size_t const distance = scorer.distance(dictionary[id], bound);
            if (distance <= bound) matches.push_back({id, static_cast<std::uint8_t>(distance)});
        }
        std::uint64_t const matches_size = matches.size();
        output.write(reinterpret_cast<char const *>(&matches_size), sizeof(matches_size));
        for (auto const &match : matches) {
            output.write(reinterpret_cast<char const *>(&match.id), sizeof(match.id));
            output.write(reinterpret_cast<char const *>(&match.distance), sizeof(match.distance));
        }
    }
    return output.good();
}

int main(int argc, char **argv) {
    if (argc < 3 || argc > 5) {
        std::cerr << "usage: rapidfuzz_levenshtein_index DICTIONARY QUERIES [QUERY_LIMIT] [DUMP_PREFIX]\n";
        return 2;
    }
    std::size_t const query_limit = argc >= 4 ? std::stoull(argv[3]) : 0;
    std::string const dump_prefix = argc == 5 ? argv[4] : "";
    auto const dictionary = load_lines(argv[1]);
    auto const queries = load_lines(argv[2], query_limit);
    int const repeats = std::getenv("RF_REPEATS") ? std::stoi(std::getenv("RF_REPEATS")) : 3;
    int const max_distance = std::getenv("RF_MAX_DISTANCE") ? std::stoi(std::getenv("RF_MAX_DISTANCE")) : 4;
    if (max_distance < 1 || max_distance > 4) {
        std::cerr << "RF_MAX_DISTANCE must be between 1 and 4\n";
        return 2;
    }
    std::cout << "dictionary=" << dictionary.size() << " queries=" << queries.size() << '\n';

    for (std::uint8_t bound = 1; bound <= max_distance; ++bound) {
        for (int repeat = 0; repeat != repeats; ++repeat) {
            std::size_t matches_count = 0;
            auto const start = std::chrono::steady_clock::now();
            for (auto const &query : queries) {
                rapidfuzz::CachedLevenshtein<char> scorer(query);
                for (auto const &candidate : dictionary)
                    if (scorer.distance(candidate, bound) <= bound) ++matches_count;
            }
            double const elapsed =
                std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
            std::cout << "k=" << unsigned(bound) << " query=" << elapsed << "s matches=" << matches_count << '\n';
        }
        if (!dump_prefix.empty()) {
            std::string const path = dump_prefix + ".k" + std::to_string(bound) + ".bin";
            if (!dump_matches(dictionary, queries, bound, path)) {
                std::cerr << "dump failed: " << path << '\n';
                return 3;
            }
        }
    }
}
