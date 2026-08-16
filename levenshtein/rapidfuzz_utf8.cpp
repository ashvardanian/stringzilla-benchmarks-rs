/**
 *  @brief Native RapidFuzz baseline for validated UTF-8/codepoint immutable-dictionary retrieval.
 *
 *  This is intentionally outside the default build because it requires rapidfuzz-cpp. UTF-8 decoding happens while
 *  loading the corpus, outside the timed query loop, favoring RapidFuzz relative to StringZilla's per-query facade.
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

static bool decode_utf8(std::string const &source, std::u32string &destination) {
    destination.clear();
    for (std::size_t i = 0; i != source.size();) {
        std::uint8_t const lead = static_cast<std::uint8_t>(source[i]);
        char32_t rune = 0;
        std::size_t length = 0;
        if (lead < 0x80) rune = lead, length = 1;
        else if (lead >= 0xC2 && lead <= 0xDF && i + 1 < source.size())
            rune = (char32_t(lead & 0x1F) << 6) | (static_cast<std::uint8_t>(source[i + 1]) & 0x3F), length = 2;
        else if (lead >= 0xE0 && lead <= 0xEF && i + 2 < source.size())
            rune = (char32_t(lead & 0x0F) << 12) | ((static_cast<std::uint8_t>(source[i + 1]) & 0x3F) << 6) |
                   (static_cast<std::uint8_t>(source[i + 2]) & 0x3F),
            length = 3;
        else if (lead >= 0xF0 && lead <= 0xF4 && i + 3 < source.size())
            rune = (char32_t(lead & 0x07) << 18) | ((static_cast<std::uint8_t>(source[i + 1]) & 0x3F) << 12) |
                   ((static_cast<std::uint8_t>(source[i + 2]) & 0x3F) << 6) |
                   (static_cast<std::uint8_t>(source[i + 3]) & 0x3F),
            length = 4;
        else
            return false;
        for (std::size_t continuation = 1; continuation != length; ++continuation)
            if ((static_cast<std::uint8_t>(source[i + continuation]) & 0xC0) != 0x80) return false;
        if ((length == 3 && ((lead == 0xE0 && static_cast<std::uint8_t>(source[i + 1]) < 0xA0) ||
                             (lead == 0xED && static_cast<std::uint8_t>(source[i + 1]) >= 0xA0))) ||
            (length == 4 && ((lead == 0xF0 && static_cast<std::uint8_t>(source[i + 1]) < 0x90) ||
                             (lead == 0xF4 && static_cast<std::uint8_t>(source[i + 1]) >= 0x90))))
            return false;
        destination.push_back(rune);
        i += length;
    }
    return true;
}

static std::vector<std::u32string> load_lines(char const *path, std::size_t limit = 0) {
    std::ifstream input(path);
    if (!input) throw std::runtime_error(std::string("failed to open ") + path);
    std::vector<std::u32string> lines;
    std::string encoded;
    while ((!limit || lines.size() != limit) && std::getline(input, encoded)) {
        if (!encoded.empty() && encoded.back() == '\r') encoded.pop_back();
        std::u32string decoded;
        if (!decode_utf8(encoded, decoded)) throw std::runtime_error(std::string("invalid UTF-8 in ") + path);
        lines.push_back(std::move(decoded));
    }
    return lines;
}

static bool dump_matches(std::vector<std::u32string> const &dictionary,
                         std::vector<std::u32string> const &queries, std::uint8_t bound, std::string const &path) {
    std::ofstream output(path, std::ios::binary);
    if (!output) return false;
    char const magic[8] = {'S', 'Z', 'L', 'E', 'V', '0', '0', '1'};
    std::uint64_t const dictionary_size = dictionary.size(), queries_size = queries.size();
    output.write(magic, sizeof(magic));
    output.write(reinterpret_cast<char const *>(&dictionary_size), sizeof(dictionary_size));
    output.write(reinterpret_cast<char const *>(&queries_size), sizeof(queries_size));
    output.write(reinterpret_cast<char const *>(&bound), sizeof(bound));

    std::vector<match_t> matches;
    for (auto const &query : queries) {
        matches.clear();
        rapidfuzz::CachedLevenshtein<char32_t> scorer(query);
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
        std::cerr << "usage: rapidfuzz_levenshtein_index_utf8 DICTIONARY QUERIES [QUERY_LIMIT] [DUMP_PREFIX]\n";
        return 2;
    }
    std::size_t const query_limit = argc >= 4 ? std::stoull(argv[3]) : 0;
    std::string const dump_prefix = argc == 5 ? argv[4] : "";
    auto const dictionary = load_lines(argv[1]);
    auto const queries = load_lines(argv[2], query_limit);
    int const repeats = std::getenv("RF_REPEATS") ? std::stoi(std::getenv("RF_REPEATS")) : 3;
    int const max_distance = std::getenv("RF_MAX_DISTANCE") ? std::stoi(std::getenv("RF_MAX_DISTANCE")) : 4;
    if (max_distance < 1 || max_distance > 4) return 2;
    std::cout << "dictionary=" << dictionary.size() << " queries=" << queries.size()
              << " semantics=utf8-codepoints\n";

    for (std::uint8_t bound = 1; bound <= max_distance; ++bound) {
        for (int repeat = 0; repeat != repeats; ++repeat) {
            std::size_t matches_count = 0;
            auto const start = std::chrono::steady_clock::now();
            for (auto const &query : queries) {
                rapidfuzz::CachedLevenshtein<char32_t> scorer(query);
                for (auto const &candidate : dictionary)
                    if (scorer.distance(candidate, bound) <= bound) ++matches_count;
            }
            double const elapsed =
                std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
            std::cout << "k=" << unsigned(bound) << " query=" << elapsed << "s matches=" << matches_count << '\n';
        }
        if (!dump_prefix.empty() && !dump_matches(dictionary, queries, bound,
                                                  dump_prefix + ".k" + std::to_string(bound) + ".bin"))
            return 3;
    }
}
