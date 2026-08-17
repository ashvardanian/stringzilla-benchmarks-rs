/**
 *  @brief Deterministic query generator for immutable-dictionary Levenshtein benchmarks.
 *
 *  Samples dictionary entries and applies a named edit operation. The mixed workload gives equal weight to exact
 *  queries, substitutions, insertions, deletions, adjacent swaps, and several two-edit combinations. Labels describe
 *  how a query was made from its source word; the result oracle still decides which dictionary entries match.
 *  Mutations use bytes by default or validated Unicode codepoints when `SZ_LEVENSHTEIN_UTF8` is set.
 */
#include <stringzilla/utf8_runes/serial.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <string_view>
#include <vector>

static std::vector<std::string> load_lines(char const *path) {
    std::ifstream input(path);
    if (!input) throw std::runtime_error(std::string("failed to open ") + path);
    std::vector<std::string> lines;
    std::string line;
    while (std::getline(input, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        lines.push_back(std::move(line));
    }
    return lines;
}

static std::uint64_t splitmix64(std::uint64_t &state) {
    std::uint64_t value = (state += 0x9E3779B97F4A7C15ull);
    value = (value ^ (value >> 30)) * 0xBF58476D1CE4E5B9ull;
    value = (value ^ (value >> 27)) * 0x94D049BB133111EBull;
    return value ^ (value >> 31);
}

static bool decode_utf8(std::string const &encoded, std::vector<sz_rune_t> &decoded) {
    decoded.clear();
    char const *position = encoded.data(), *end = position + encoded.size();
    while (position != end) {
        sz_rune_t rune = 0;
        sz_rune_length_t const length = sz_rune_decode(position, end, &rune);
        if (length == sz_rune_invalid_k) return false;
        decoded.push_back(rune);
        position += length;
    }
    return true;
}

static std::string encode_utf8(std::vector<sz_rune_t> const &decoded) {
    std::string encoded;
    encoded.reserve(decoded.size() * 3);
    for (sz_rune_t rune : decoded) {
        sz_u8_t bytes[4];
        sz_rune_length_t const length = sz_rune_encode(rune, bytes);
        if (length == sz_rune_invalid_k) throw std::runtime_error("invalid Unicode codepoint");
        encoded.append(reinterpret_cast<char const *>(bytes), length);
    }
    return encoded;
}

template <typename sequence_type_>
static std::size_t substitute(sequence_type_ &query, std::uint64_t random) {
    std::size_t const position = random % query.size();
    query[position] = query[position] == '~' ? '^' : '~';
    return position;
}

static bool known_mode(std::string_view mode) {
    return mode == "exact" || mode == "substitute1" || mode == "insert1" || mode == "delete1" ||
           mode == "substitute2" || mode == "insert_delete" || mode == "insert2" || mode == "delete2" ||
           mode == "transpose" || mode == "source_plus_5" || mode == "mixed";
}

template <typename sequence_type_>
static bool supports_mode(sequence_type_ const &query, std::string_view mode) {
    if ((mode == "substitute1" || mode == "delete1" || mode == "insert_delete") && query.empty()) return false;
    if ((mode == "substitute2" || mode == "delete2") && query.size() < 2) return false;
    if (mode == "transpose") {
        for (std::size_t index = 1; index != query.size(); ++index)
            if (query[index - 1] != query[index]) return true;
        return false;
    }
    return true;
}

template <typename sequence_type_>
static void mutate(sequence_type_ &query, std::string_view mode, std::uint64_t &random_state) {
    using symbol_t = typename sequence_type_::value_type;
    auto const marker = [](symbol_t symbol) { return symbol == symbol_t('~') ? symbol_t('^') : symbol_t('~'); };
    auto const insert_one = [&] {
        std::size_t const position = splitmix64(random_state) % (query.size() + 1);
        symbol_t const inserted = position < query.size() ? marker(query[position]) : symbol_t('~');
        query.insert(query.begin() + position, inserted);
    };
    auto const delete_one = [&] {
        std::size_t const position = splitmix64(random_state) % query.size();
        query.erase(query.begin() + position);
    };

    if (mode == "exact") return;
    if (mode == "substitute1") {
        substitute(query, splitmix64(random_state));
        return;
    }
    if (mode == "insert1") {
        insert_one();
        return;
    }
    if (mode == "delete1") {
        delete_one();
        return;
    }
    if (mode == "substitute2") {
        std::size_t const first = substitute(query, splitmix64(random_state));
        std::size_t second = splitmix64(random_state) % (query.size() - 1);
        if (second >= first) ++second;
        query[second] = marker(query[second]);
        return;
    }
    if (mode == "insert_delete") {
        delete_one();
        insert_one();
        return;
    }
    if (mode == "insert2") {
        insert_one();
        insert_one();
        return;
    }
    if (mode == "delete2") {
        delete_one();
        delete_one();
        return;
    }
    if (mode == "transpose") {
        std::size_t position = splitmix64(random_state) % (query.size() - 1);
        while (query[position] == query[position + 1]) position = (position + 1) % (query.size() - 1);
        std::swap(query[position], query[position + 1]);
        return;
    }
    query.insert(query.begin(), 5, symbol_t('~'));
}

int main(int argc, char **argv) {
    if (argc != 6) {
        std::cerr << "usage: levenshtein_index_queries DICTIONARY OUTPUT COUNT MODE SEED\n";
        return 2;
    }
    auto const dictionary = load_lines(argv[1]);
    if (dictionary.empty()) {
        std::cerr << "dictionary is empty\n";
        return 3;
    }
    std::ofstream output(argv[2]);
    if (!output) {
        std::cerr << "failed to create " << argv[2] << '\n';
        return 4;
    }
    std::size_t const count = std::stoull(argv[3]);
    std::string_view const requested_mode = argv[4];
    if (!known_mode(requested_mode)) {
        std::cerr << "unknown mode: " << requested_mode << '\n';
        return 5;
    }
    std::uint64_t random_state = std::stoull(argv[5]);
    bool const utf8 = std::getenv("SZ_LEVENSHTEIN_UTF8") != nullptr;
    for (std::size_t query_index = 0; query_index != count; ++query_index) {
        std::string_view mode = requested_mode;
        if (mode == "mixed") {
            static constexpr std::string_view modes[] = {"exact",       "substitute1", "insert1",   "delete1",
                                                         "substitute2", "insert_delete", "insert2",   "delete2",
                                                         "transpose",   "source_plus_5"};
            mode = modes[query_index % (sizeof(modes) / sizeof(modes[0]))];
        }
        if (utf8) {
            std::vector<sz_rune_t> decoded;
            bool selected = false;
            for (std::size_t attempt = 0; attempt != dictionary.size(); ++attempt) {
                std::string const &source = dictionary[splitmix64(random_state) % dictionary.size()];
                if (!decode_utf8(source, decoded)) {
                    std::cerr << "invalid UTF-8 dictionary entry\n";
                    return 6;
                }
                if (supports_mode(decoded, mode)) {
                    selected = true;
                    break;
                }
            }
            if (!selected) {
                std::cerr << "dictionary has no entry suitable for mode " << mode << '\n';
                return 6;
            }
            mutate(decoded, mode, random_state);
            output << encode_utf8(decoded) << '\n';
        }
        else {
            std::string query;
            bool selected = false;
            for (std::size_t attempt = 0; attempt != dictionary.size(); ++attempt) {
                query = dictionary[splitmix64(random_state) % dictionary.size()];
                if (supports_mode(query, mode)) {
                    selected = true;
                    break;
                }
            }
            if (!selected) {
                std::cerr << "dictionary has no entry suitable for mode " << mode << '\n';
                return 6;
            }
            mutate(query, mode, random_state);
            output << query << '\n';
        }
    }
    return output.good() ? 0 : 6;
}
