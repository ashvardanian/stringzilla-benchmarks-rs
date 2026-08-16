/**
 *  @brief Deterministic query generator for immutable-dictionary Levenshtein benchmarks.
 *
 *  Generates exact, one-edit, two-edit, five-symbol length-extended, or mixed queries by sampling lines from an
 *  existing dictionary. The extension is guaranteed beyond four edits from its sampled source, not necessarily from
 *  every other dictionary entry. Mutations use bytes by default or validated Unicode codepoints when
 *  `SZ_LEVENSHTEIN_UTF8` is set.
 */
#include <stringzilla/utf8_runes/serial.h>

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

static std::size_t substitute(std::string &query, std::uint64_t random) {
    if (query.empty()) {
        query.push_back('~');
        return 0;
    }
    std::size_t const position = random % query.size();
    query[position] = query[position] == '~' ? '^' : '~';
    return position;
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

static std::size_t substitute(std::vector<sz_rune_t> &query, std::uint64_t random) {
    if (query.empty()) {
        query.push_back('~');
        return 0;
    }
    std::size_t const position = random % query.size();
    query[position] = query[position] == '~' ? '^' : '~';
    return position;
}

int main(int argc, char **argv) {
    if (argc != 6) {
        std::cerr << "usage: levenshtein_index_queries DICTIONARY OUTPUT COUNT exact|edit1|edit2|reject|mixed SEED\n";
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
    std::uint64_t random_state = std::stoull(argv[5]);
    bool const utf8 = std::getenv("SZ_LEVENSHTEIN_UTF8") != nullptr;
    for (std::size_t query_index = 0; query_index != count; ++query_index) {
        std::uint64_t const sample_random = splitmix64(random_state);
        std::string query = dictionary[sample_random % dictionary.size()];
        std::string_view mode = requested_mode;
        if (mode == "mixed") {
            static constexpr std::string_view modes[] = {"exact", "edit1", "edit2", "reject"};
            mode = modes[query_index % 4];
        }
        if (utf8) {
            std::vector<sz_rune_t> decoded;
            if (!decode_utf8(query, decoded)) {
                std::cerr << "invalid UTF-8 dictionary entry\n";
                return 6;
            }
            if (mode == "edit1" || mode == "edit2") {
                std::size_t const first_position = substitute(decoded, splitmix64(random_state));
                if (mode == "edit2") {
                    if (decoded.size() > 1) {
                        std::size_t second_position = splitmix64(random_state) % (decoded.size() - 1);
                        if (second_position >= first_position) ++second_position;
                        decoded[second_position] = decoded[second_position] == '~' ? '^' : '~';
                    }
                    else
                        decoded.push_back('~');
                }
            }
            if (mode == "reject") decoded.insert(decoded.begin(), 5, '~');
            query = encode_utf8(decoded);
        }
        else {
            if (mode == "edit1" || mode == "edit2") {
                std::size_t const first_position = substitute(query, splitmix64(random_state));
                if (mode == "edit2") {
                    if (query.size() > 1) {
                        std::size_t second_position = splitmix64(random_state) % (query.size() - 1);
                        if (second_position >= first_position) ++second_position;
                        query[second_position] = query[second_position] == '~' ? '^' : '~';
                    }
                    else
                        query.push_back('~');
                }
            }
            if (mode == "reject") query.insert(0, "~~~~~"); // Beyond four edits from the sampled source word.
        }
        if (mode != "exact" && mode != "edit1" && mode != "edit2" && mode != "reject") {
            std::cerr << "unknown mode: " << mode << '\n';
            return 5;
        }
        output << query << '\n';
    }
    return output.good() ? 0 : 6;
}
