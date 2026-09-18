#include "jinja/string.h"
#include "jinja/value.h"
#include "unicode.h"

#include <algorithm>
#include <functional>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

namespace jinja {

//
// string_part
//

bool string_part::is_uppercase() const {
    for (char c : val) {
        if (std::islower(static_cast<unsigned char>(c))) {
            return false;
        }
    }
    return true;
}

bool string_part::is_lowercase() const {
    for (char c : val) {
        if (std::isupper(static_cast<unsigned char>(c))) {
            return false;
        }
    }
    return true;
}

//
// string
//

void string::mark_input() {
    for (auto & part : parts) {
        part.is_input = true;
    }
}

std::string string::str() const {
    if (parts.size() == 1) {
        return parts[0].val;
    }
    std::ostringstream oss;
    for (const auto & part : parts) {
        oss << part.val;
    }
    return oss.str();
}

size_t string::length() const {
    size_t len = 0;
    for (const auto & part : parts) {
        len += part.val.length();
    }
    return len;
}

void string::hash_update(hasher & hash) const noexcept {
    for (const auto & part : parts) {
        hash.update(part.val.data(), part.val.length());
    }
}

bool string::all_parts_are_input() const {
    for (const auto & part : parts) {
        if (!part.is_input) {
            return false;
        }
    }
    return true;
}

bool string::is_uppercase() const {
    for (const auto & part : parts) {
        if (!part.is_uppercase()) {
            return false;
        }
    }
    return true;
}

bool string::is_lowercase() const {
    for (const auto & part : parts) {
        if (!part.is_lowercase()) {
            return false;
        }
    }
    return true;
}

// mark this string as input if other has ALL parts as input
void string::mark_input_based_on(const string & other) {
    if (other.all_parts_are_input()) {
        for (auto & part : parts) {
            part.is_input = true;
        }
    }
}

string & string::append(const string & other) {
    for (const auto & part : other.parts) {
        parts.push_back(part);
    }
    return *this;
}

// in-place transformation

using transform_fn = std::function<std::string(const std::string&)>;
static string apply_transform(string & self, const transform_fn & fn) {
    for (auto & part : self.parts) {
        part.val = fn(part.val);
    }
    return self;
}

string string::uppercase() {
    return apply_transform(*this, [](const std::string & s) {
        std::string res = s;
        std::transform(res.begin(), res.end(), res.begin(), ::toupper);
        return res;
    });
}
string string::lowercase() {
    return apply_transform(*this, [](const std::string & s) {
        std::string res = s;
        std::transform(res.begin(), res.end(), res.begin(), ::tolower);
        return res;
    });
}
string string::capitalize() {
    return apply_transform(*this, [](const std::string & s) {
        if (s.empty()) return s;
        std::string res = s;
        res[0] = ::toupper(static_cast<unsigned char>(res[0]));
        std::transform(res.begin() + 1, res.end(), res.begin() + 1, ::tolower);
        return res;
    });
}
string string::titlecase() {
    return apply_transform(*this, [](const std::string & s) {
        std::string res = s;
        bool capitalize_next = true;
        for (char &c : res) {
            if (isspace(static_cast<unsigned char>(c))) {
                capitalize_next = true;
            } else if (capitalize_next) {
                c = ::toupper(static_cast<unsigned char>(c));
                capitalize_next = false;
            } else {
                c = ::tolower(static_cast<unsigned char>(c));
            }
        }
        return res;
    });
}
string string::strip(bool left, bool right, std::optional<const std::string_view> chars) {
    static auto strip_part = [](const std::string & s, bool left, bool right, std::optional<const std::string_view> chars) -> std::string {
        std::vector<uint32_t> strip_chars;
        if (chars) {
            for (size_t i = 0; i < chars->size();) {
                auto cp = common_parse_utf8_codepoint(*chars, i);
                strip_chars.push_back(cp.status == utf8_parse_result::SUCCESS ? cp.codepoint : uint8_t((*chars)[i]));
                i += cp.status == utf8_parse_result::SUCCESS ? cp.bytes_consumed : 1;
            }
        }
        auto match_char = [&](uint32_t c) -> bool {
            if (chars) {
                return std::find(strip_chars.begin(), strip_chars.end(), c) != strip_chars.end();
            }
            // Python str.strip() whitespace, independent of the process locale.
            return (c >= 0x09 && c <= 0x0d) || (c >= 0x1c && c <= 0x20) || c == 0x85 ||
                   c == 0xa0 || c == 0x1680 || (c >= 0x2000 && c <= 0x200a) ||
                   c == 0x2028 || c == 0x2029 || c == 0x202f || c == 0x205f || c == 0x3000;
        };
        std::vector<std::pair<size_t, bool>> points;
        for (size_t i = 0; i < s.size();) {
            auto cp = common_parse_utf8_codepoint(s, i);
            points.emplace_back(i, cp.status == utf8_parse_result::SUCCESS && match_char(cp.codepoint));
            i += cp.status == utf8_parse_result::SUCCESS ? cp.bytes_consumed : 1;
        }
        size_t first = 0, last = points.size();
        if (left) { while (first < last && points[first].second) { ++first; } }
        if (right) { while (last > first && points[last - 1].second) { --last; } }
        const size_t start = first < points.size() ? points[first].first : s.size();
        const size_t end = last < points.size() ? points[last].first : s.size();
        return s.substr(start, end - start);
    };
    if (parts.empty()) {
        return *this;
    }
    if (left) {
        for (size_t i = 0; i < parts.size(); ++i) {
            parts[i].val = strip_part(parts[i].val, true, false, chars);
            if (parts[i].val.empty()) {
                // remove empty part
                parts.erase(parts.begin() + i);
                --i;
                continue;
            } else {
                break;
            }
        }
    }
    if (right) {
        for (size_t i = parts.size(); i-- > 0;) {
            parts[i].val = strip_part(parts[i].val, false, true, chars);
            if (parts[i].val.empty()) {
                // remove empty part
                parts.erase(parts.begin() + i);
                continue;
            } else {
                break;
            }
        }
    }
    return *this;
}

} // namespace jinja
