#include <string>

/**
 * @brief A C++ representation of a single chat message.
 *
 * This struct mirrors the C-style `llama_chat_message` but uses `std::string`
 * to ensure safe, automatic memory management of the role and content text.
 */
struct ChatMessage {
    std::string role;
    std::string content;
};
