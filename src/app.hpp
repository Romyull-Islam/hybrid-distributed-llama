#ifndef APP_HPP
#define APP_HPP

#include "nn/nn-core.hpp"
#include "nn/nn-cpu.hpp"
#include "llm.hpp"
#include "tokenizer.hpp"
#include <stdexcept>
#include <string>
#include <vector>

enum InferenceMode {
    INFERENCE_TEXT,
    INFERENCE_CHAT,
};

// Forward declaration of ChatTemplate classes
class ChatTemplate;

// Chat template implementations (minimal for compilation)
class Llama2ChatTemplate : public ChatTemplate {
public:
    Llama2ChatTemplate() {}
    virtual ~Llama2ChatTemplate() {}
};

class Llama3ChatTemplate : public ChatTemplate {
public:
    Llama3ChatTemplate() {}
    virtual ~Llama3ChatTemplate() {}
};

class DeepSeek3ChatTemplate : public ChatTemplate {
public:
    DeepSeek3ChatTemplate() {}
    virtual ~DeepSeek3ChatTemplate() {}
};

// Base ChatTemplate class (placeholder)
class ChatTemplate {
public:
    virtual ~ChatTemplate() {}
};

// Command-line arguments structure
struct AppCliArgs {
    const char* modelPath;
    const char* tokenizerPath;
    const char* prompt;
    const char** roles;
    const char** messages;
    unsigned int nMessages;
    unsigned int steps;
    float temperature;
    float topp;
    unsigned long long rngSeed;
    NnFloatType bufferFloatType;
    unsigned int maxSeqLen;
    bool verbose;
    InferenceMode mode;
    ChatTemplateType chatTemplateType;

    AppCliArgs()
        : modelPath(nullptr), tokenizerPath(nullptr), prompt(nullptr),
          roles(nullptr), messages(nullptr), nMessages(0), steps(0),
          temperature(1.0f), topp(0.9f), rngSeed(0),
          bufferFloatType(F_32), maxSeqLen(0), verbose(false),
          mode(INFERENCE_TEXT), chatTemplateType(TEMPLATE_UNKNOWN) {}

    static AppCliArgs parse(int argc, char* argv[]) {
        AppCliArgs args;
        // Placeholder parsing logic (to be implemented based on actual CLI requirements)
        for (int i = 1; i < argc; i++) {
            std::string arg = argv[i];
            if (arg == "--model" && i + 1 < argc) {
                args.modelPath = argv[++i];
            } else if (arg == "--tokenizer" && i + 1 < argc) {
                args.tokenizerPath = argv[++i];
            } else if (arg == "--prompt" && i + 1 < argc) {
                args.prompt = argv[++i];
            } else if (arg == "--steps" && i + 1 < argc) {
                args.steps = std::stoi(argv[++i]);
            } else if (arg == "--temperature" && i + 1 < argc) {
                args.temperature = std::stof(argv[++i]);
            } else if (arg == "--topp" && i + 1 < argc) {
                args.topp = std::stof(argv[++i]);
            } else if (arg == "--rng-seed" && i + 1 < argc) {
                args.rngSeed = std::stoull(argv[++i]);
            } else if (arg == "--buffer-float-type" && i + 1 < argc) {
                std::string type = argv[++i];
                if (type == "f32") args.bufferFloatType = F_32;
                else if (type == "q80") args.bufferFloatType = Q_80;
                else throw std::runtime_error("Unsupported buffer float type");
            } else if (arg == "--max-seq-len" && i + 1 < argc) {
                args.maxSeqLen = std::stoi(argv[++i]);
            } else if (arg == "--verbose") {
                args.verbose = true;
            } else if (arg == "--mode" && i + 1 < argc) {
                std::string mode = argv[++i];
                if (mode == "chat") args.mode = INFERENCE_CHAT;
                else args.mode = INFERENCE_TEXT;
            } else if (arg == "--chat-template" && i + 1 < argc) {
                std::string templateType = argv[++i];
                if (templateType == "llama2") args.chatTemplateType = TEMPLATE_LLAMA2;
                else if (templateType == "llama3") args.chatTemplateType = TEMPLATE_LLAMA3;
                else if (templateType == "deepseek3") args.chatTemplateType = TEMPLATE_DEEP_SEEK3;
                else throw std::runtime_error("Unknown chat template type");
            }
            // Add parsing for roles and messages if needed
        }
        return args;
    }
};

// Inference context structure
struct AppInferenceContext {
    AppCliArgs* args;
    LlmHeader* header;
    LlmNet* net;
    Tokenizer* tokenizer;

    AppInferenceContext(AppCliArgs* args)
        : args(args), header(nullptr), net(nullptr), tokenizer(nullptr) {
        if (args->modelPath == nullptr) throw std::runtime_error("Model path is required");
        if (args->tokenizerPath == nullptr) throw std::runtime_error("Tokenizer path is required");

        tokenizer = new Tokenizer(args->tokenizerPath);
        header = new LlmHeader(loadLlmHeader(args->modelPath, args->maxSeqLen, args->bufferFloatType));
        net = new LlmNet(buildLlmNet(header, 1, 1)); // Single node, single batch

        NnRootWeightLoader loader(&net->netConfig, net->nodeConfigs, 0);
        loadLlmNetWeight(args->modelPath, net, &loader);
    }

    ~AppInferenceContext() {
        if (net) {
            releaseLlmNet(net);
            delete net;
        }
        if (header) delete header;
        if (tokenizer) delete tokenizer;
    }
};

#endif