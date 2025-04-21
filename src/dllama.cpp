#include "nn/nn-core.hpp"
#include "nn/nn-config-builder.hpp"
#include "nn/nn-cpu.hpp"
#include "nn/nn-network.hpp"
#include "nn/nn-executor.hpp"
#include "llm.hpp"
#include "tokenizer.hpp"
#include "app.hpp"

static void inference(AppInferenceContext *context) {
    if (context->args->prompt == nullptr) throw std::runtime_error("Prompt is required");
    if (context->args->steps == 0) throw std::runtime_error("Number of steps is required");

    NnExecutor executor(&context->net->netConfig, context->net->nodeConfigs);
    LlmHeader *header = context->net->header;
    Sampler sampler(header->vocabSize, context->args->temperature, context->args->topp, context->args->rngSeed);
    int *tokens = new int[header->seqLen];
    int nTokens;

    context->tokenizer->encode(const_cast<char*>(context->args->prompt), tokens, &nTokens, true, false);
    if (nTokens >= header->seqLen) throw std::runtime_error("Prompt is too long");

    int positionId = 0;
    int tokenId;

    if (context->args->verbose) {
        printf("🎤 Prompt: %s\n", context->args->prompt);
        printf("🎤 Tokens: ");
        for (int i = 0; i < nTokens; i++) printf("%d ", tokens[i]);
        printf("\n");
    }

    context->tokenizer->resetDecoder();
    for (int i = 0; i < nTokens; i++) {
        tokenId = tokens[i];
        executor.forward(context->net->tokenPipeIndex, &tokenId, context->net->positionPipeIndex, &positionId);
        positionId++;
        if (context->args->verbose) {
            char *piece = context->tokenizer->decode(tokenId);
            printf("%s", piece);
            fflush(stdout);
        }
    }

    float *logits = new float[header->vocabSize];
    for (int step = 0; step < context->args->steps; step++) {
        executor.forwardSync(context->net->logitsPipeIndex, logits);
        tokenId = sampler.sample(logits);
        if (context->tokenizer->isEos(tokenId)) break;

        executor.forward(context->net->tokenPipeIndex, &tokenId, context->net->positionPipeIndex, &positionId);
        positionId++;

        char *piece = context->tokenizer->decode(tokenId);
        printf("%s", piece);
        fflush(stdout);
    }
    printf("\n");

    delete[] tokens;
    delete[] logits;
}

static void chatInference(AppInferenceContext *context) {
    ChatTemplate *chatTemplate = nullptr;
    if (context->args->chatTemplateType == TEMPLATE_LLAMA2)
        chatTemplate = new Llama2ChatTemplate();
    else if (context->args->chatTemplateType == TEMPLATE_LLAMA3)
        chatTemplate = new Llama3ChatTemplate();
    else if (context->args->chatTemplateType == TEMPLATE_DEEP_SEEK3)
        chatTemplate = new DeepSeek3ChatTemplate();
    else
        throw std::runtime_error("Unknown chat template");

    std::vector<ChatItem> items;
    for (unsigned int i = 0; i < context->args->nMessages; i++) {
        items.push_back({std::string(context->args->roles[i]), std::string(context->args->messages[i])});
    }

    ChatTemplateGenerator generator(context->args->chatTemplateType, context->tokenizer->chatTemplate, nullptr);
    GeneratedChat prompt = generator.generate(items.size(), &items[0], true);

    if (context->args->verbose) {
        printf("🎤 Prompt: %s\n", prompt.content);
    }

    NnExecutor executor(&context->net->netConfig, context->net->nodeConfigs);
    LlmHeader *header = context->net->header;
    Sampler sampler(header->vocabSize, context->args->temperature, context->args->topp, context->args->rngSeed);
    int *tokens = new int[header->seqLen];
    int nTokens;

    context->tokenizer->encode(const_cast<char*>(prompt.content), tokens, &nTokens, true, true);
    if (nTokens >= header->seqLen) throw std::runtime_error("Prompt is too long");

    int positionId = 0;
    int tokenId;

    if (context->args->verbose) {
        printf("🎤 Tokens: ");
        for (int i = 0; i < nTokens; i++) printf("%d ", tokens[i]);
        printf("\n");
    }

    context->tokenizer->resetDecoder();
    for (int i = 0; i < nTokens; i++) {
        tokenId = tokens[i];
        executor.forward(context->net->tokenPipeIndex, &tokenId, context->net->positionPipeIndex, &positionId);
        positionId++;
        if (context->args->verbose) {
            char *piece = context->tokenizer->decode(tokenId);
            printf("%s", piece);
            fflush(stdout);
        }
    }

    float *logits = new float[header->vocabSize];
    TokenizerChatStops stops(context->tokenizer);
    EosDetector detector(stops.nStops, tokens, stops.stops, 0, 0);
    int steps = context->args->steps;

    for (int step = 0; step < steps; step++) {
        executor.forwardSync(context->net->logitsPipeIndex, logits);
        tokenId = sampler.sample(logits);
        char *piece = context->tokenizer->decode(tokenId);

        EosDetectorType eosType = detector.append(tokenId, piece);
        if (eosType == EOS) break;
        if (eosType == MAYBE_EOS) {
            steps++;
            continue;
        }

        executor.forward(context->net->tokenPipeIndex, &tokenId, context->net->positionPipeIndex, &positionId);
        positionId++;

        char *delta = detector.getDelta();
        printf("%s", delta);
        fflush(stdout);
    }
    printf("\n");

    delete chatTemplate;
    delete[] tokens;
    delete[] logits;
}

int main(int argc, char* argv[]) {
    AppCliArgs args = AppCliArgs::parse(argc, argv);
    AppInferenceContext context(&args);
    if (args.mode == INFERENCE_CHAT) chatInference(&context);
    else inference(&context);
    return 0;
}