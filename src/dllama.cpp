#include "nn/nn-core.hpp" #include "nn/nn-config-builder.hpp" #include "nn/nn-cpu.hpp" #include "nn/nn-network.hpp" #include "nn/nn-executor.hpp" #include "llm.hpp" #include "tokenizer.hpp" #include "app.hpp" #include

static void inference(AppInferenceContext \*context) { if (context-&gt;args-&gt;prompt == nullptr) throw std::runtime_error("Prompt is required"); if (context-&gt;args-&gt;steps == 0) throw std::runtime_error("Number of steps is required");

```
std::vector<int> inputTokensVec(std::strlen(context->args->prompt));
NnUint inputTokens = context->tokenizer->encode(context->args->prompt, &inputTokensVec[0]);
inputTokensVec.resize(inputTokens);

context->inference->setBatchSize(1);
for (NnUint i = 0; i < inputTokens; i++) {
    context->inference->setPosition(i);
    context->inference->setToken(0, inputTokensVec[i]);
    context->inference->forward();
}

printf("🟩 Input processed\n");

for (NnUint i = 0; i < context->args->steps; i++) {
    context->inference->setPosition(inputTokens + i);
    context->inference->setToken(0, context->sampler->sample(context->inference->logitsPipe));
    context->inference->forward();

    int token = (int)context->inference->tokenPipe[0];
    char *text = context->tokenizer->decode(token);
    printf("%s", text);
    fflush(stdout);
}
printf("\n");

if (context->network != NULL) {
    NnSize sentBytes, recvBytes;
    context->network->getStats(&sentBytes, &recvBytes);
    printf("📈 Network: sent %.2f MB, received %.2f MB\n", sentBytes / (1024.0 * 1024.0), recvBytes / (1024.0 * 1024.0));
}
```

}

static void chatInference(AppInferenceContext \*context) { ChatTemplate \*chatTemplate = nullptr; if (context-&gt;args-&gt;chatTemplateType == TEMPLATE_LLAMA2) chatTemplate = new Llama2ChatTemplate(); else if (context-&gt;args-&gt;chatTemplateType == TEMPLATE_LLAMA3) chatTemplate = new Llama3ChatTemplate(); else if (context-&gt;args-&gt;chatTemplateType == TEMPLATE_DEEP_SEEK3) chatTemplate = new DeepSeek3ChatTemplate(); else throw std::runtime_error("Unknown chat template");

```
std::string systemPrompt;
std::vector<std::string> userPrompts;
std::vector<std::string> assistantReplies;

if (context->args->prompt != nullptr) {
    userPrompts.push_back(context->args->prompt);
}

printf("🟩 Welcome to Distributed Llama Chat\n");
printf("Type '/exit' to quit or '/clear' to clear the conversation\n\n");

context->inference->setBatchSize(1);
while (true) {
    printf("👨‍💻 You: ");
    std::string input;
    std::getline(std::cin, input);
    if (input == "/exit") {
        break;
    } else if (input == "/clear") {
        userPrompts.clear();
        assistantReplies.clear();
        printf("🟩 Conversation cleared\n");
        continue;
    }

    userPrompts.push_back(input);
    std::string fullPrompt = chatTemplate->buildPrompt(systemPrompt, userPrompts, assistantReplies);
    std::vector<int> inputTokensVec(fullPrompt.size());
    NnUint inputTokens = context->tokenizer->encode(fullPrompt.c_str(), &inputTokensVec[0]);
    inputTokensVec.resize(inputTokens);

    for (NnUint i = 0; i < inputTokens; i++) {
        context->inference->setPosition(i);
        context->inference->setToken(0, inputTokensVec[i]);
        context->inference->forward();
    }

    printf("🤖 Assistant: ");
    std::string reply;
    for (NnUint i = 0; i < context->args->steps; i++) {
        context->inference->setPosition(inputTokens + i);
        context->inference->setToken(0, context->sampler->sample(context->inference->logitsPipe));
        context->inference->forward();

        int token = (int)context->inference->tokenPipe[0];
        char *text = context->tokenizer->decode(token);
        printf("%s", text);
        reply += text;
        fflush(stdout);
    }
    printf("\n");
    assistantReplies.push_back(reply);

    if (context->network != NULL) {
        NnSize sentBytes, recvBytes;
        context->network->getStats(&sentBytes, &recvBytes);
        printf("📈 Network: sent %.2f MB, received %.2f MB\n", sentBytes / (1024.0 * 1024.0), recvBytes / (1024.0 * 1024.0));
    }
}

delete chatTemplate;
```

}

int main(int argc, char\* argv\[\]) { AppCliArgs args = AppCliArgs::parse(argc,