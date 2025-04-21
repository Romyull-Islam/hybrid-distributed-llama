#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <fcntl.h>
#include <ctype.h>
#include <ctime>
#include <cassert>
#include <stdexcept>
#include <sstream>
#include <vector>
#include "nn/nn-core.hpp"
#include "nn/nn-cpu-ops.hpp"
#include "tokenizer.hpp"
#if defined(__ARM_NEON)
    #include <arm_neon.h>
#endif

#define DEBUG_TOKENIZER_ENCODER false
#define DEBUG_TOKENIZER_BENCHMARK false
#define DEBUG_TEMPLATE_GENERATOR false
#define DEBUG_SAMPLER_BENCHMARK false

#define TOK_VERSION 1000
#define TOK_VOCAB_SIZE 1001
#define MAX_TOKEN_LENGTH 1002
#define BOS_ID 1003
#define EOS_ID 1004
#define CHAT_EOS_ID 1005
#define CHAT_TEMPLATE 1006
#define CHAT_STOP 1007
#define PAD_ID 1008

int compareTokens(const void *a, const void *b) {
    return strcmp(((TokenIndex*)a)->str, ((TokenIndex*)b)->str);
}

Tokenizer::Tokenizer(const char* tokenizerPath)
    : eosTokenIds() {
    bosId = -1;
    chatTemplate = nullptr;
    maxTokenLength = 0;

    // read in the file
    FILE *file = fopen(tokenizerPath, "rb");
    if (!file)
        throw std::runtime_error("Failed to open tokenizer file");
    int magic;
    if (fread(&magic, sizeof(int), 1, file) != 1)
        throw std::runtime_error("Cannot read tokenizer magic number");

    if (magic == 0x567123) {
        TokenizerOldHeader header;
        if (fread(&header, sizeof(TokenizerOldHeader), 1, file) != 1)
            throw std::runtime_error("Cannot read tokenizer header");
        maxTokenLength = header.maxTokenLength;
        vocabSize = header.vocabSize;
        bosId = header.bosId;
        eosTokenIds.push_back(header.eosId);
    } else if (magic == 0x567124) {
        int headerSize;
        if (fread(&headerSize, sizeof(int), 1, file) != 1)
            throw std::runtime_error("Cannot read tokenizer header size");
        int nKv = (headerSize - 2 * sizeof(int)) / sizeof(int);
        std::vector<int> buffer(nKv);
        if (fread(&buffer[0], nKv * sizeof(int), 1, file) != 1) {
            throw std::runtime_error("Cannot read header values");
        }
        int version = -1;
        int chatTemplateLength = -1;
        for (int i = 0; i < nKv; i += 2) {
            int key = buffer[i];
            int value = buffer[i + 1];
            if (key == TOK_VERSION) version = value;
            else if (key == TOK_VOCAB_SIZE) vocabSize = value;
            else if (key == MAX_TOKEN_LENGTH) maxTokenLength = (unsigned int)value;
            else if (key == BOS_ID) bosId = value;
            else if (key == EOS_ID) eosTokenIds.push_back(value);
            else if (key == CHAT_EOS_ID) eosTokenIds.push_back(value);
            else if (key == CHAT_TEMPLATE) chatTemplateLength = value;
            else if (key == CHAT_STOP) fseek(file, value, SEEK_CUR); // ignore
            else if (key == PAD_ID) {} // ignore
            else {
                throw std::runtime_error("Invalid tokenizer header key:" + std::to_string(key));
            }
        }

        if (version != 1)
            throw std::runtime_error("Old tokenizer version, please regenerate your tokenizer");

        if (chatTemplateLength > 0) {
            chatTemplate = new char[chatTemplateLength + 1];
            if (fread(chatTemplate, chatTemplateLength, 1, file) != 1)
                throw std::runtime_error("Cannot read chat template from tokenizer file");
            chatTemplate[chatTemplateLength] = '\0';
        }
    } else {
        throw std::runtime_error("Invalid tokenizer file");
    }

    if (maxTokenLength < 1)
        throw std::runtime_error("Invalid tokenizer max token length");

    // malloc space to hold the scores and the strings
    vocab = new char*[vocabSize];
    vocabLength = new unsigned int[vocabSize];
    vocabScores = new float[vocabSize];

    int length;
    for (int i = 0; i < vocabSize; i++) {
        if (fread(vocabScores + i, sizeof(float), 1, file) != 1)
            throw std::runtime_error("Cannot read size from tokenizer file");
        if (fread(&length, sizeof(int), 1, file) != 1)
            throw std::runtime_error("Cannot read length from tokenizer file");
        vocab[i] = new char[length + 1];
        if (fread(vocab[i], length, 1, file) != 1)
            throw std::runtime_error("Cannot read word from tokenizer file");
        vocab[i][length] = '\0'; // add the string terminating token
        vocabLength[i] = length;
    }

    // TODO: this is very unstable assumption that bosId splits regular and special vocab
    regularVocabSize = bosId;
    specialVocabSize = vocabSize - regularVocabSize;

    regularVocab = new TokenIndex[regularVocabSize];
    for (int i = 0; i < regularVocabSize; i++) {
        regularVocab[i].str = vocab[i];
        regularVocab[i].id = i;
    }
    qsort(regularVocab, regularVocabSize, sizeof(TokenIndex), compareTokens);

    specialVocab = new TokenIndex[specialVocabSize];
    for (int i = 0; i < specialVocabSize; i++) {
        specialVocab[i].str = vocab[i + regularVocabSize];
        specialVocab[i].id = i + regularVocabSize;
    }

    strBufferSize = maxTokenLength * 2 + 1 + 2;
    strBuffer = new char[strBufferSize];

    if (bosId >= 0) printf("📄 BosId: %d (%s)\n", bosId, vocab[bosId]);
    if (eosTokenIds.size() > 0) {
        printf("📄 EosId: ");
        for (unsigned int i = 0; i < eosTokenIds.size(); i++) {
            printf("%d (%s) ", eosTokenIds[i], vocab[eosTokenIds[i]]);
        }
        printf("\n");
    }
    printf("📄 RegularVocabSize: %d\n", regularVocabSize);
    printf("📄 SpecialVocabSize: %d\n", specialVocabSize);

    fclose(file);
}

Tokenizer::Tokenizer(void* tokenizerData)
    : eosTokenIds() {
    bosId = -1;
    chatTemplate = nullptr;
    maxTokenLength = 0;

    // Use a pointer to read from memory instead of a file
    unsigned char* data = (unsigned char*)tokenizerData;
    size_t offset = 0;

    // Read the magic number
    int magic;
    std::memcpy(&magic, data + offset, sizeof(int));
    offset += sizeof(int);

    if (magic == 0x567123) {
        TokenizerOldHeader header;
        std::memcpy(&header, data + offset, sizeof(TokenizerOldHeader));
        offset += sizeof(TokenizerOldHeader);
        maxTokenLength = header.maxTokenLength;
        vocabSize = header.vocabSize;
        bosId = header.bosId;
        eosTokenIds.push_back(header.eosId);
    } else if (magic == 0x567124) {
        int headerSize;
        std::memcpy(&headerSize, data + offset, sizeof(int));
        offset += sizeof(int);
        int nKv = (headerSize - 2 * sizeof(int)) / sizeof(int);
        std::vector<int> buffer(nKv);
        std::memcpy(&buffer[0], data + offset, nKv * sizeof(int));
        offset += nKv * sizeof(int);

        int version = -1;
        int chatTemplateLength = -1;
        for (int i = 0; i < nKv; i += 2) {
            int key = buffer[i];
            int value = buffer[i + 1];
            if (key == TOK_VERSION) version = value;
            else if (key == TOK_VOCAB_SIZE) vocabSize = value;
            else if (key == MAX_TOKEN_LENGTH) maxTokenLength = (unsigned int)value;
            else if (key == BOS_ID) bosId = value;
            else if (key == EOS_ID) eosTokenIds.push_back(value);
            else if (key == CHAT_EOS_ID) eosTokenIds.push_back(value);
            else if (key == CHAT_TEMPLATE) chatTemplateLength = value;
            else if (key == CHAT_STOP) offset += value; // Skip
            else if (key == PAD_ID) {} // Ignore
            else {
                throw std::runtime_error("Invalid tokenizer header key:" + std::to_string(key));
            }
        }

        if (version != 1)
            throw std::runtime_error("Old tokenizer version, please regenerate your tokenizer");

        if (chatTemplateLength > 0) {
            chatTemplate = new char[chatTemplateLength + 1];
            std::memcpy(chatTemplate, data + offset, chatTemplateLength);
            offset += chatTemplateLength;
            chatTemplate[chatTemplateLength] = '\0';
        }
    } else {
        throw std::runtime_error("Invalid tokenizer data");
    }

    if (maxTokenLength < 1)
        throw std::runtime_error("Invalid tokenizer max token length");

    // Allocate space for vocab, scores, and lengths
    vocab = new char*[vocabSize];
    vocabLength = new unsigned int[vocabSize];
    vocabScores = new float[vocabSize];

    int length;
    for (int i = 0; i < vocabSize; i++) {
        // Read score
        std::memcpy(vocabScores + i, data + offset, sizeof(float));
        offset += sizeof(float);
        // Read length
        std::memcpy(&length, data + offset, sizeof(int));
        offset += sizeof(int);
        // Read token string
        vocab[i] = new char[length + 1];
        std::memcpy(vocab[i], data + offset, length);
        offset += length;
        vocab[i][length] = '\0';
        vocabLength[i] = length;
    }

    // Set up regular and special vocab
    regularVocabSize = bosId;
    specialVocabSize = vocabSize - regularVocabSize;

    regularVocab = new TokenIndex[regularVocabSize];
    for (int i = 0; i < regularVocabSize; i++) {
        regularVocab[i].str = vocab[i];
        regularVocab[i].id = i;
    }
    qsort(regularVocab, regularVocabSize, sizeof(TokenIndex), compareTokens);

    specialVocab = new TokenIndex[specialVocabSize];
    for (int i = 0; i < specialVocabSize; i++) {
        specialVocab[i].str = vocab[i + regularVocabSize];
        specialVocab[i].id = i + regularVocabSize;
    }

    strBufferSize = maxTokenLength * 2 + 1 + 2;
    strBuffer = new char[strBufferSize];

    if (bosId >= 0) printf("📄 BosId: %d (%s)\n", bosId, vocab[bosId]);
    if (eosTokenIds.size() > 0) {
        printf("📄 EosId: ");
        for (unsigned int i = 0; i < eosTokenIds.size(); i++) {
            printf("%d (%s) ", eosTokenIds[i], vocab[eosTokenIds[i]]);
        }
        printf("\n");
    }
    printf("📄 RegularVocabSize: %d\n", regularVocabSize);
    printf("📄 SpecialVocabSize: %d\n", specialVocabSize);
}

Tokenizer::~Tokenizer() {
    for (unsigned int i = 0; i < vocabSize; i++) {
        delete[] vocab[i];
    }
    delete[] vocab;
    delete[] vocabScores;
    delete[] vocabLength;
    delete[] regularVocab;
    delete[] specialVocab;
    delete[] strBuffer;
    delete[] chatTemplate;
}

// Placeholder for other Tokenizer methods (not shown in the original snippet)
int Tokenizer::findSpecialTokenStartWith(char *piece) {
    // Implementation would be here
    return -1; // Placeholder
}

int Tokenizer::findRegularToken(char *piece) {
    // Implementation would be here
    return -1; // Placeholder
}

void Tokenizer::encode(char *text, int *tokens, int *nTokens, bool addBos, bool addSpecialTokens) {
    // Implementation would be here
}

bool Tokenizer::isEos(int token) {
    for (int eosId : eosTokenIds) {
        if (token == eosId) return true;
    }
    return false;
}

char *Tokenizer::decode(int token) {
    // Implementation would be here
    return nullptr; // Placeholder
}

void Tokenizer::resetDecoder() {
    // Implementation would be here
}

// Rest of the file (Sampler, TokenizerChatStops, etc.) would follow...
// Since the full file wasn't provided, these are assumed to be unchanged
unsigned int randomU32(unsigned long long *state) {
    // xorshift rng: https://en.wikipedia.org/wiki/Xorshift#xorshift.2A
    *state ^= *state >> 12;
    *state ^= *state << 25;
    *state ^= *state >> 27;
    return (*state * 0x2545F4914F6CDD1Dull) >> 32;
}

float randomF32(unsigned long long *state) {
    // random float32 in <0,1)
    return (randomU32(state) >> 8) / 16777216.0f;
}

// Sampler implementation (assumed unchanged)
Sampler::Sampler(int vocab_size, float temperature, float topp, unsigned long long rngSeed)
    : vocab_size(vocab_size), temperature(temperature), topp(topp), rngState(rngSeed) {
    probindex = new ProbIndex[vocab_size];
}

Sampler::~Sampler() {
    delete[] probindex;
}

int Sampler::sample(float *logits) {
    // Placeholder implementation
    return 0;
}

void Sampler::setTemp(float temp) {
    temperature = temp;
}

void Sampler::setSeed(unsigned long long rngSeed) {
    rngState = rngSeed;
}

// TokenizerChatStops implementation (assumed unchanged)
TokenizerChatStops::TokenizerChatStops(Tokenizer *tokenizer) {
    // Placeholder implementation
    stops = nullptr;
    nStops = 0;
    maxStopLength = 0;
}

TokenizerChatStops::~TokenizerChatStops() {
    // Placeholder implementation
}

// ChatTemplateGenerator implementation (assumed unchanged)
ChatTemplateGenerator::ChatTemplateGenerator(const ChatTemplateType type, const char *chatTemplate, const char *eos)
    : type(type), eos(eos) {
    // Placeholder implementation
}

GeneratedChat ChatTemplateGenerator::generate(unsigned int nItems, ChatItem *items, bool appendGenerationPrompt) {
    GeneratedChat result = {nullptr, 0, nullptr};
    // Placeholder implementation
    return result;
}

// EosDetector implementation (assumed unchanged)
EosDetector::EosDetector(size_t nTokens, const int *tokens, const char* *pieces, int paddingLeft, int paddingRight)
    : nTokens(nTokens), tokens(tokens), pieces(pieces), paddingLeft(paddingLeft), paddingRight(paddingRight) {
    // Placeholder implementation
    buffer = nullptr;
    bufferPos = 0;
    bufferSize = 0;
    eosPos = -1;
}

EosDetector::~EosDetector() {
    // Placeholder implementation
}

EosDetectorType EosDetector::append(int tokenId, const char *piece) {
    // Placeholder implementation
    return NOT_EOS;
}

bool EosDetector::isEos(int tokenId) {
    // Placeholder implementation
    return false;
}

char *EosDetector::getDelta() {
    // Placeholder implementation
    return nullptr;
}

void EosDetector::reset() {
    // Placeholder implementation
}