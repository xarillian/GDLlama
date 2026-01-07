#pragma once

#include "chorus_core/chorus_common.hpp"
#include "chorus_core/inference_engine.hpp"

#include <memory>

class LlamaScheduler;

namespace Chorus {
    class LlamaEngine : public InferenceEngine {
        public:
            LlamaEngine();
            ~LlamaEngine();

            bool initialize(const Chorus::ChorusConfig& config) override;
            void submit_request(const Chorus::ChorusRequest& chorus_request) override;
            void stop() override;
            bool is_initialized() const override;

        private:
            std::unique_ptr<LlamaScheduler> scheduler;
            bool _initialized = false;
    };
}