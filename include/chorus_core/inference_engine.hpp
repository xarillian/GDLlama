#pragma once

#include "chorus_common.hpp"

namespace Chorus {

    class InferenceEngine {
    public:
        virtual ~InferenceEngine() = default;

        virtual bool initialize(const Chorus::ChorusConfig& config) = 0;
        virtual bool is_initialized() const = 0;

        virtual void submit_request(const Chorus::ChorusRequest& chorus_request) = 0;

        virtual void stop() = 0;
        
    };

}