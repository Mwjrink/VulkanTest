#pragma once

#include <vulkan/vulkan.h>

#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/hash.hpp>

#include <array>

struct InstanceData
{
    glm::mat4 projViewModel;
    uint32_t textureIndex;

	bool operator==(const InstanceData& other) const
    {
        return projViewModel == other.projViewModel && textureIndex == other.textureIndex;
    }

    static VkVertexInputBindingDescription getBindingDescription(uint32_t binding)
    {
        VkVertexInputBindingDescription bindingDescription = {};
        bindingDescription.binding                         = binding;
        bindingDescription.stride                          = sizeof(InstanceData);
        bindingDescription.inputRate                       = VK_VERTEX_INPUT_RATE_INSTANCE;

        // inputRate parameter can have one of the following values:
        // VK_VERTEX_INPUT_RATE_VERTEX:   // Move to the next data entry after each vertex
        // VK_VERTEX_INPUT_RATE_INSTANCE: // Move to the next data entry after each instance

        return bindingDescription;
    }

    static void getAttributeDescriptions(uint32_t binding, std::vector<VkVertexInputAttributeDescription> &attributeDescriptions)
    {
        int index = attributeDescriptions.size();
        attributeDescriptions.push_back({});
        attributeDescriptions[index].binding  = binding;
        attributeDescriptions[index].location = index;
        attributeDescriptions[index].format   = VK_FORMAT_R32G32B32A32_SFLOAT;
        attributeDescriptions[index].offset   = offsetof(InstanceData, projViewModel);

        // The format parameter describes the type of data for the attribute. A bit confusingly, the formats are specified
        // using the same enumeration as color formats. The following shader types and formats are commonly used together:
        // float: VK_FORMAT_R32_SFLOAT
        // vec2: VK_FORMAT_R32G32_SFLOAT
        // vec3: VK_FORMAT_R32G32B32_SFLOAT
        // vec4: VK_FORMAT_R32G32B32A32_SFLOAT

        // ivec2: VK_FORMAT_R32G32_SINT, a 2-component vector of 32-bit signed integers
        // uvec4: VK_FORMAT_R32G32B32A32_UINT, a 4-component vector of 32-bit unsigned integers
        // double: VK_FORMAT_R64_SFLOAT, a double-precision (64-bit) float
        index = attributeDescriptions.size();
        attributeDescriptions.push_back({});
        attributeDescriptions[index].binding  = binding;
        attributeDescriptions[index].location = index;
        attributeDescriptions[index].format   = VK_FORMAT_R8_UINT;
        attributeDescriptions[index].offset   = offsetof(InstanceData, textureIndex);
    }
};
