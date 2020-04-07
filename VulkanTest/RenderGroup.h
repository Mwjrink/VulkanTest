#pragma once

#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/glm.hpp>
#include <string>
#include <vector>

#include "Model.h"
#include "observer_ptr.h"

class _RenderGroup
{
  private:
    std::vector<_Model>              _models;
    std::vector<VkCommandBuffer>     _commandBuffers;  // size of frames_in_flight
    std::observer_ptr<VkDevice>      _device;

    // this can change based on which thread calls rebuildCommandBuffers as this is essentially
    // a per thread allocator for GPU memory and commandBuffers
    std::observer_ptr<VkCommandPool> _commandPool;

    std::vector<Vertex>   _vertices;
    std::vector<uint32_t> _indices;  // It is possible to use either uint16_t or uint32_t for your index buffer depending on
                                    // the number of entries in vertices. We can stick to uint16_t for now because we're
                                    // using less than 65535 unique vertices.

    int _totalInstanceCount;

    bool _dynamic = false;  // rebuilt often

    VkBuffer       _vertexBuffer;
    VkDeviceMemory _vertexBufferMemory;
    // VkDeviceSize   vertexBufferSize;
    VkBuffer       _indexBuffer;
    VkDeviceMemory _indexBufferMemory;
    // VkDeviceSize   indexBufferSize;

    VkBuffer       _instanceDataStagingBuffer;
    VkDeviceMemory _instanceDataStagingBufferMemory;
    // VkDeviceSize   instanceDataStagingBufferSize;

    std::vector<VkBuffer>       _instanceDataBuffer;
    std::vector<VkDeviceMemory> _instanceDataBufferMemory;
    // std::vector<VkDeviceSize>   instanceDataBufferSize;
    VkDeviceSize _instanceDataBufferSize;

    VkPipeline            _graphicsPipeline;
    VkDescriptorSetLayout _descriptorSetLayout;

    VkDescriptorPool*            _descriptorPool;
    std::vector<VkDescriptorSet> _descriptorSets;

    // VkImage        textureImage;
    // VkDeviceMemory textureImageMemory;
    // VkImageView    textureImageView;

    VkSampler _textureSampler;

    bool _cleaned_up = false;

    friend class VulkanApplication;

    _RenderGroup(std::observer_ptr<VkDevice> device, std::observer_ptr<VkCommandPool> commandPool,
                 const int max_frames_in_flight)
    {
        _device      = device;
        _commandPool = commandPool;
        _commandBuffers.resize(max_frames_in_flight);
        _instanceDataBuffer.resize(max_frames_in_flight);
        _instanceDataBufferMemory.resize(max_frames_in_flight);
    }

  public:
    void unload() {}

    void unloadFromGpu() {}

    void unloadFromRam() {}

    ~_RenderGroup()
    {
        if (!_cleaned_up)
        {
            for (auto i = 0; i < _models.size(); i++)
            {
                _models[i].~_Model();
            }

            vkFreeCommandBuffers(*_device, *_commandPool, static_cast<uint32_t>(_commandBuffers.size()), _commandBuffers.data());

            _cleaned_up = true;
        }
    }
};

// TODO: @MaxCompleteAPI, rename private members to _variable and functions

// TODO: @MaxCompleteAPI, add debug only or optional checks
