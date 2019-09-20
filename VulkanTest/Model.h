#pragma once

#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/glm.hpp>

#include <vector>

#include "Vertex.h"

class Model
{
  private:
    std::string mesh_path    = "models/chalet.obj";
    std::string texture_path = "textures/chalet.jpg";

    uint32_t mipLevels;

    VkDevice* device;

    // Move these to RenderGroup ...
    // VkBuffer       vertexBuffer;
    // VkDeviceMemory vertexBufferMemory;
    // VkBuffer       indexBuffer;
    // VkDeviceMemory indexBufferMemory;

    // how the fuck is this handled?
    VkImage        textureImage;
    VkDeviceMemory textureImageMemory;
    VkImageView    textureImageView;
    // ...

    // TODO: @MaxCompleteAPI, this likely shouldn't be here and if it needs to be it should be dumped after loading
    // put this as a temp variable in loading then dump it once uploaded?
    // maybe not possible when we are recreating RenderGroups and need to reupload these to the buffer
    //std::vector<Vertex>   vertices;
    //std::vector<uint32_t> indices;  // It is possible to use either uint16_t or uint32_t for your index buffer depending on
                                    // the number of entries in vertices. We can stick to uint16_t for now because we're
                                    // using less than 65535 unique vertices.
    size_t verticesOffset;
    size_t verticesCount;
    size_t indicesOffset;
    size_t indicesCount;

    std::vector<glm::mat4> _modelMatrices;

    bool cleaned_up = false;
    
    friend class VulkanApplication;
    friend class RenderGroup;
    
  public:
    ~Model()
    {
        // TODO: @MaxConcurrency, make this a compareExchange???
        // not sure if necessary though cause destructors might be thread-safe in the standard??
        if (!cleaned_up)
        {
            // vkDestroyImageView(*device, textureImageView, nullptr);

            // vkDestroyImage(*device, textureImage, nullptr);
            // vkFreeMemory(*device, textureImageMemory, nullptr);

            // vkDestroyBuffer(*device, indexBuffer, nullptr);
            // vkFreeMemory(*device, indexBufferMemory, nullptr);

            // vkDestroyBuffer(*device, vertexBuffer, nullptr);
            // vkFreeMemory(*device, vertexBufferMemory, nullptr);  // used by buffer, deleted after it

            cleaned_up = true;
        }
    }
};

class RenderGroup
{
  public:
    // TODO: @MaxCompleteAPI, don't think this is necessary
    std::string name;

	// TODO: @MaxCompleteAPI, allow option to cull entire groups
  private:
    std::vector<Model>           models;
    std::vector<VkCommandBuffer> commandBuffers;  // size of frames_in_flight
    VkDevice*                    device;
    VkCommandPool* commandPool;  // this can change based on which thread calls rebuildCommandBuffers as this is essentially
                                 // a per thread allocator for GPU memory and commandBuffers

	std::vector<Vertex>   vertices;
    std::vector<uint32_t> indices;// It is possible to use either uint16_t or uint32_t for your index buffer depending on
                                    // the number of entries in vertices. We can stick to uint16_t for now because we're
                                    // using less than 65535 unique vertices.

	int totalInstanceCount;

    bool dynamic = false;  // rebuilt often

    VkBuffer       vertexBuffer;
    VkDeviceMemory vertexBufferMemory;
    VkBuffer       indexBuffer;
    VkDeviceMemory indexBufferMemory;

    std::vector<VkBuffer>       instanceDataBuffer;
    std::vector<VkDeviceMemory> instanceDataBufferMemory;
    
    VkPipeline graphicsPipeline;
    VkDescriptorSetLayout descriptorSetLayout;
    
    VkDescriptorPool*            descriptorPool;
    std::vector<VkDescriptorSet> descriptorSets;

    // VkImage        textureImage;
    // VkDeviceMemory textureImageMemory;
    // VkImageView    textureImageView;

    VkSampler textureSampler;

    bool cleaned_up = false;
    
    friend class VulkanApplication;
    
  public:
    ~RenderGroup()
    {
        if (!cleaned_up)
        {
            for (auto i = 0; i < models.size(); i++)
            {
                models[i].~Model();
            }

            vkFreeCommandBuffers(*device, *commandPool, static_cast<uint32_t>(commandBuffers.size()), commandBuffers.data());

            cleaned_up = true;
        }
    }
};

// TODO: @MaxCompleteAPI, rename private members to _variable and functions

// TODO: @MaxCompleteAPI, add debug only or optional checks
